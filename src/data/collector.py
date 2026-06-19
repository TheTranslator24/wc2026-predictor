# ==============================================================
# FILE: src/data/collector.py
# PURPOSE: Download and cache football data with a full security chain.
#
# SECURITY CHAIN (every download passes ALL of these, in order):
#   1. URL whitelist check     — only URLs declared in DATA_SOURCES
#   2. HTTPS + SSL cert verify  — certifi (Mozilla) root bundle
#   3. Request timeout          — never hangs forever
#   4. Streaming download        — body read in 8 KB chunks, memory-flat
#   5. File-size limit           — rejects unexpectedly large payloads
#   6. SHA-256 integrity check   — file must match the trusted baseline
#   7. Schema validation         — DataFrame has expected columns + rows
#   8. Full audit logging        — every step timestamped to predictor.log
#
# DATA SOURCE (CORRECTED):
#   martj42/international_results  (note the underscore)
#   License: CC0 Public Domain | ~49,000 matches | 1872–2026
#   https://github.com/martj42/international_results
# ==============================================================

import logging
import time
from pathlib import Path

import pandas as pd
import requests
import certifi   # Mozilla CA bundle — used to validate every HTTPS cert

from src.config import DATA_SOURCES, SECURITY_CONFIG, RAW_DATA_DIR
from src.security.integrity_check import (
    load_checksum_registry,
    register_file,
    verify_file,
)
from src.security.data_validator import validate_dataframe, RESULTS_REQUIRED_COLS

logger = logging.getLogger("wc2026.data.collector")


class SecureDataCollector:
    """
    Downloads and loads historical football data under strict controls.

    Guarantees on every download:
      - URL is on the pre-approved whitelist (no user-supplied URLs)
      - TLS certificate is verified (verify is NEVER disabled)
      - Size is capped (default 50 MB)
      - SHA-256 matches the trusted baseline
      - Schema is validated before the caller sees the data
    """

    def __init__(self) -> None:
        # A Session reuses the TCP connection across downloads (faster)
        # and applies shared TLS settings + headers to every request.
        self.session = requests.Session()

        # SECURITY: pin verification to the certifi bundle. This is more
        # consistent than the OS store across machines. NEVER set
        # verify=False — that would disable certificate validation.
        self.session.verify = certifi.where()

        # Identify the client honestly (no browser impersonation).
        self.session.headers.update({
            "User-Agent":      "WC2026-Predictor/1.0 (Research)",
            "Accept":          "text/csv,text/plain,application/json",
            "Accept-Encoding": "gzip, deflate",
        })

        # checksums.json lives beside data/ (one level up from raw/).
        self.registry_path = RAW_DATA_DIR.parent / "checksums.json"

        # Pre-compute the URL whitelist as a frozenset for O(1) checks.
        self._allowed_urls: frozenset[str] = frozenset(
            src["url"] for src in DATA_SOURCES.values()
        )

        logger.info(
            f"SecureDataCollector ready | SSL=certifi | "
            f"whitelisted_urls={len(self._allowed_urls)}"
        )

    # ──────────────────────────────────────────────────────────
    def _download_with_security(self, url: str, dest: Path) -> Path:
        """
        Download one file with the full security chain. INTERNAL ONLY —
        never call this with a URL that came from user input.
        """
        # ── Guard 1: URL whitelist ─────────────────────────────
        if url not in self._allowed_urls:
            logger.critical(f"BLOCKED non-whitelisted URL: {url}")
            raise ValueError(f"URL not in approved whitelist: {url}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = SECURITY_CONFIG["max_file_size_mb"] * 1024 * 1024
        timeout   = SECURITY_CONFIG["request_timeout_seconds"]
        retries   = SECURITY_CONFIG["max_retries"]

        logger.info(f"Download starting | {dest.name} | {url[:60]}...")

        for attempt in range(1, retries + 1):
            try:
                # ── Guard 2+3: TLS verify + streaming + timeout ───
                response = self.session.get(url, stream=True, timeout=timeout)
                response.raise_for_status()   # raise on any 4xx/5xx

                # ── Guard 4: reject if server advertises an oversized body ─
                content_len = response.headers.get("Content-Length")
                if content_len and int(content_len) > max_bytes:
                    raise ValueError(
                        f"Server reports {int(content_len)/1e6:.1f} MB, "
                        f"over the {SECURITY_CONFIG['max_file_size_mb']} MB limit"
                    )

                # ── Guard 5: stream to disk, enforcing the cap live ─
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            f.close()
                            dest.unlink(missing_ok=True)   # remove partial file
                            raise ValueError(
                                f"Download exceeded size limit mid-stream: "
                                f"{downloaded/1e6:.1f} MB"
                            )
                        f.write(chunk)

                logger.info(
                    f"Download complete | {dest.name} | "
                    f"{downloaded/1024:.1f} KB | attempt {attempt}/{retries}"
                )
                break   # success — leave the retry loop

            except requests.exceptions.SSLError as exc:
                # SECURITY: a TLS failure means an untrustworthy connection.
                # Never swallow it, never retry around it — surface it.
                logger.critical(f"SSL verification FAILED for {url}: {exc}")
                raise

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt}/{retries}")
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)   # backoff: 2s, 4s, 8s

            except requests.exceptions.RequestException as exc:
                logger.error(f"Network error (attempt {attempt}/{retries}): {exc}")
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)

        # ── Guard 6: SHA-256 integrity ─────────────────────────
        registry = load_checksum_registry(self.registry_path)
        if registry.get(dest.name) is None:
            register_file(dest, self.registry_path)            # first time: trust
            logger.info(f"New file registered as trusted baseline: {dest.name}")
        elif not verify_file(dest, self.registry_path):        # later: must match
            dest.unlink(missing_ok=True)                       # quarantine bad file
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: {dest.name} does not match the "
                f"trusted hash. File deleted — re-download to retry."
            )

        return dest

    # ──────────────────────────────────────────────────────────
    def load_historical_results(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load the international results dataset (downloads on first use).

        Subsequent calls load from local cache but STILL re-verify the
        SHA-256, so a silently altered cache is caught.

        force_download=True re-fetches even when a cache exists — use this
        DAILY during the tournament to pick up newly added results.
        """
        source     = DATA_SOURCES["historical_results"]
        local_path = source["local_path"]

        if local_path.exists() and not force_download:
            logger.info(f"Cache found: {local_path.name} — verifying integrity...")
            if not verify_file(local_path, self.registry_path):
                logger.warning("Cache integrity failed — re-downloading...")
                self._download_with_security(source["url"], local_path)
        else:
            logger.info("Downloading historical results...")
            self._download_with_security(source["url"], local_path)

        # Explicit dtypes stop pandas from guessing column types wrong.
        df = pd.read_csv(
            local_path,
            dtype={
                "home_team":  str,
                "away_team":  str,
                "tournament": str,
                "city":       str,
                "country":    str,
            },
            parse_dates=["date"],
        )

        # ── Guard 7: schema + size validation (from the security layer) ─
        df = validate_dataframe(
            df,
            required_columns=RESULTS_REQUIRED_COLS,
            min_rows=10_000,        # we expect ~49k; 10k is a generous floor
            name="international_results",
        )

        # Coerce scores to numeric; voided/abandoned matches become NaN.
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

        logger.info(
            f"Historical results ready | rows={len(df):,} | "
            f"date_range={df['date'].min().year}–{df['date'].max().year}"
        )
        return df

    def __del__(self):
        # Best-effort cleanup of the TCP session at garbage-collection.
        try:
            self.session.close()
        except Exception:
            pass
