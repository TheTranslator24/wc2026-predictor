# ==============================================================
# FILE: src/security/integrity_check.py
# PURPOSE: Prove a data file on disk is byte-for-byte the same file
#          you previously verified and trusted — using SHA-256.
#
# THREAT MODEL (what this defends against):
#   - A truncated / partially downloaded CSV (accidental corruption)
#   - A silently swapped or edited data file (tampering)
#   Either one would skew every prediction with no visible error.
#   A hash mismatch turns a SILENT failure into a LOUD, refusable one.
#
# WORKFLOW:
#   1. First clean download  -> compute hash -> store as the baseline
#   2. Every later load      -> recompute    -> compare to baseline
#   3. Mismatch              -> reject the file, log CRITICAL, re-download
#
# WHY SHA-256: collision resistance ~2^128 work. This guarantees
# integrity (the file is unchanged). It is NOT confidentiality and
# NOT authentication of a remote server — TLS in collector.py does
# the transport authentication. Different security goals, layered.
# ==============================================================

import hashlib          # stdlib: cryptographic hash functions (no install)
import hmac             # stdlib: provides compare_digest (constant-time compare)
import json             # stdlib: read/write the checksum registry
import logging          # stdlib: the audit trail
from pathlib import Path

logger = logging.getLogger("wc2026.security.integrity")

# Read in 8 KB blocks so a 50 MB CSV never loads fully into RAM.
_CHUNK_SIZE = 8192


def compute_sha256(filepath: Path) -> str:
    """
    Compute the SHA-256 hash of any file, streaming it in chunks.

    Streaming (vs read-all) keeps memory flat regardless of file size —
    a memory-safety property that matters on a 16 GB laptop and is
    good hygiene generally.

    Args:
        filepath: file to hash
    Returns:
        64-character lowercase hex digest, e.g. "a3f2b8c1...".
    Raises:
        FileNotFoundError / PermissionError: if the file can't be read.
    """
    hasher = hashlib.sha256()              # fresh, empty hash state
    with open(filepath, "rb") as f:        # "rb" = binary; correct for any file
        # The walrus ':=' reads a chunk AND tests it in one expression.
        # Loop ends when read() returns empty bytes (b'') at EOF.
        while chunk := f.read(_CHUNK_SIZE):
            hasher.update(chunk)           # fold each chunk into the running hash
    digest = hasher.hexdigest()
    logger.info(f"SHA-256 computed | {filepath.name} | {digest[:16]}...")
    return digest


def load_checksum_registry(registry_path: Path) -> dict[str, str]:
    """
    Load the trusted {filename: sha256} map from a JSON file.

    Returns an EMPTY dict on first run (no registry yet) — that is a
    normal state, not an error.
    """
    if not registry_path.exists():
        logger.info(
            f"No checksum registry at {registry_path.name} — "
            "one will be created after the first download."
        )
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    logger.info(f"Checksum registry loaded | entries={len(registry)}")
    return registry


def save_checksum_registry(registry: dict[str, str], registry_path: Path) -> None:
    """
    Write the {filename: sha256} map to disk as JSON.

    This file is SAFE (and useful) to commit: it contains only
    filenames and hashes. Committing it lets anyone who clones the
    repo verify they fetched identical data. sort_keys keeps Git
    diffs clean and deterministic.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    logger.info(
        f"Checksum registry saved | entries={len(registry)} | {registry_path.name}"
    )


def register_file(filepath: Path, registry_path: Path) -> str:
    """
    Hash a freshly downloaded file and record it as the trusted baseline.

    Call this ONCE after a known-clean download. Thereafter use
    verify_file() to detect any change.

    Returns: the hash that was registered.
    """
    file_hash = compute_sha256(filepath)
    registry = load_checksum_registry(registry_path)
    # Key on the bare filename (not the absolute path) so the registry
    # is portable between machines / clones.
    registry[filepath.name] = file_hash
    save_checksum_registry(registry, registry_path)
    logger.info(f"File registered as trusted | {filepath.name} | {file_hash[:16]}...")
    return file_hash


def verify_file(filepath: Path, registry_path: Path) -> bool:
    """
    Verify a file against the trusted registry.

    Behavior:
      - In registry      -> recompute and compare. True if equal.
      - Not in registry  -> trust-on-first-use: register it now, True.
      - Missing file     -> FileNotFoundError.

    SECURITY DETAIL — constant-time comparison:
      We compare the two hex digests with hmac.compare_digest() rather
      than ==. A plain == can short-circuit on the first differing byte;
      compare_digest takes the same time regardless, removing a timing
      side-channel. For a local football CSV the risk is negligible, but
      using the constant-time primitive for ALL secret/digest comparisons
      is the correct habit to build — so we do it here.

    Returns:
        True  -> file verified clean (or trusted on first use)
        False -> hash mismatch: corrupted or tampered, do NOT use it.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot verify — file not found: {filepath}")

    registry = load_checksum_registry(registry_path)
    stored_hash = registry.get(filepath.name)

    # Trust-on-first-use: no baseline yet, so establish one now.
    if stored_hash is None:
        logger.warning(
            f"No baseline for {filepath.name} — registering it as trusted now."
        )
        register_file(filepath, registry_path)
        return True

    actual_hash = compute_sha256(filepath)

    # Constant-time compare of the two 64-char hex strings.
    if hmac.compare_digest(actual_hash, stored_hash):
        logger.info(f"Integrity VERIFIED | {filepath.name}")
        return True

    # Loud, explicit failure — the whole point of this module.
    logger.critical(
        f"INTEGRITY FAILURE | {filepath.name}\n"
        f"  expected: {stored_hash[:32]}...\n"
        f"  actual:   {actual_hash[:32]}...\n"
        f"  ACTION:   treat as corrupted/tampered; delete and re-download."
    )
    return False
