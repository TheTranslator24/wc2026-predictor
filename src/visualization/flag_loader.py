#!/usr/bin/env python3
# ==============================================================
# FILE: src/visualization/flag_loader.py
# PURPOSE: Download real national-flag PNGs once and cache them locally,
#          so charts can show actual flags instead of emoji "tofu" boxes.
#
# LEGAL NOTE: national FLAG DESIGNS are public domain — free to use. This
# does NOT download team crests/logos (those are trademarked and excluded).
# Source: flagcdn.com, a free flag CDN serving public-domain flag images.
#
# RUN ONCE (on your Mac, internet required):
#   python3 src/visualization/flag_loader.py
#
# It writes PNGs to assets/flags/<iso>.png. Charts auto-detect them; if you
# skip this step, charts gracefully fall back to color bars + ISO labels.
# ==============================================================

import logging
import sys
import time
from pathlib import Path

import requests

# Allow running this file directly (python3 src/visualization/flag_loader.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.visualization.team_identity import TEAMS, FLAG_DIR

logger = logging.getLogger("wc2026.visualization.flags")

# flagcdn serves flags at a fixed width; w160 is crisp enough for chart icons.
FLAG_BASE = "https://flagcdn.com/w160/{iso}.png"


def download_all_flags(force: bool = False) -> int:
    """
    Download a flag PNG for every team's ISO code into assets/flags/.

    Args:
        force: re-download even if a cached file already exists.
    Returns:
        count of flags now present on disk.
    """
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "WC2026-Predictor/1.0 (flags)"})

    present = 0
    for team, meta in TEAMS.items():
        code = meta["iso"]
        dest = FLAG_DIR / f"{code}.png"
        if dest.exists() and not force:
            present += 1
            continue
        url = FLAG_BASE.format(iso=code)
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            dest.write_bytes(r.content)        # PNG bytes straight to disk
            present += 1
            print(f"  + {code:<6} {team}")
            time.sleep(0.1)                    # be polite to the free CDN
        except requests.RequestException as e:
            print(f"  - {code:<6} {team}  (skipped: {e})")

    print(f"\n{present}/{len(TEAMS)} flags available in {FLAG_DIR}")
    return present


if __name__ == "__main__":
    print("Downloading public-domain national flags (no logos/crests)...")
    download_all_flags()
