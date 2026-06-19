#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/predict_group_stage.py
# PURPOSE: Generate a prediction chart (.png) for EVERY group-stage match —
#          all 72 of them (12 groups x 6 matches) — in a single run.
#
# WHY A SCRIPT (not MATCHES_TO_PREDICT): hand-typing 72 fixtures is tedious and
# error-prone. The 72 pairings are fully determined by the groups (each team
# plays the other three in its group), so we enumerate them straight from
# config — guaranteed complete and correct.
#
# USAGE (from the project root, venv active):
#   python3 scripts/predict_group_stage.py
#
# OUTPUT: outputs/predictions/GroupX_Home_vs_Away.png  (72 files)
# Takes a few minutes: models load once, then ~1-2s per match + chart.
# ==============================================================

import sys
import logging
from pathlib import Path

# Make the project root importable no matter where you launch from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import WC2026_GROUPS, OUTPUTS_DIR
from src.predictor.match_predictor import WC2026Predictor
from src.visualization.probability_charts import plot_match_prediction

logger = logging.getLogger("wc2026.group_stage")


def all_group_fixtures() -> list[tuple[str, str, str]]:
    """Every within-group pairing as (group_letter, home, away). 72 total."""
    fixtures = []
    for group, teams in WC2026_GROUPS.items():
        for i, home in enumerate(teams):          # each team...
            for away in teams[i + 1:]:            # ...vs every later team in its group
                fixtures.append((group, home, away))
    return fixtures


def main():
    print("=" * 56)
    print("  WC2026 — FULL GROUP STAGE  (all 72 match charts)")
    print("=" * 56)

    # Build once; reuses saved models (XGBoost + LSTM if trained).
    predictor = WC2026Predictor(auto_load=True).setup()

    fixtures = all_group_fixtures()
    out_dir = OUTPUTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    done, failed = 0, 0
    for n, (group, home, away) in enumerate(fixtures, 1):
        try:
            pred = predictor.predict(home, away)              # ensemble prediction
            # Save with a group-prefixed name so they sort by group in the folder.
            fname = f"Group{group}_{home.replace(' ', '_')}_vs_{away.replace(' ', '_')}.png"
            plot_match_prediction(pred, save_path=out_dir / fname, show=False)
            done += 1
            # compact progress line
            print(f"  [{n:2d}/72] Group {group}: {home} vs {away:18s} -> "
                  f"{pred['predicted_outcome']:9s} ({pred['confidence']:.0%})")
        except Exception as e:
            failed += 1
            logger.error(f"Failed {home} vs {away}: {e}")
            print(f"  [{n:2d}/72] Group {group}: {home} vs {away} -> SKIPPED ({e})")

    print("=" * 56)
    print(f"  Done. {done} charts saved to {out_dir}" + (f" | {failed} failed" if failed else ""))
    print("=" * 56)


if __name__ == "__main__":
    main()
