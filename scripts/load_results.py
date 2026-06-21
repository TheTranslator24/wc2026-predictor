#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/load_results.py
# PURPOSE: Record EVERY match result in one go. Fill data/results_batch.csv
#          (one row per played match, any/all nations), then run this once.
#
# For each row it does two things:
#   1. add_result(...)              -> feeds the score into your Elo ratings
#   2. record_actual_from_score(...) -> grades the prediction you logged for it
#      (skipped, with a note, if you never logged a prediction for that match)
#
# WHY A BATCH FILE: typing results one at a time in the terminal is slow and
# error-prone (you saw the quoting/heredoc pain). A CSV you edit in VS Code and
# load in one command scales to all 104 matches cleanly.
#
# USAGE (from project root, venv active):
#     python3 scripts/load_results.py
# ==============================================================

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.tracking.results_store import add_result
from src.tracking.scorer import record_actual_from_score

BATCH = DATA_DIR / "results_batch.csv"


def main():
    if not BATCH.exists():
        print(f"No batch file at {BATCH}. Create it (see the template) and add rows.")
        return

    # comment="#" lets the instruction lines at the top be ignored.
    df = pd.read_csv(BATCH, comment="#", dtype={"home_team": str, "away_team": str, "stage": str})
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    if df.empty:
        print("No result rows found in results_batch.csv yet — add some below the header.")
        return

    fed, graded, ungraded = 0, 0, 0
    print(f"Loading {len(df)} result(s) from results_batch.csv...\n")
    for _, r in df.iterrows():
        home, away = r["home_team"].strip(), r["away_team"].strip()
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        date, stage = str(r["date"]).strip(), str(r.get("stage", "FIFA World Cup")).strip()
        try:
            # 1) always feed Elo (this never needs a prior prediction)
            add_result(home, away, hs, as_, date, stage or "FIFA World Cup")
            fed += 1
            # 2) grade the logged prediction if one exists
            try:
                record_actual_from_score(home, away, date, hs, as_)
                graded += 1
                print(f"  graded  | {home} {hs}-{as_} {away}  ({date})")
            except ValueError:
                ungraded += 1
                print(f"  elo-only| {home} {hs}-{as_} {away}  ({date})  [no prediction was logged]")
        except Exception as e:
            print(f"  SKIPPED | {home} vs {away} ({date}): {e}")

    print(f"\nDone. Elo updated from {fed} result(s) | {graded} graded | "
          f"{ungraded} had no logged prediction.")
    print("Check calibration with: python3 -c \"from src.tracking.scorer import print_scorecard; print_scorecard()\"")


if __name__ == "__main__":
    main()
