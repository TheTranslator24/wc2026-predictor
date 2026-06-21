#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/sync_results.py
# PURPOSE: Pull the latest match results straight from the SOURCE dataset on
#          GitHub (maintained by others, updated as games are played) — so you
#          DON'T hand-type scores. It then grades every forecast you logged.
#
# WHAT IT DOES
#   1. Re-baselines the dataset's integrity hash (the upstream file legitimately
#      changes when new scores are added, so an intentional refresh updates the
#      trusted hash — silent cache tampering between syncs is still caught).
#   2. Re-downloads the dataset over verified TLS (new scores included).
#   3. For each newly-played WC2026 FINALS match involving your 48 teams: records
#      it and grades the prediction you logged (Elo for the next run comes from
#      this refreshed dataset).
#
# WHY THE FILTERS: the dataset also contains "FIFA World Cup qualification"
# matches with non-finalist teams (e.g. Czech Republic). Those must be excluded,
# or the team-name validator (correctly) rejects them. We keep only the finals
# tournament AND only matches whose teams are in your configured 48.
#
# USAGE (from project root, venv active):
#   python3 scripts/sync_results.py
#
# NOTE: the community dataset usually updates within ~a day of a match. If you
# need a result instantly, use the manual path (results_batch.csv + load_results.py).
# ==============================================================

import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_SOURCES, ALL_WC2026_TEAMS
from src.data.collector import SecureDataCollector
from src.tracking.results_store import add_result, list_results
from src.tracking.scorer import record_actual_from_score

TEAMS = set(ALL_WC2026_TEAMS)


def main():
    collector = SecureDataCollector()

    # 1) Re-baseline: drop the stored hash so the fresh download is re-trusted.
    fname = DATA_SOURCES["historical_results"]["local_path"].name
    reg_path = collector.registry_path
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        if reg.pop(fname, None) is not None:
            reg_path.write_text(json.dumps(reg, indent=2))
            print(f"Re-baselining integrity hash for {fname} (intentional refresh).")

    # 2) Force re-download (latest scores) over verified TLS.
    print("Pulling latest results from the source dataset...")
    df = collector.load_historical_results(force_download=True)
    df["date"] = pd.to_datetime(df["date"])

    # 3) WC2026 FINALS only (exclude qualification), this tournament, with a score.
    is_wc = (df["tournament"].str.contains("World Cup", case=False, na=False) &
             ~df["tournament"].str.contains("qualif", case=False, na=False))
    today = pd.Timestamp.today().normalize()
    played = df[is_wc & (df["date"] >= "2026-06-01") & (df["date"] <= today) &
                (df["home_score"].notna()) & (df["away_score"].notna())]

    have = list_results()
    have_keys = set()
    if not have.empty:
        have_keys = {(str(r.date), str(r.home_team), str(r.away_team)) for r in have.itertuples()}

    new = graded = skipped = 0
    for _, m in played.iterrows():
        date = m["date"].strftime("%Y-%m-%d")
        home, away = m["home_team"], m["away_team"]
        # Skip any team not in your configured 48 (e.g. a stray non-finalist row).
        if home not in TEAMS or away not in TEAMS:
            skipped += 1
            continue
        if (date, home, away) in have_keys:
            continue                                   # already synced — skip
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        try:
            add_result(home, away, hs, as_, date, m["tournament"]); new += 1
        except Exception as e:
            print(f"  SKIPPED | {home} vs {away} ({date}): {e}")
            continue
        try:
            record_actual_from_score(home, away, date, hs, as_); graded += 1
            print(f"  graded  | {home} {hs}-{as_} {away} ({date})")
        except ValueError:
            print(f"  elo-only| {home} {hs}-{as_} {away} ({date})  [no prediction was logged]")

    print(f"\nSynced {new} new result(s) | {graded} graded"
          + (f" | {skipped} non-finalist row(s) skipped" if skipped else "") + ".")
    if new:
        print("Re-run predictions to use the sharper ratings: "
              "python3 scripts/predict_and_log.py --upcoming 1")
    else:
        print("Nothing new in the source yet — check again after the next matches are logged upstream.")


if __name__ == "__main__":
    main()