#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/predict_and_log.py
# PURPOSE: Predict upcoming WC2026 fixtures and LOG each forecast BEFORE the
#          match is played — the half that makes calibration honest.
#
# Depends ONLY on your core stack (match_predictor + scorer). No scoreline model,
# no chart libraries — so it can't fail on a missing optional module. Generate
# charts separately with main.py or scripts/make_flag_charts.py.
#
# It reads the real WC2026 schedule (dates) from your downloaded dataset, so you
# don't hand-type fixtures, predicts each upcoming match, and logs it.
#
# USAGE (from project root, venv active):
#   python3 scripts/predict_and_log.py --upcoming 1     # tomorrow's fixtures
#   python3 scripts/predict_and_log.py --date 2026-06-21
# ==============================================================

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor.match_predictor import WC2026Predictor
from src.tracking.scorer import log_prediction


def upcoming_fixtures(predictor, target_date=None, upcoming_days=None):
    """Unplayed WC2026 fixtures (this tournament only) from the loaded dataset."""
    df = predictor.results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df["tournament"].str.contains("World Cup", case=False, na=False)) &
            (df["date"] >= "2026-06-01") & (df["home_score"].isna())]
    if target_date:
        d = pd.Timestamp(target_date)
        wc = wc[wc["date"].dt.date == d.date()]
    elif upcoming_days:
        today = pd.Timestamp.today().normalize()
        wc = wc[(wc["date"] >= today) & (wc["date"] < today + pd.Timedelta(days=upcoming_days))]
    return wc.sort_values("date")


def main():
    ap = argparse.ArgumentParser(description="Predict + log upcoming WC2026 fixtures")
    ap.add_argument("--date", help="single match-day YYYY-MM-DD")
    ap.add_argument("--upcoming", type=int, help="next N days of fixtures")
    args = ap.parse_args()
    if not args.date and not args.upcoming:
        args.upcoming = 1                                # default: tomorrow

    predictor = WC2026Predictor().setup()
    fixtures = upcoming_fixtures(predictor, args.date, args.upcoming)
    if fixtures.empty:
        print("No upcoming unplayed fixtures match that window.")
        return

    print(f"\nLogging {len(fixtures)} forecast(s):\n")
    logged = 0
    for _, m in fixtures.iterrows():
        home, away = m["home_team"], m["away_team"]
        date = m["date"].strftime("%Y-%m-%d")
        try:
            pred = predictor.predict(home, away, match_date=date)
            log_prediction(pred)
            logged += 1
            print(f"  {date} | {home} vs {away:18s} -> "
                  f"{pred['predicted_outcome']:9s} {pred['confidence']:.0%}")
        except Exception as e:
            print(f"  {date} | {home} vs {away}: SKIPPED ({e})")

    print(f"\n{logged} forecast(s) logged. After these matches are played, run:")
    print("  python3 scripts/sync_results.py     (pull scores from the source + grade)")


if __name__ == "__main__":
    main()