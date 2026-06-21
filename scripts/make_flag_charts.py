#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/make_flag_charts.py
# PURPOSE: Save prediction charts with each team's REAL colors and downloaded
#          flag icons (no emoji "tofu" boxes). Uses plot_match_colored.
#
# It downloads the 48 public-domain flags once (cached), then renders a colored
# probability chart per fixture into outputs/predictions/flags/.
#
# USAGE (from project root, venv active):
#   python3 scripts/make_flag_charts.py --date 2026-06-21   # one day's fixtures
#   python3 scripts/make_flag_charts.py --all-groups        # every group pairing (72)
#   python3 scripts/make_flag_charts.py --upcoming 3        # next 3 days
# ==============================================================

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUTS_DIR, WC2026_GROUPS
from src.predictor.match_predictor import WC2026Predictor
from src.visualization.flag_loader import download_all_flags
from src.visualization.advanced_charts import plot_match_colored


def main():
    ap = argparse.ArgumentParser(description="Flag + color prediction charts")
    ap.add_argument("--date", help="single match-day YYYY-MM-DD")
    ap.add_argument("--all-groups", action="store_true", help="every group round-robin pairing")
    ap.add_argument("--upcoming", type=int, help="next N days of fixtures")
    args = ap.parse_args()

    # 1) flags once (idempotent — skips any already on disk)
    print("Ensuring flag images are available...")
    n = download_all_flags()
    print(f"Flags ready ({n} fetched this run).")

    predictor = WC2026Predictor().setup()
    out_dir = OUTPUTS_DIR / "predictions" / "flags"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) choose fixtures
    fixtures = []
    if args.all_groups:
        for teams in WC2026_GROUPS.values():
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    fixtures.append((teams[i], teams[j], None))
    else:
        df = predictor.results_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        wc = df[(df["tournament"].str.contains("World Cup", case=False, na=False)) &
                (df["date"] >= "2026-06-01") & (df["home_score"].isna())]
        if args.date:
            wc = wc[wc["date"].dt.date == pd.Timestamp(args.date).date()]
        else:
            today = pd.Timestamp.today().normalize()
            days = args.upcoming or 3
            wc = wc[(wc["date"] >= today) & (wc["date"] < today + pd.Timedelta(days=days))]
        fixtures = [(r.home_team, r.away_team, r.date.strftime("%Y-%m-%d"))
                    for r in wc.sort_values("date").itertuples()]

    if not fixtures:
        print("No fixtures matched. Try --all-groups or a specific --date.")
        return

    # 3) render
    print(f"Generating {len(fixtures)} flag chart(s) into {out_dir}...")
    for home, away, date in fixtures:
        try:
            pred = predictor.predict(home, away, match_date=date)
            fname = f"{home}_vs_{away}.png".replace(" ", "_")
            plot_match_colored(pred, save_path=out_dir / fname, show=False)
            print(f"  saved {fname}")
        except Exception as e:
            print(f"  skip {home} vs {away}: {e}")

    print(f"\nDone. Open them in {out_dir}")


if __name__ == "__main__":
    main()
