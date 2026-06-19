#!/usr/bin/env python3
# ==============================================================
# FILE: main.py
# PURPOSE: Entry point. Edit MATCHES_TO_PREDICT below, then run.
#
# USAGE:
#   python3 main.py                # predict the matches listed below
#   python3 main.py --retrain      # force both models to re-train
#   python3 main.py --group A      # predict one group (A-L)
#   python3 main.py --all-groups   # predict all 12 groups
#   python3 main.py --tournament   # full-tournament championship sim
#   python3 main.py --no-charts    # skip saving PNG charts (faster)
#
# FIRST RUN trains + saves both models; later runs load them in seconds.
# ==============================================================

import sys
import argparse
import logging
from pathlib import Path

# Make project root importable BEFORE any src.* import, so the script
# works no matter which directory you launch it from.
sys.path.insert(0, str(Path(__file__).parent))

from src.predictor.match_predictor import WC2026Predictor
from src.visualization.probability_charts import (
    plot_match_prediction,
    plot_championship_probabilities,
    plot_group_standings,
)

logger = logging.getLogger("wc2026.main")


# ==============================================================
# ▼▼▼  EDIT THIS SECTION TO ADD YOUR MATCHES  ▼▼▼
# Format: ("Team A (home)", "Team B (away)", adjustments_or_None, "YYYY-MM-DD")
# Adjustment factors (all optional, 1.0 = ideal): weather_factor,
# pitch_condition, lineup_stability. Use EXACT names from config.py
# ("United States" not "USA", "DR Congo" not "Congo").
# ==============================================================
MATCHES_TO_PREDICT = [
    ("Mexico",        "South Africa", None, "2026-06-11"),
    ("Brazil",        "Morocco",      None, "2026-06-13"),
    ("Spain", "Cape Verde", {"weather_factor": 1.0, "pitch_condition": 0.98,
                             "lineup_stability": 0.95}, "2026-06-15"),
    ("France",        "Senegal",      None, "2026-06-16"),
    ("Argentina",     "Algeria",      None, "2026-06-16"),
    ("United States", "Paraguay", {"weather_factor": 0.9, "pitch_condition": 1.0,
                                   "lineup_stability": 0.95}, "2026-06-12"),
    ("Portugal",      "Colombia",     None, "2026-06-17"),
    ("England",       "Croatia",      None, "2026-06-18"),
]

RUN_ALL_GROUPS = False   # or pass --all-groups
RUN_TOURNAMENT = False   # or pass --tournament
# ==============================================================
# ▲▲▲  END OF EDITABLE SECTION  ▲▲▲
# ==============================================================


def parse_args():
    p = argparse.ArgumentParser(description="WC2026 Match Predictor")
    p.add_argument("--retrain",    action="store_true", help="force model re-training")
    p.add_argument("--group",      type=str, default=None, help="predict a single group (A-L)")
    p.add_argument("--all-groups", action="store_true", help="predict all 12 groups")
    p.add_argument("--tournament", action="store_true", help="full tournament simulation")
    p.add_argument("--no-charts",  action="store_true", help="skip saving charts")
    return p.parse_args()


def print_prediction(pred: dict) -> None:
    """Compact terminal summary of one prediction."""
    m, mc = pred["match"], pred["monte_carlo"]
    print(f"\n  {'─'*50}")
    print(f"  {m['team_a']}  vs  {m['team_b']}  [{m['date']}]")
    print(f"  {'─'*50}")
    print(f"  Predicted: {pred['predicted_outcome']}  (confidence: {pred['confidence']:.1%})")
    print(f"  Home Win: {pred['home_win_prob']:.1%}  |  Draw: {pred['draw_prob']:.1%}  "
          f"|  Away Win: {pred['away_win_prob']:.1%}")
    print(f"  Monte Carlo ({mc['n_simulations']:,} sims):")
    print(f"    Home {mc['home_win_pct']}% [{mc['home_win_95ci'][0]}-{mc['home_win_95ci'][1]}%]  "
          f"Draw {mc['draw_pct']}% [{mc['draw_95ci'][0]}-{mc['draw_95ci'][1]}%]  "
          f"Away {mc['away_win_pct']}% [{mc['away_win_95ci'][0]}-{mc['away_win_95ci'][1]}%]")
    print(f"  Elo: {m['team_a']} {m['elo_home']:.0f}  |  {m['team_b']} {m['elo_away']:.0f}  "
          f"|  Δ={m['elo_diff']:+.0f}")


def main():
    args = parse_args()
    print("=" * 55)
    print("  WC2026 MATCH PREDICTOR")
    print("  XGBoost + LSTM Ensemble + Monte Carlo")
    print("  M1 Mac  |  Secure Build")
    print("=" * 55)

    predictor = WC2026Predictor(auto_load=True)
    predictor.setup(force_retrain=args.retrain)

    # ── Individual matches ─────────────────────────────────────
    if MATCHES_TO_PREDICT:
        print(f"\n\n{'═'*55}\n  MATCH PREDICTIONS ({len(MATCHES_TO_PREDICT)})\n{'═'*55}")
        for team_a, team_b, adj, date in MATCHES_TO_PREDICT:
            try:
                pred = predictor.predict(team_a, team_b, adj, date)
                print_prediction(pred)
                if not args.no_charts:
                    plot_match_prediction(pred, show=False)
            except Exception as e:
                logger.error(f"Failed to predict {team_a} vs {team_b}: {e}")
                print(f"  Skipped {team_a} vs {team_b}: {e}")

    # ── Group predictions ──────────────────────────────────────
    run_groups = args.all_groups or RUN_ALL_GROUPS
    if run_groups or args.group:
        groups = (list("ABCDEFGHIJKL") if run_groups else [args.group.upper()])
        print(f"\n\n{'═'*55}\n  GROUP PREDICTIONS\n{'═'*55}")
        for g in groups:
            try:
                gp = predictor.predict_group(g)
                print(f"\n  Group {g}:")
                for s in gp["projected_standings"]:
                    adv = "ADV" if s["team"] in gp["advance"] else "   "
                    print(f"    {s['rank']}. {s['team']:25s} {s['points']} pts  "
                          f"GD:{s['goal_diff']:+d}  {adv}")
                if not args.no_charts:
                    plot_group_standings(gp, show=False)
            except Exception as e:
                logger.error(f"Group {g} prediction failed: {e}")

    # ── Tournament simulation ──────────────────────────────────
    if args.tournament or RUN_TOURNAMENT:
        print(f"\n\n{'═'*55}\n  CHAMPIONSHIP PROBABILITIES\n{'═'*55}")
        champ = predictor.predict_tournament()
        for rank, (team, pct) in enumerate(list(champ.items())[:20], 1):
            bar = "█" * int(pct / 0.5)
            print(f"  {rank:2d}. {team:25s} {pct:5.1f}%  {bar}")
        if not args.no_charts:
            plot_championship_probabilities(champ, top_n=20, show=False)

    print(f"\n\n  Charts saved to: outputs/predictions/")
    print(f"  Logs at:         logs/predictor.log")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
