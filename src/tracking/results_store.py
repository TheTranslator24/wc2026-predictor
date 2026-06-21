# ==============================================================
# FILE: src/tracking/results_store.py
# PURPOSE: Persist real WC2026 results so they COMPOUND into your ratings.
#
# THE PROBLEM IT SOLVES:
#   Every run rebuilds Elo from the downloaded CSV, and update_elo() only
#   affects the current session. So a result you watched last night is
#   forgotten next run — unless it's already in the upstream data (which lags).
#   This module keeps a local results file you append to; it's replayed into
#   Elo at the start of every run, so your ratings stay current immediately.
#
# This is the reliable "manual half" of the hybrid live-results approach.
#
# FILE: data/tournament_results.csv
#   columns: date, home_team, away_team, home_score, away_score, stage
# ==============================================================

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.security.data_validator import validate_team_name

logger = logging.getLogger("wc2026.tracking.results")

RESULTS_PATH = DATA_DIR / "tournament_results.csv"
_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "stage"]


def _load() -> pd.DataFrame:
    """Load the results file, or an empty frame with the right columns."""
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH, dtype={"home_team": str, "away_team": str, "stage": str})
    return pd.DataFrame(columns=_COLUMNS)


def add_result(home_team: str, away_team: str, home_score: int, away_score: int,
               date: str | None = None, stage: str = "FIFA World Cup") -> None:
    """
    Record one completed WC2026 result (validated, de-duplicated, persisted).

    Team names are whitelisted (same security boundary as predictions), and a
    repeat of the same date+teams overwrites rather than duplicating.

    Example:
        add_result("Mexico", "South Africa", 2, 1, "2026-06-11", "Group A")
    """
    home_team = validate_team_name(home_team, "results_store")
    away_team = validate_team_name(away_team, "results_store")
    date = date or pd.Timestamp.today().strftime("%Y-%m-%d")

    df = _load()
    # Drop any existing row for the same fixture+date (idempotent updates).
    mask = ~((df["date"] == date) & (df["home_team"] == home_team) & (df["away_team"] == away_team))
    df = df[mask]

    row = {"date": date, "home_team": home_team, "away_team": away_team,
           "home_score": int(home_score), "away_score": int(away_score), "stage": stage}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.sort_values("date", inplace=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Result recorded | {home_team} {home_score}-{away_score} {away_team} ({stage})")


def apply_to_elo(elo_calc, tournament: str = "FIFA World Cup") -> int:
    """
    Replay all stored results into a freshly-computed EloCalculator.

    Call this AFTER calculate_all_elo() and BEFORE predicting, so the ratings
    reflect every result you've recorded — even ones not yet in the upstream
    dataset. World Cup matches carry the full importance multiplier.

    Returns: number of results applied.
    """
    df = _load()
    if df.empty:
        logger.info("No stored tournament results to apply (none played/recorded yet).")
        return 0
    applied = 0
    for _, r in df.sort_values("date").iterrows():
        elo_calc.update_from_match(
            r["home_team"], r["away_team"],
            int(r["home_score"]), int(r["away_score"]),
            tournament, pd.Timestamp(r["date"]),
        )
        applied += 1
    logger.info(f"Applied {applied} stored result(s) to Elo ratings.")
    return applied


def list_results() -> pd.DataFrame:
    """Return all stored results (for display/inspection)."""
    return _load()
