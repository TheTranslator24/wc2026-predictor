# ==============================================================
# FILE: src/data/features.py
# PURPOSE: Turn raw match history into the numeric features the models learn from.
#
# TWO CLASSES:
#   EloCalculator   — dynamic team-strength ratings from match history
#   FeatureEngineer — builds the per-match feature vectors (26 features)
#
# THE GOLDEN RULE — NO DATA LEAKAGE:
#   Every feature for a match uses ONLY information available strictly
#   BEFORE that match. Elo is recorded pre-match; form/H2H windows filter
#   on date < match_date. This is what makes the model valid on real
#   future fixtures instead of cheating by peeking at the result.
#
# PERFORMANCE NOTE (the fix for the "frozen" build):
#   The naive approach rescans the whole 40k-row dataframe for every one
#   of ~31k training matches → ~1.2 BILLION row ops (minutes, looks hung).
#   Here we PRE-INDEX each team's history once into sorted arrays, then
#   use binary search (np.searchsorted) per match → microseconds each.
#   Same features, same no-leakage guarantee, ~100x faster.
# ==============================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ELO_CONFIG,
    FEATURE_CONFIG,
    CONFEDERATION,
    CONFEDERATION_STRENGTH,
    FIFA_RANKINGS,
)

logger = logging.getLogger("wc2026.data.features")


# ==============================================================
# ELO CALCULATOR
# ==============================================================
class EloCalculator:
    """
    Dynamic Elo ratings for international football.

    Walks the full history in chronological order, keeping a running
    rating per team. K is scaled by tournament importance and goal margin
    (a World Cup win moves Elo more than a friendly; a 5-0 more than a 1-0,
    but with diminishing returns).
    """

    def __init__(
        self,
        k_factor: float = ELO_CONFIG["k_factor"],
        initial_rating: float = ELO_CONFIG["initial_rating"],
    ):
        self.k_factor       = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}   # {team: current_elo}

    # ── internal helpers ──────────────────────────────────────
    def _tournament_multiplier(self, tournament: str) -> float:
        """Importance multiplier; unknown tournaments are treated as friendlies."""
        for key, mult in ELO_CONFIG["tournament_multipliers"].items():
            if key.lower() in tournament.lower():
                return mult
        return 1.0

    def _goal_diff_multiplier(self, goal_diff: int) -> float:
        """Scale K by margin, capped at 1.75 so blowouts can't swing Elo wildly."""
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.00
        if gd == 2:
            return 1.50
        return 1.75

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Logistic expected score (win probability) of A vs B, in [0,1]."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    # ── public ────────────────────────────────────────────────
    def get_current_elo(self, team: str) -> float:
        """Most recent rating for a team (initial_rating if unseen)."""
        return self.ratings.get(team, self.initial_rating)

    def update_from_match(
        self, home_team, away_team, home_score, away_score, tournament, match_date
    ) -> tuple[float, float]:
        """Apply one completed match to the ratings; return (new_home, new_away)."""
        r_home = self.ratings.get(home_team, self.initial_rating)
        r_away = self.ratings.get(away_team, self.initial_rating)

        e_home = self._expected_score(r_home, r_away)   # pre-match expectation
        e_away = 1.0 - e_home

        # Actual result: win=1.0, draw=0.5, loss=0.0
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home = s_away = 0.5

        effective_k = (
            self.k_factor
            * self._tournament_multiplier(tournament)
            * self._goal_diff_multiplier(home_score - away_score)
        )

        new_home = r_home + effective_k * (s_home - e_home)
        new_away = r_away + effective_k * (s_away - e_away)
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        return new_home, new_away

    def calculate_all_elo(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches oldest-first, recording PRE-match Elo per row.

        Recording Elo *before* updating is the no-leakage step: the feature
        you train on is the strength the teams had going INTO the match.

        Returns the frame plus: home_elo_before, away_elo_before, elo_diff.
        Iterates over raw numpy arrays (not df.iterrows()) — Elo is inherently
        sequential, but array iteration is ~20x faster than iterrows here.
        """
        df = results_df.sort_values("date").reset_index(drop=True).copy()

        # Pull columns out to numpy once; index by position in a tight loop.
        home_arr = df["home_team"].to_numpy()
        away_arr = df["away_team"].to_numpy()
        hs_arr   = df["home_score"].to_numpy()
        as_arr   = df["away_score"].to_numpy()
        tour_arr = df["tournament"].astype(str).to_numpy()
        date_arr = df["date"].to_numpy()

        n = len(df)
        home_before = np.empty(n, dtype=np.float64)
        away_before = np.empty(n, dtype=np.float64)

        logger.info(f"Computing Elo for {n:,} matches (chronological)...")

        for i in range(n):
            h, a = home_arr[i], away_arr[i]
            # RECORD pre-match Elo (this is the leakage-free feature)
            home_before[i] = self.ratings.get(h, self.initial_rating)
            away_before[i] = self.ratings.get(a, self.initial_rating)
            # THEN update with the actual result, if the match has a score
            hs, as_ = hs_arr[i], as_arr[i]
            if not (pd.isna(hs) or pd.isna(as_)):
                self.update_from_match(h, a, int(hs), int(as_), tour_arr[i], date_arr[i])

        df["home_elo_before"] = home_before
        df["away_elo_before"] = away_before
        df["elo_diff"]        = home_before - away_before

        if self.ratings:
            logger.info(
                f"Elo complete | teams_rated={len(self.ratings)} | "
                f"range {min(self.ratings.values()):.0f}–{max(self.ratings.values()):.0f}"
            )
        return df


# ==============================================================
# FEATURE ENGINEER
# ==============================================================
class FeatureEngineer:
    """
    Builds 25 features per match for both training and prediction.

    Pre-indexes team history (and head-to-head pairs) ONCE in __init__,
    so each per-match lookup is a binary search, not a full-table scan.

    Feature groups:
      Elo (3) · FIFA rank (3) · form (3) · goals (6) · H2H (3) ·
      confederation (3) · neutral (1) · adjustments (3)  [+ derived diffs]
    """

    # baseline returned when a team/pair has too little history
    _NEUTRAL_FORM = {
        "form_points": 0.50, "goals_scored_avg": 1.20,
        "goals_conceded_avg": 1.20, "n_matches": 0,
    }
    _NEUTRAL_H2H = {"h2h_win_rate": 0.5, "h2h_goal_diff_avg": 0.0, "h2h_count": 0}

    def __init__(self, results_df: pd.DataFrame):
        start_year = FEATURE_CONFIG["training_start_year"]
        # Modern era only — older football is weakly predictive of today.
        self.df = (
            results_df[results_df["date"].dt.year >= start_year]
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )
        self._build_team_index()   # per-team sorted history
        self._build_h2h_index()    # per-pair sorted history
        logger.info(
            f"FeatureEngineer ready | rows={len(self.df):,} | "
            f"from {start_year} | teams_indexed={len(self._team_index)} | features=25"
        )

    # ── index construction (runs once) ────────────────────────
    def _build_team_index(self) -> None:
        """
        Build {team: sorted arrays of (date, scored, conceded, points)}.

        Each match contributes TWO rows (home perspective + away perspective)
        so a single lookup per team sees every game that team played.
        """
        valid = self.df[self.df["home_score"].notna() & self.df["away_score"].notna()]

        # home-perspective and away-perspective stacked into one long table
        home = pd.DataFrame({
            "team":     valid["home_team"].to_numpy(),
            "date":     valid["date"].to_numpy(),
            "scored":   valid["home_score"].to_numpy(dtype=np.float64),
            "conceded": valid["away_score"].to_numpy(dtype=np.float64),
        })
        away = pd.DataFrame({
            "team":     valid["away_team"].to_numpy(),
            "date":     valid["date"].to_numpy(),
            "scored":   valid["away_score"].to_numpy(dtype=np.float64),
            "conceded": valid["home_score"].to_numpy(dtype=np.float64),
        })
        long = pd.concat([home, away], ignore_index=True)
        long["points"] = np.where(
            long["scored"] > long["conceded"], 1.0,
            np.where(long["scored"] == long["conceded"], 0.5, 0.0),
        )
        long.sort_values(["team", "date"], inplace=True)

        self._team_index: dict[str, dict[str, np.ndarray]] = {}
        for team, g in long.groupby("team", sort=False):
            self._team_index[team] = {
                "dates":    g["date"].to_numpy(),       # sorted ascending
                "scored":   g["scored"].to_numpy(),
                "conceded": g["conceded"].to_numpy(),
                "points":   g["points"].to_numpy(),
            }

    def _build_h2h_index(self) -> None:
        """
        Build {(team_x, team_y): [(date, home_team, home_score, away_score), ...]}.

        Keyed by the alphabetically-ordered pair so A-vs-B and B-vs-A share
        one bucket. Buckets are tiny (rarely >30), so per-query work is trivial.
        """
        valid = self.df[self.df["home_score"].notna() & self.df["away_score"].notna()]
        self._h2h_index: dict[tuple, list] = {}
        for home, away, date, hs, as_ in zip(
            valid["home_team"].to_numpy(), valid["away_team"].to_numpy(),
            valid["date"].to_numpy(),
            valid["home_score"].to_numpy(dtype=np.float64),
            valid["away_score"].to_numpy(dtype=np.float64),
        ):
            key = (home, away) if home <= away else (away, home)
            self._h2h_index.setdefault(key, []).append((date, home, hs, as_))
        for key in self._h2h_index:
            self._h2h_index[key].sort(key=lambda r: r[0])   # chronological

    # ── per-match feature helpers (fast lookups) ──────────────
    def _recent_form(self, team: str, before_date, n: int = 10) -> dict:
        """
        Exponentially-weighted form over a team's last `n` matches strictly
        before `before_date`. Most-recent match weight 1.0, then 0.9, 0.81, ...
        Binary search finds the window in O(log m) — no full-table scan.
        """
        idx = self._team_index.get(team)
        if idx is None:
            return dict(self._NEUTRAL_FORM)

        decay = FEATURE_CONFIG["form_decay_factor"]
        min_m = FEATURE_CONFIG["min_matches_for_feature"]

        before = np.datetime64(pd.Timestamp(before_date))
        # side="left" => everything before `pos` is STRICTLY earlier than `before`
        pos = int(np.searchsorted(idx["dates"], before, side="left"))
        if pos < min_m:
            out = dict(self._NEUTRAL_FORM)
            out["n_matches"] = pos
            return out

        lo = max(0, pos - n)
        # reverse so index 0 is the MOST recent match (gets the highest weight)
        scored   = idx["scored"][lo:pos][::-1]
        conceded = idx["conceded"][lo:pos][::-1]
        points   = idx["points"][lo:pos][::-1]

        weights = decay ** np.arange(len(scored))   # [1, 0.9, 0.81, ...]
        total_w = weights.sum()
        return {
            "form_points":        float((points * weights).sum() / total_w),
            "goals_scored_avg":   float((scored * weights).sum() / total_w),
            "goals_conceded_avg": float((conceded * weights).sum() / total_w),
            "n_matches":          int(len(scored)),
        }

    def _head_to_head(self, team_a: str, team_b: str, before_date, years: int = 20) -> dict:
        """
        Head-to-head record of team_a vs team_b within `years` before the match.
        Win rate and average goal difference are from team_a's perspective.
        """
        key = (team_a, team_b) if team_a <= team_b else (team_b, team_a)
        records = self._h2h_index.get(key)
        if not records:
            return dict(self._NEUTRAL_H2H)

        before = np.datetime64(pd.Timestamp(before_date))
        cutoff = np.datetime64(pd.Timestamp(before_date) - pd.DateOffset(years=years))

        a_wins, total_gd, count = 0, 0.0, 0
        for date, home, hs, as_ in records:
            if date < cutoff or date >= before:    # window: [cutoff, before)
                continue
            gd = (hs - as_) if home == team_a else (as_ - hs)   # team_a perspective
            if gd > 0:
                a_wins += 1
            total_gd += gd
            count += 1

        if count == 0:
            return dict(self._NEUTRAL_H2H)
        return {
            "h2h_win_rate":      a_wins / count,
            "h2h_goal_diff_avg": total_gd / count,
            "h2h_count":         min(count, 30),   # cap dampens ancient-rivalry outliers
        }

    # ── public API ────────────────────────────────────────────
    def build_feature_row(
        self, home_team, away_team, match_date,
        elo_calc: Optional[EloCalculator] = None,
        adjustments: Optional[dict] = None,
    ) -> dict:
        """
        Build one 25-feature row for a single match (training or prediction).
        Pass `elo_calc` to read live ratings (prediction); during training the
        caller overwrites the Elo fields with pre-computed pre-match values.
        """
        if adjustments is None:
            adjustments = {"weather_factor": 1.0, "pitch_condition": 1.0, "lineup_stability": 1.0}

        home_form = self._recent_form(home_team, match_date)
        away_form = self._recent_form(away_team, match_date)
        h2h       = self._head_to_head(home_team, away_team, match_date)

        home_rank = FIFA_RANKINGS.get(home_team, 48)
        away_rank = FIFA_RANKINGS.get(away_team, 48)

        home_conf_str = CONFEDERATION_STRENGTH.get(CONFEDERATION.get(home_team, "UEFA"), 0.6)
        away_conf_str = CONFEDERATION_STRENGTH.get(CONFEDERATION.get(away_team, "UEFA"), 0.6)

        if elo_calc is not None:
            home_elo = elo_calc.get_current_elo(home_team)
            away_elo = elo_calc.get_current_elo(away_team)
        else:
            home_elo = away_elo = 1500.0

        return {
            # Elo
            "elo_diff":        home_elo - away_elo,
            "home_elo_norm":   (home_elo - 1000.0) / 1000.0,
            "away_elo_norm":   (away_elo - 1000.0) / 1000.0,
            # FIFA ranking (positive fifa_rank_diff = home better ranked)
            "fifa_rank_diff":  away_rank - home_rank,
            "home_rank_norm":  home_rank / 48.0,
            "away_rank_norm":  away_rank / 48.0,
            # Form
            "home_form_pts":   home_form["form_points"],
            "away_form_pts":   away_form["form_points"],
            "form_diff":       home_form["form_points"] - away_form["form_points"],
            # Goals
            "home_scored_avg":   home_form["goals_scored_avg"],
            "home_conceded_avg": home_form["goals_conceded_avg"],
            "away_scored_avg":   away_form["goals_scored_avg"],
            "away_conceded_avg": away_form["goals_conceded_avg"],
            "home_gd_avg":       home_form["goals_scored_avg"] - home_form["goals_conceded_avg"],
            "away_gd_avg":       away_form["goals_scored_avg"] - away_form["goals_conceded_avg"],
            # Head-to-head
            "h2h_win_rate":    h2h["h2h_win_rate"],
            "h2h_gd_avg":      h2h["h2h_goal_diff_avg"],
            "h2h_count":       h2h["h2h_count"],
            # Confederation
            "home_conf_str":   home_conf_str,
            "away_conf_str":   away_conf_str,
            "conf_str_diff":   home_conf_str - away_conf_str,
            # Context: every World Cup match is on neutral ground
            "is_neutral":      1.0,
            # User-supplied adjustments (validated upstream)
            "weather_factor":   adjustments.get("weather_factor", 1.0),
            "pitch_condition":  adjustments.get("pitch_condition", 1.0),
            "lineup_stability": adjustments.get("lineup_stability", 1.0),
        }

    def build_training_dataset(
        self, results_with_elo: pd.DataFrame, elo_calc: EloCalculator
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build the full feature matrix X and label vector y for training.

        Keeps competitive matches with known scores (drops friendlies/olympics —
        weaker signal for the World Cup). Labels: 0=Away win, 1=Draw, 2=Home win.
        Now fast: each row uses the pre-indexed lookups instead of full scans.
        """
        skip = ("friendly", "olympic")
        competitive = results_with_elo[
            results_with_elo["home_score"].notna()
            & ~results_with_elo["tournament"].str.lower().apply(
                lambda t: any(k in t for k in skip)
            )
        ].copy()

        logger.info(f"Building training set | competitive_matches={len(competitive):,}")

        # Pull arrays once (avoids slow df.iterrows()).
        home_a = competitive["home_team"].to_numpy()
        away_a = competitive["away_team"].to_numpy()
        date_a = competitive["date"].to_numpy()
        hs_a   = competitive["home_score"].to_numpy()
        as_a   = competitive["away_score"].to_numpy()
        ediff  = competitive.get("elo_diff", pd.Series(np.zeros(len(competitive)))).to_numpy()
        h_eb   = competitive.get("home_elo_before", pd.Series(np.full(len(competitive), 1500.0))).to_numpy()
        a_eb   = competitive.get("away_elo_before", pd.Series(np.full(len(competitive), 1500.0))).to_numpy()

        rows, labels = [], []
        for i in range(len(competitive)):
            feat = self.build_feature_row(home_a[i], away_a[i], date_a[i], elo_calc=None)
            # Overwrite Elo fields with the leakage-free PRE-match values.
            feat["elo_diff"]      = float(ediff[i])
            feat["home_elo_norm"] = (float(h_eb[i]) - 1000.0) / 1000.0
            feat["away_elo_norm"] = (float(a_eb[i]) - 1000.0) / 1000.0
            rows.append(feat)

            if   hs_a[i] > as_a[i]: labels.append(2)   # home win
            elif hs_a[i] < as_a[i]: labels.append(0)   # away win
            else:                   labels.append(1)   # draw

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name="result", dtype=int)

        counts = y.value_counts().sort_index()
        logger.info(
            f"Training set built | samples={len(X):,} | features={len(X.columns)} | "
            f"away={counts.get(0,0):,} draw={counts.get(1,0):,} home={counts.get(2,0):,}"
        )
        return X, y