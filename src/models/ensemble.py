# ==============================================================
# FILE: src/models/ensemble.py
# PURPOSE: Blend XGBoost + LSTM, add Monte Carlo simulation + condition adjustments.
#
# WHY BLEND: XGBoost reads the static snapshot (Elo, rankings, H2H); the LSTM
#   reads momentum. Averaging their probabilities cancels some of each model's
#   independent error, so the blend beats either alone. Default weights:
#   XGBoost 0.65 / LSTM 0.35 (XGBoost is the stronger base on tabular data).
#
# WHY MONTE CARLO: a single number ("Home 62%") hides uncertainty. Sampling
#   10,000 outcomes from the blended distribution turns it into
#   "Home 62% [95% CI 60.2-63.8%]" — an honest statement of confidence.
#
# CONDITION ADJUSTMENTS: weather/pitch/lineup factors nudge the distribution
#   toward [1/3,1/3,1/3] (max uncertainty), because bad conditions add
#   randomness and quietly favour the underdog.
# ==============================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MODEL_CONFIG

logger = logging.getLogger("wc2026.models.ensemble")

LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
UNIFORM   = np.array([1/3, 1/3, 1/3])   # the "maximum uncertainty" distribution


class EnsemblePredictor:
    """Combines the two models and runs Monte Carlo + tournament simulation."""

    def __init__(self, xgb_model, lstm_trainer=None):
        """
        Args:
            xgb_model:    a trained XGBoostPredictor (required)
            lstm_trainer: a trained LSTMTrainer (optional; falls back to XGBoost-only)
        """
        self.xgb    = xgb_model
        self.lstm   = lstm_trainer
        self.w_xgb  = MODEL_CONFIG["ensemble"]["xgboost_weight"]     # 0.65
        self.w_lstm = MODEL_CONFIG["ensemble"]["lstm_weight"]        # 0.35
        self.n_sims = MODEL_CONFIG["monte_carlo"]["n_simulations"]   # 10000
        np.random.seed(MODEL_CONFIG["monte_carlo"]["random_state"])  # reproducible sims

        status = "loaded" if (lstm_trainer and lstm_trainer.is_trained) else "not available"
        logger.info(f"Ensemble ready | xgb={self.w_xgb:.0%} | lstm={self.w_lstm:.0%} "
                    f"[{status}] | mc_sims={self.n_sims:,}")

    # ── blend the two models ──────────────────────────────────
    def _ensemble_probs(self, features: dict,
                        lstm_sequence: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (xgboost_probs, blended_probs), each shape (3,)."""
        xgb_p = self.xgb.predict_proba(pd.DataFrame([features]))[0]   # always available
        if self.lstm is not None and self.lstm.is_trained and lstm_sequence is not None:
            lstm_p   = self.lstm.predict_proba(lstm_sequence)
            ensemble = self.w_xgb * xgb_p + self.w_lstm * lstm_p
        else:
            ensemble = xgb_p                                          # XGBoost-only fallback
        ensemble = ensemble / ensemble.sum()                         # renormalize to sum 1
        return xgb_p, ensemble

    # ── condition adjustments ─────────────────────────────────
    def _apply_adjustments(self, probs: np.ndarray, adjustments: dict) -> np.ndarray:
        """
        Pull the distribution toward uniform in proportion to how BAD each
        condition is. Formula per factor: p' = (1-α)·p + α·uniform,
        where α = (1 - factor) · max_pull. factor=1.0 (ideal) => no change.
        """
        def _blend(p, factor, max_alpha):
            alpha = (1.0 - factor) * max_alpha
            blended = (1.0 - alpha) * p + alpha * UNIFORM
            return blended / blended.sum()

        adj = probs.copy()
        weather = adjustments.get("weather_factor", 1.0)
        pitch   = adjustments.get("pitch_condition", 1.0)
        lineup  = adjustments.get("lineup_stability", 1.0)
        if weather < 1.0: adj = _blend(adj, weather, 0.30)   # weather pulls up to 30%
        if pitch   < 1.0: adj = _blend(adj, pitch,   0.20)   # pitch   up to 20%
        if lineup  < 1.0: adj = _blend(adj, lineup,  0.25)   # lineup  up to 25%
        return adj / adj.sum()

    # ── Monte Carlo ───────────────────────────────────────────
    def _monte_carlo(self, probs: np.ndarray) -> dict:
        """
        Sample n_sims outcomes from `probs` and summarize with Wilson 95% CIs.
        Wilson intervals stay valid near 0/1 where the normal approximation fails.
        """
        outcomes = np.random.choice([0, 1, 2], size=self.n_sims, p=probs)
        away_n = int(np.sum(outcomes == 0))
        draw_n = int(np.sum(outcomes == 1))
        home_n = int(np.sum(outcomes == 2))
        n = self.n_sims

        def wilson_ci(k, n, z=1.96):
            p = k / n
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
            return (round(max(0.0, center - margin) * 100, 1),
                    round(min(1.0, center + margin) * 100, 1))

        return {
            "n_simulations": n,
            "home_win_count": home_n, "draw_count": draw_n, "away_win_count": away_n,
            "home_win_pct": round(home_n / n * 100, 1),
            "draw_pct":     round(draw_n / n * 100, 1),
            "away_win_pct": round(away_n / n * 100, 1),
            "home_win_95ci": wilson_ci(home_n, n),
            "draw_95ci":     wilson_ci(draw_n, n),
            "away_win_95ci": wilson_ci(away_n, n),
        }

    # ── full single-match prediction ──────────────────────────
    def predict_match(self, features: dict,
                      lstm_sequence: Optional[np.ndarray] = None) -> dict:
        """Blend -> adjust -> Monte Carlo -> packaged result dict."""
        adjustments = {k: features.get(k, 1.0)
                       for k in ("weather_factor", "pitch_condition", "lineup_stability")}
        xgb_p, ensemble_p = self._ensemble_probs(features, lstm_sequence)
        adjusted_p        = self._apply_adjustments(ensemble_p, adjustments)
        mc                = self._monte_carlo(adjusted_p)
        best = int(np.argmax(adjusted_p))
        return {
            "home_win_prob": float(adjusted_p[2]),
            "draw_prob":     float(adjusted_p[1]),
            "away_win_prob": float(adjusted_p[0]),
            "predicted_outcome": LABEL_MAP[best],
            "confidence":        float(adjusted_p[best]),
            "xgboost_prediction": {
                "home_win": float(xgb_p[2]), "draw": float(xgb_p[1]), "away_win": float(xgb_p[0]),
            },
            "ensemble_weights": {
                "xgboost": self.w_xgb,
                "lstm": self.w_lstm if (self.lstm and self.lstm.is_trained) else 0.0,
            },
            "monte_carlo": mc,
            "adjustments_applied": adjustments,
        }

    # ── full-tournament Monte Carlo ───────────────────────────
    def simulate_tournament(self, groups: dict, elo_calc, n_sims: int = 1000) -> dict[str, float]:
        """
        Simulate the whole tournament n_sims times using fast Elo-only odds
        (the full model is too slow to run thousands of times x 104 matches).
        Returns {team: championship_%} sorted high to low.
        """
        counts = {team: 0 for teams in groups.values() for team in teams}
        logger.info(f"Full tournament simulation | {n_sims:,} runs...")
        for _ in range(n_sims):
            champ = self._sim_one_tournament(groups, elo_calc)
            if champ:
                counts[champ] += 1
        result = {team: round(c / n_sims * 100, 1) for team, c in counts.items()}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def _sim_one_tournament(self, groups: dict, elo_calc) -> Optional[str]:
        """One full run: group stage (top 2 advance) then single-elim knockout."""
        ko_pool: list[str] = []
        for group_teams in groups.values():
            points = {t: 0 for t in group_teams}
            gd     = {t: 0 for t in group_teams}
            for i, ta in enumerate(group_teams):
                for tb in group_teams[i+1:]:
                    w = self._elo_match_probs(ta, tb, elo_calc)
                    r = np.random.random()
                    if r < w[2]:                 points[ta] += 3; gd[ta] += 1; gd[tb] -= 1
                    elif r < w[2] + w[1]:        points[ta] += 1; points[tb] += 1
                    else:                        points[tb] += 3; gd[tb] += 1; gd[ta] -= 1
            ranked = sorted(group_teams, key=lambda t: (points[t], gd[t]), reverse=True)
            ko_pool.extend(ranked[:2])           # simplified: ignores best-3rd-place rule

        while len(ko_pool) > 1:                  # knockout rounds until one remains
            nxt = []
            for i in range(0, len(ko_pool), 2):
                if i + 1 >= len(ko_pool):
                    nxt.append(ko_pool[i]); continue
                w = self._elo_match_probs(ko_pool[i], ko_pool[i+1], elo_calc)
                r = np.random.random()
                if r < w[2]:            winner = ko_pool[i]
                elif r < w[2] + w[1]:   winner = ko_pool[i] if np.random.random() < 0.5 else ko_pool[i+1]
                else:                   winner = ko_pool[i+1]
                nxt.append(winner)
            ko_pool = nxt
        return ko_pool[0] if ko_pool else None

    def _elo_match_probs(self, team_a, team_b, elo_calc) -> np.ndarray:
        """Fast Elo-only [away, draw, home] estimate for bulk simulation."""
        ea = elo_calc.get_current_elo(team_a) if elo_calc else 1500.0
        eb = elo_calc.get_current_elo(team_b) if elo_calc else 1500.0
        exp_a = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))
        home_win = exp_a * 0.75
        draw     = 0.20
        away_win = max(0.0, 1.0 - home_win - draw)
        p = np.array([away_win, draw, home_win])
        return p / p.sum()
