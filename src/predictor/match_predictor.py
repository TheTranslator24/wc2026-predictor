# ==============================================================
# FILE: src/predictor/match_predictor.py
# PURPOSE: The high-level class you actually call — ties the whole stack together.
#
# USAGE:
#   from src.predictor.match_predictor import WC2026Predictor
#   p = WC2026Predictor().setup()          # ~2-3 min if models saved, longer first time
#   p.predict("Spain", "Germany")
#   p.predict("France", "Brazil", adjustments={"weather_factor": 0.7}, match_date="2026-06-29")
#   p.predict_group("C")                    # 6 matches + projected standings
#   p.predict_tournament()                  # championship % for all 48 teams
#   p.update_elo("Spain", "Germany", 2, 1)  # feed a real result back in
# ==============================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import WC2026_GROUPS, MODELS_DIR, MODEL_CONFIG
from src.security.data_validator import validate_match_input, validate_team_name
from src.data.collector import SecureDataCollector
from src.data.features import EloCalculator, FeatureEngineer
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMTrainer
from src.models.ensemble import EnsemblePredictor
from src.tracking.results_store import apply_to_elo

logger = logging.getLogger("wc2026.predictor.main")


class WC2026Predictor:
    """
    End-to-end system: data -> validation -> Elo -> features -> XGBoost + LSTM
    -> weighted ensemble -> Monte Carlo. Build once with setup(), then call predict().
    """

    def __init__(self, auto_load: bool = True):
        """auto_load=True reuses saved models on disk; False forces retraining."""
        self.auto_load = auto_load
        self.collector:  Optional[SecureDataCollector] = None
        self.elo_calc:   Optional[EloCalculator]       = None
        self.feat_eng:   Optional[FeatureEngineer]     = None
        self.xgb:        Optional[XGBoostPredictor]    = None
        self.lstm:       Optional[LSTMTrainer]         = None
        self.ensemble:   Optional[EnsemblePredictor]   = None
        self.results_df: Optional[pd.DataFrame]        = None
        self.is_ready = False
        logger.info("WC2026Predictor created — call .setup() to initialize")

    # ──────────────────────────────────────────────────────────
    def setup(self, force_retrain: bool = False) -> "WC2026Predictor":
        """
        Five-step initialization: data -> Elo -> features -> XGBoost -> LSTM,
        then assemble the ensemble. First run trains and saves both models;
        later runs load them in seconds. Returns self for chaining.
        """
        logger.info("=" * 55)
        logger.info("  WC2026 Predictor — Setup Starting")
        logger.info("=" * 55)

        logger.info("[1/5] Downloading / loading historical data...")
        self.collector  = SecureDataCollector()
        self.results_df = self.collector.load_historical_results()

        logger.info("[2/5] Computing Elo ratings (chronological)...")
        self.elo_calc    = EloCalculator()
        results_with_elo = self.elo_calc.calculate_all_elo(self.results_df)
        apply_to_elo(self.elo_calc)

        logger.info("[3/5] Initializing feature engineer...")
        self.feat_eng = FeatureEngineer(results_with_elo)

        # ── XGBoost: load if saved, else train + save ──────────
        xgb_path = MODELS_DIR / "xgboost_wc2026.pkl"
        if xgb_path.exists() and not force_retrain and self.auto_load:
            logger.info("[4/5] Loading saved XGBoost model...")
            self.xgb = XGBoostPredictor.load(xgb_path)
        else:
            logger.info("[4/5] Training XGBoost model...")
            X_train, y_train = self.feat_eng.build_training_dataset(results_with_elo, self.elo_calc)
            self.xgb = XGBoostPredictor().fit(X_train, y_train)
            self.xgb.save(xgb_path)

        # ── LSTM: load if saved, else build sequences + train ──
        lstm_path = MODELS_DIR / "lstm_wc2026.pt"
        if lstm_path.exists() and not force_retrain and self.auto_load:
            logger.info("[5/5] Loading saved LSTM model...")
            self.lstm = LSTMTrainer.load(lstm_path)
        else:
            logger.info("[5/5] Training LSTM model (CPU, a few minutes)...")
            self.lstm = LSTMTrainer()
            seqs, labels = self.lstm.build_sequences(results_with_elo, self.elo_calc)
            if len(seqs) > 500:                       # need enough samples to train
                self.lstm.train(seqs, labels)
                if self.lstm.is_trained:              # only persist a genuinely trained model
                    self.lstm.save(lstm_path)
                else:
                    logger.warning("LSTM did not finish training — running XGBoost-only.")
                    self.lstm = None
            else:
                logger.warning("Too few sequences — LSTM skipped, XGBoost-only")
                self.lstm = None

        self.ensemble = EnsemblePredictor(self.xgb, self.lstm)
        self.is_ready = True
        logger.info("=" * 55)
        logger.info("  WC2026 Predictor Ready — try .predict('Spain', 'Germany')")
        logger.info("=" * 55)
        return self

    # ──────────────────────────────────────────────────────────
    def predict(self, team_a: str, team_b: str,
                adjustments: Optional[dict] = None,
                match_date: Optional[str] = None) -> dict:
        """
        Predict one match. team_a is treated as home, team_b as away.
        adjustments keys (each 0.0 bad .. 1.0 ideal): weather_factor,
        pitch_condition, lineup_stability. match_date is "YYYY-MM-DD" (default today).
        Returns the full ensemble result dict + a 'match' metadata block.
        """
        if not self.is_ready:
            raise RuntimeError("Not initialized. Call .setup() first.")

        # Security gate: whitelist + sanitize every input before use.
        v = validate_match_input(team_a, team_b, adjustments)
        home_team, away_team, adj = v["home_team"], v["away_team"], v["adjustments"]
        pred_date = pd.Timestamp(match_date) if match_date else pd.Timestamp.today()
        logger.info(f"Predicting: {home_team} vs {away_team} | {pred_date.date()}")

        # Tabular features (uses live Elo via elo_calc).
        features = self.feat_eng.build_feature_row(
            home_team, away_team, pred_date, elo_calc=self.elo_calc, adjustments=adj
        )

        # LSTM sequence (only if the LSTM trained). build_match_sequence returns
        # (seq_len, 16); add a batch dimension -> (1, seq_len, 16).
        lstm_seq = None
        if self.lstm and self.lstm.is_trained:
            seq2d = self.lstm.build_match_sequence(self.feat_eng.df, home_team, away_team, pred_date)
            lstm_seq = seq2d[np.newaxis, ...]

        result = self.ensemble.predict_match(features, lstm_seq)

        # Attach human-readable match metadata.
        eh = self.elo_calc.get_current_elo(home_team)
        ea = self.elo_calc.get_current_elo(away_team)
        result["match"] = {
            "team_a": home_team, "team_b": away_team, "date": str(pred_date.date()),
            "elo_home": round(eh, 1), "elo_away": round(ea, 1), "elo_diff": round(eh - ea, 1),
        }
        logger.info(
            f"Result: {home_team} vs {away_team} -> {result['predicted_outcome']} "
            f"(conf={result['confidence']:.1%} | H:{result['home_win_prob']:.1%} "
            f"D:{result['draw_prob']:.1%} A:{result['away_win_prob']:.1%})"
        )
        return result

    # ──────────────────────────────────────────────────────────
    def predict_group(self, group_letter: str) -> dict:
        """Predict all 6 matches in a group and derive projected standings (top 2 advance)."""
        if not self.is_ready:
            raise RuntimeError("Not initialized. Call .setup() first.")
        group_letter = group_letter.upper()
        if group_letter not in WC2026_GROUPS:
            raise ValueError(f"Invalid group '{group_letter}'. Must be A-L.")

        teams = WC2026_GROUPS[group_letter]
        logger.info(f"Group {group_letter} prediction: {teams}")
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        matches_output = []
        for i, ta in enumerate(teams):
            for tb in teams[i+1:]:
                pred = self.predict(ta, tb)
                matches_output.append(pred)
                outcome = pred["predicted_outcome"]
                if   outcome == "Home Win": pts[ta] += 3; gd[ta] += 1; gd[tb] -= 1
                elif outcome == "Away Win": pts[tb] += 3; gd[tb] += 1; gd[ta] -= 1
                else:                       pts[ta] += 1; pts[tb] += 1
        standings = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        return {
            "group": group_letter, "teams": teams, "matches": matches_output,
            "projected_standings": [
                {"rank": i+1, "team": t, "points": pts[t], "goal_diff": gd[t]}
                for i, t in enumerate(standings)
            ],
            "advance": standings[:2],
        }

    # ──────────────────────────────────────────────────────────
    def predict_tournament(self) -> dict[str, float]:
        """Run the full-tournament Monte Carlo; returns {team: championship_%}."""
        if not self.is_ready:
            raise RuntimeError("Not initialized. Call .setup() first.")
        n_sims = MODEL_CONFIG["monte_carlo"]["tournament_sims"]
        return self.ensemble.simulate_tournament(WC2026_GROUPS, self.elo_calc, n_sims=n_sims)

    # ──────────────────────────────────────────────────────────
    def update_elo(self, team_a: str, team_b: str, score_a: int, score_b: int,
                   tournament: str = "FIFA World Cup") -> None:
        """
        Feed a REAL completed result back into the ratings so later predictions
        stay calibrated. Call after each match: update_elo("Spain","Germany",2,1).
        This is the manual half of the hybrid live-results approach.
        """
        if not self.is_ready:
            raise RuntimeError("Not initialized. Call .setup() first.")
        team_a = validate_team_name(team_a, "update_elo")
        team_b = validate_team_name(team_b, "update_elo")
        old_a = self.elo_calc.get_current_elo(team_a)
        old_b = self.elo_calc.get_current_elo(team_b)
        new_a, new_b = self.elo_calc.update_from_match(
            team_a, team_b, score_a, score_b, tournament, pd.Timestamp.today()
        )
        logger.info(
            f"Elo updated | {team_a} {score_a}-{score_b} {team_b} | "
            f"{team_a}: {old_a:.0f}->{new_a:.0f} ({new_a-old_a:+.0f}) | "
            f"{team_b}: {old_b:.0f}->{new_b:.0f} ({new_b-old_b:+.0f})"
        )