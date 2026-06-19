# ==============================================================
# FILE: src/models/xgboost_model.py
# PURPOSE: Gradient-boosted-tree classifier for match outcomes.
#
# WHY XGBOOST:
#   - Consistently the strongest model class for tabular sports data
#   - Exact SHAP explanations (TreeExplainer) — you can see WHY it predicts
#   - "hist" tree method runs natively on the M1 CPU across all cores
#   - Early stopping handles "when to stop training" for you
#
# OUTPUT: 3-class probabilities -> [Away Win %, Draw %, Home Win %]
# SAVED AS: models/xgboost_wc2026.pkl
#
# THE DRAW FIX (this is the important change):
#   Draws are the minority outcome (~22-24%). With no correction, the
#   model maximizes raw accuracy by almost never predicting a draw
#   (your earlier run: Draw recall 0.01). We now pass BALANCED sample
#   weights to fit(), so every class contributes equally to the loss and
#   the model is forced to take draws seriously. Expect draw recall to
#   rise to ~0.20-0.35 — football's real, hard ceiling for ties.
# ==============================================================

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight   # <-- the draw fix
from sklearn.metrics import classification_report, log_loss, accuracy_score

from src.config import MODEL_CONFIG, MODELS_DIR

logger = logging.getLogger("wc2026.models.xgboost")

LABEL_MAP: dict[int, str] = {0: "Away Win", 1: "Draw", 2: "Home Win"}


class XGBoostPredictor:
    """
    3-class XGBoost classifier: Away Win / Draw / Home Win.

    Training time on an M1:  a few minutes (early stopping usually ends it
    well before the 1000-tree cap). Prediction: sub-millisecond per match.
    Realistic validation accuracy: ~54-58% (football's ceiling is ~60%).
    """

    def __init__(self):
        p = MODEL_CONFIG["xgboost"]
        self.model = xgb.XGBClassifier(
            n_estimators          = p["n_estimators"],
            learning_rate         = p["learning_rate"],
            max_depth             = p["max_depth"],
            min_child_weight      = p["min_child_weight"],
            subsample             = p["subsample"],
            colsample_bytree      = p["colsample_bytree"],
            gamma                 = p["gamma"],
            reg_alpha             = p["reg_alpha"],
            reg_lambda            = p["reg_lambda"],
            objective             = "multi:softprob",   # probabilities, not hard labels
            num_class             = 3,
            eval_metric           = "mlogloss",
            tree_method           = "hist",             # fastest CPU method on M1
            n_jobs                = -1,                 # all cores
            random_state          = 42,
            early_stopping_rounds = p["early_stopping_rounds"],
        )
        self.feature_columns: list[str] = []
        self.is_trained: bool = False
        logger.info(
            f"XGBoostPredictor init | trees={p['n_estimators']} | "
            f"lr={p['learning_rate']} | depth={p['max_depth']} | hist | M1 CPU"
        )

    # ──────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostPredictor":
        """
        Train with an 80/20 stratified split and balanced class weights.

        Early stopping halts when validation log-loss stops improving for
        early_stopping_rounds consecutive rounds — no manual epoch count.
        """
        self.feature_columns = list(X.columns)
        logger.info(f"Training XGBoost | samples={len(X):,} | features={len(self.feature_columns)}")

        # Stratify keeps the away/draw/home proportions identical in both splits.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── THE DRAW FIX ───────────────────────────────────────
        # "balanced" assigns each sample a weight inversely proportional to its
        # class frequency: rare draws get UP-weighted, common home wins DOWN.
        # This removes the model's incentive to ignore draws for easy accuracy.
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        logger.info(
            "Applying balanced class weights | "
            f"mean draw weight={sample_weight[y_train.values == 1].mean():.2f} "
            f"(vs ~1.0 baseline)"
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,          # <-- without this, draws collapse
            eval_set=[(X_val, y_val)],
            verbose=200,
        )
        self.is_trained = True

        # ── Validation metrics ─────────────────────────────────
        val_probs = self.model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc   = accuracy_score(y_val, val_preds)
        val_ll    = log_loss(y_val, val_probs)
        logger.info(
            f"Training complete | best_iteration={self.model.best_iteration} | "
            f"val_accuracy={val_acc:.3f} | val_log_loss={val_ll:.4f}"
        )
        # The per-class report is where you confirm draw recall is no longer ~0.
        logger.info("\n" + classification_report(
            y_val, val_preds, target_names=list(LABEL_MAP.values()), zero_division=0
        ))
        return self

    # ──────────────────────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probabilities for one or more matches, shape (n, 3):
        column 0 = Away Win, 1 = Draw, 2 = Home Win.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        return self.model.predict_proba(X[self.feature_columns])   # exact column order

    def predict_single(self, features: dict) -> dict:
        """Predict one match from a feature dict (FeatureEngineer.build_feature_row)."""
        probs = self.predict_proba(pd.DataFrame([features]))[0]
        best  = int(np.argmax(probs))
        return {
            "away_win_prob": float(probs[0]),
            "draw_prob":     float(probs[1]),
            "home_win_prob": float(probs[2]),
            "predicted":     LABEL_MAP[best],
            "confidence":    float(probs[best]),
        }

    # ──────────────────────────────────────────────────────────
    def explain(self, X: pd.DataFrame, max_display: int = 15) -> dict:
        """
        SHAP feature attributions: which features drove the prediction, by how much.
        TreeExplainer is exact for tree models (milliseconds, not minutes).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Cannot compute SHAP values.")
        X_ordered = X[self.feature_columns]
        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer(X_ordered)
        # Multiclass shape: (n_samples, n_features, n_classes) -> mean |value|
        mean_abs = np.abs(shap_vals.values).mean(axis=(0, 2))
        top_idx  = np.argsort(mean_abs)[::-1][:max_display]
        return {
            "shap_values":   shap_vals,
            "feature_names": self.feature_columns,
            "top_features":  [(self.feature_columns[i], float(mean_abs[i])) for i in top_idx],
        }

    # ──────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        """Serialize the model + its feature column order to disk."""
        if path is None:
            path = MODELS_DIR / "xgboost_wc2026.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_columns": self.feature_columns, "label_map": LABEL_MAP},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"XGBoost saved | {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostPredictor":
        """Load a saved model (skips __init__ to avoid rebuilding an empty estimator)."""
        if path is None:
            path = MODELS_DIR / "xgboost_wc2026.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            # RISK ASSESSMENT: unpickles ONLY our own locally-produced model file
            # — gitignored, never committed, published only as an integrity-verified
            # release artifact. Never fed untrusted input, so the pickle
            # deserialization risk bandit flags does not apply here.
            payload = pickle.load(f)  # nosec B301 - own local file only; never untrusted input
        inst = cls.__new__(cls)
        inst.model           = payload["model"]
        inst.feature_columns = payload["feature_columns"]
        inst.is_trained      = True
        logger.info(f"XGBoost loaded | features={len(inst.feature_columns)} | {path.name}")
        return inst
