# ==============================================================
# FILE: src/tracking/scorer.py
# PURPOSE: Log every prediction, then grade it against reality.
#
# THE OVERLOOKED IDEA (your "game-changer"):
#   Amateurs measure ACCURACY ("did I call the winner?"). Pros also measure
#   CALIBRATION ("when I said 70%, did it happen ~70% of the time?"). A model
#   can be accurate but wildly overconfident, or modestly accurate but perfectly
#   calibrated — and only calibration tells you whether the probabilities mean
#   anything. We track both.
#
# THREE METRICS:
#   hit_rate  — fraction of matches where the top pick was correct (0-1)
#   brier     — multiclass Brier score: mean Σ(p_k - y_k)^2, LOWER is better
#               (0 = perfect, ~0.66 = random 3-way). Measures probability quality.
#   log_loss  — penalizes confident wrong calls harshly, LOWER is better.
#   + a reliability table: predicted-confidence bins vs actual hit rate.
#
# FILES:
#   data/prediction_log.csv  (one row per prediction, actual filled in later)
# ==============================================================

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger("wc2026.tracking.scorer")

LOG_PATH = DATA_DIR / "prediction_log.csv"
_OUTCOMES = ["Away Win", "Draw", "Home Win"]      # index 0,1,2
_COLUMNS = ["logged_at", "match_date", "home_team", "away_team",
            "p_away", "p_draw", "p_home", "predicted", "actual"]


def _load() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        # Blank 'actual' values load back as NaN, which makes pandas type the
        # whole column float64 — then writing a string into it fails. Force the
        # text columns to object/str with "" for blanks.
        for c in ("logged_at", "match_date", "home_team", "away_team", "predicted", "actual"):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        return df
    return pd.DataFrame(columns=_COLUMNS)


def log_prediction(prediction: dict) -> None:
    """
    Record a prediction BEFORE the match is played. Pass the dict returned by
    WC2026Predictor.predict(). 'actual' is left blank until you fill it in.
    Re-logging the same fixture+date overwrites (so you always store your
    latest, closest-to-kickoff forecast).
    """
    m = prediction["match"]
    df = _load()
    mask = ~((df["match_date"] == m["date"]) &
             (df["home_team"] == m["team_a"]) & (df["away_team"] == m["team_b"]))
    df = df[mask]
    row = {
        "logged_at":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "match_date": m["date"], "home_team": m["team_a"], "away_team": m["team_b"],
        "p_away": round(prediction["away_win_prob"], 4),
        "p_draw": round(prediction["draw_prob"], 4),
        "p_home": round(prediction["home_win_prob"], 4),
        "predicted": prediction["predicted_outcome"], "actual": "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)
    logger.info(f"Logged prediction | {m['team_a']} vs {m['team_b']} -> {prediction['predicted_outcome']}")


def record_actual(home_team: str, away_team: str, match_date: str, actual_outcome: str) -> None:
    """
    Fill in what actually happened. actual_outcome must be one of
    'Home Win' / 'Draw' / 'Away Win'. Match is found by teams + date.
    """
    if actual_outcome not in _OUTCOMES:
        raise ValueError(f"actual_outcome must be one of {_OUTCOMES}")
    df = _load()
    sel = ((df["match_date"] == match_date) &
           (df["home_team"] == home_team) & (df["away_team"] == away_team))
    if not sel.any():
        raise ValueError(f"No logged prediction for {home_team} vs {away_team} on {match_date}")
    df.loc[sel, "actual"] = actual_outcome
    df.to_csv(LOG_PATH, index=False)
    logger.info(f"Recorded actual | {home_team} vs {away_team} = {actual_outcome}")


def record_actual_from_score(home_team: str, away_team: str, match_date: str,
                             home_score: int, away_score: int) -> None:
    """Convenience: derive the outcome from a scoreline and record it."""
    outcome = ("Home Win" if home_score > away_score else
               "Away Win" if home_score < away_score else "Draw")
    record_actual(home_team, away_team, match_date, outcome)


def score() -> dict:
    """
    Compute accuracy + calibration over all GRADED predictions (those with an
    'actual'). Returns a metrics dict; also logs a readable summary.
    """
    df = _load()
    graded = df[df["actual"].isin(_OUTCOMES)].copy()
    n = len(graded)
    if n == 0:
        logger.info("No graded predictions yet — record some actuals first.")
        return {"n": 0}

    probs = graded[["p_away", "p_draw", "p_home"]].to_numpy(dtype=float)
    y_idx = graded["actual"].map({o: i for i, o in enumerate(_OUTCOMES)}).to_numpy()
    pred_idx = probs.argmax(axis=1)

    # one-hot truth for Brier / log-loss
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y_idx] = 1.0

    hit_rate = float((pred_idx == y_idx).mean())
    brier    = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    eps      = 1e-15
    log_loss = float(-np.mean(np.log(np.clip(probs[np.arange(n), y_idx], eps, 1.0))))

    # ── Reliability (calibration) table ───────────────────────
    # Bin by the model's CONFIDENCE (its top probability). In each bin compare
    # mean confidence to the actual hit rate; close = well calibrated.
    conf = probs.max(axis=1)
    correct = (pred_idx == y_idx).astype(float)
    bins = [0.33, 0.45, 0.55, 0.65, 0.75, 1.01]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        table.append({
            "confidence_bin": f"{int(lo*100)}-{int(hi*100)}%",
            "n": int(m.sum()),
            "avg_confidence": round(float(conf[m].mean()) * 100, 1),
            "actual_hit_rate": round(float(correct[m].mean()) * 100, 1),
        })

    logger.info(
        f"SCORECARD | n={n} | hit_rate={hit_rate:.1%} | "
        f"Brier={brier:.3f} (lower=better, ~0.66=random) | log_loss={log_loss:.3f}"
    )
    return {
        "n": n, "hit_rate": hit_rate, "brier": brier, "log_loss": log_loss,
        "reliability": table,
    }


def print_scorecard() -> None:
    """Human-readable summary for the terminal."""
    s = score()
    if s["n"] == 0:
        print("  No graded predictions yet."); return
    print("\n  ── FORECAST SCORECARD ─────────────────────────")
    print(f"  Graded predictions : {s['n']}")
    print(f"  Hit rate           : {s['hit_rate']:.1%}   (>52% is good for football)")
    print(f"  Brier score        : {s['brier']:.3f}   (lower better; ~0.66 = random)")
    print(f"  Log loss           : {s['log_loss']:.3f}   (lower better)")
    print("  ── Calibration (confidence vs reality) ────────")
    print(f"  {'bin':<10}{'n':>4}{'said':>9}{'actual':>9}")
    for r in s["reliability"]:
        print(f"  {r['confidence_bin']:<10}{r['n']:>4}{r['avg_confidence']:>8.1f}%{r['actual_hit_rate']:>8.1f}%")
    print("  (well calibrated = 'said' ≈ 'actual' in each row)\n")
