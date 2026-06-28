# ==============================================================
# FILE: src/visualization/advanced_charts.py
# PURPOSE: Two upgraded visuals built on team colors + the scorer.
#
#   plot_reliability_diagram() — THE game-changer chart. Plots predicted
#       confidence vs actual hit rate. The diagonal is "perfectly calibrated";
#       points above it = underconfident, below = overconfident. Almost no
#       hobby predictor shows this, and it's the truest measure of a model.
#
#   plot_match_colored() — match prediction using each team's real colors
#       (no emoji tofu) and, if you've downloaded them, real flag images.
#
# OUTPUT: PNGs in outputs/predictions/.
# ==============================================================

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

from src.config import OUTPUTS_DIR
from src.visualization.team_identity import color, label, flag_path, has_flag

logger = logging.getLogger("wc2026.visualization.advanced")

plt.style.use("dark_background")
BG_DARK, BG_PANEL = "#0d1117", "#161b22"
TXT, GREY, ACC = "#e6edf3", "#8b949e", "#1abc9c"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    logger.info(f"Chart saved -> {path.name}")


def _flag_image(team: str, zoom: float = 0.35):
    """Return an OffsetImage of the team's flag, or None if not downloaded."""
    if not has_flag(team):
        return None
    try:
        img = plt.imread(str(flag_path(team)))
        return OffsetImage(img, zoom=zoom)
    except Exception:
        return None


# ==============================================================
# THE CALIBRATION RELIABILITY DIAGRAM  (the differentiator)
# ==============================================================
def plot_reliability_diagram(scorecard: dict, save_path: Optional[Path] = None, show: bool = False):
    """
    Plot model confidence vs actual hit rate from scorer.score() output.

    Reading it:
      - The dashed diagonal = perfect calibration (said 70% -> happened 70%).
      - Bubble size = how many predictions fell in that confidence bin.
      - Points BELOW the line = overconfident; ABOVE = underconfident.
    """
    rel = scorecard.get("reliability", [])
    if not rel:
        logger.warning("No reliability data — grade some predictions first.")
        return None

    said   = np.array([r["avg_confidence"] for r in rel])
    actual = np.array([r["actual_hit_rate"] for r in rel])
    counts = np.array([r["n"] for r in rel])

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)

    # perfect-calibration reference line
    ax.plot([33, 100], [33, 100], "--", color=GREY, lw=1.2, label="Perfect calibration")

    # bubbles sized by sample count, colored by over/under confidence
    for s, a, c in zip(said, actual, counts):
        col = "#2ecc71" if a >= s else "#e74c3c"   # green=met/beat, red=overconfident
        ax.scatter(s, a, s=80 + c * 25, color=col, alpha=0.85, edgecolor=TXT, linewidth=0.6, zorder=3)
        ax.annotate(f"n={c}", (s, a), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color=GREY)

    ax.set_xlim(33, 100); ax.set_ylim(33, 100)
    ax.set_xlabel("Model said (avg confidence %)", color=GREY)
    ax.set_ylabel("Actually happened (hit rate %)", color=GREY)
    ax.set_title("Calibration / Reliability\n"
                 f"{scorecard['n']} graded predictions  ·  Brier {scorecard['brier']:.3f}",
                 color=TXT, fontsize=13, pad=12)
    ax.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor=TXT)
    ax.tick_params(colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.02, "Below the line = overconfident · Above = underconfident · "
             "on the line = honest probabilities", ha="center", color=GREY, fontsize=8)

    if save_path is None:
        save_path = OUTPUTS_DIR / "predictions" / "calibration_reliability.png"
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ==============================================================
# COLOR + FLAG MATCH CHART
# ==============================================================
def plot_match_colored(prediction: dict, save_path: Optional[Path] = None, show: bool = False):
    """
    Probability chart using each team's real colors (and flag if downloaded).
    Home bar = home team's primary color, Away bar = away team's primary color,
    Draw = neutral grey — instantly readable, no emoji.
    """
    m = prediction["match"]
    home, away = m["team_a"], m["team_b"]
    probs = [prediction["home_win_prob"] * 100,
             prediction["draw_prob"] * 100,
             prediction["away_win_prob"] * 100]
    bars_c = [color(home, "primary"), "#9aa0a6", color(away, "primary")]
    labels = [f"{label(home)}\nWin", "Draw", f"{label(away)}\nWin"]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)
    bars = ax.barh(labels, probs, color=bars_c, height=0.6, edgecolor=TXT, linewidth=0.6)

    # highlight the predicted outcome
    out = prediction["predicted_outcome"]
    hi = 0 if "Home" in out else (2 if "Away" in out else 1)
    bars[hi].set_edgecolor(TXT); bars[hi].set_linewidth(3)

    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=13, color=TXT, fontweight="bold")

    # optional flag icons at the left edge of the home/away bars
    for team, bar in ((home, bars[0]), (away, bars[2])):
        oi = _flag_image(team, zoom=0.30)
        if oi is not None:
            ab = AnnotationBbox(oi, (3, bar.get_y() + bar.get_height() / 2),
                                frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)

    ax.set_xlim(0, 108)
    ax.set_title(f"{label(home)}  vs  {label(away)}\n"
                 f"FIFA World Cup 2026  ·  {m['date']}  ·  model output: {out.capitalize()} "
                 f"({prediction['confidence']:.0%})",
                 color=TXT, fontsize=13, pad=12)
    ax.set_xlabel("Probability (%)", color=GREY)
    ax.tick_params(colors=GREY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.02,
             "For research and education only. Not betting advice and not a "
             "prediction of any specific outcome. Football is uncertain.",
             ha="center", color=GREY, fontsize=7)

    if save_path is None:
        save_path = OUTPUTS_DIR / "predictions" / \
            f"{home.replace(' ', '_')}_vs_{away.replace(' ', '_')}_colored.png"
    _save(fig, save_path)
    if show:
        plt.show()
    return fig