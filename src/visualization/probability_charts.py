# ==============================================================
# FILE: src/visualization/probability_charts.py
# PURPOSE: Dark-themed PNG charts for predictions.
#
# THREE FUNCTIONS:
#   plot_match_prediction()           — 4-panel match analysis
#   plot_championship_probabilities() — top-N championship bars
#   plot_group_standings()            — projected group table
#
# OUTPUT: PNG files in outputs/predictions/. Style: GitHub-dark background,
# neon accents. (Flag emojis may render as boxes depending on system fonts —
# purely cosmetic, never an error.)
# ==============================================================

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

from src.config import OUTPUTS_DIR, MODEL_CONFIG

logger = logging.getLogger("wc2026.visualization")

plt.style.use("dark_background")

BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
COLORS = {
    "home": "#2ecc71", "draw": "#f39c12", "away": "#e74c3c", "elo": "#9b59b6",
    "ci": "#3498db", "text": "#e6edf3", "gold": "#ffd700", "accent": "#1abc9c",
    "grey": "#8b949e",
}

FLAGS: dict[str, str] = {
    "Spain": "🇪🇸", "Germany": "🇩🇪", "France": "🇫🇷", "England": "🏴",
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "Netherlands": "🇳🇱", "Portugal": "🇵🇹",
    "Belgium": "🇧🇪", "Morocco": "🇲🇦", "Japan": "🇯🇵", "United States": "🇺🇸",
    "Uruguay": "🇺🇾", "Croatia": "🇭🇷", "Colombia": "🇨🇴", "South Korea": "🇰🇷",
    "Mexico": "🇲🇽", "Ecuador": "🇪🇨", "Norway": "🇳🇴", "Sweden": "🇸🇪",
    "Senegal": "🇸🇳", "Canada": "🇨🇦", "Austria": "🇦🇹", "Switzerland": "🇨🇭",
    "Australia": "🇦🇺", "Turkey": "🇹🇷", "Ivory Coast": "🇨🇮", "Tunisia": "🇹🇳",
    "Scotland": "🏴", "Iran": "🇮🇷", "Algeria": "🇩🇿", "Egypt": "🇪🇬",
    "Paraguay": "🇵🇾", "Ghana": "🇬🇭", "Cape Verde": "🇨🇻", "South Africa": "🇿🇦",
    "Saudi Arabia": "🇸🇦", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Iraq": "🇮🇶",
    "Czechia": "🇨🇿", "Bosnia and Herzegovina": "🇧🇦", "Panama": "🇵🇦", "Qatar": "🇶🇦",
    "New Zealand": "🇳🇿", "Jordan": "🇯🇴", "Haiti": "🇭🇹", "Curacao": "🇨🇼",
}


def _flag(team: str) -> str:
    return FLAGS.get(team, "")


def _save(fig: "plt.Figure", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    logger.info(f"Chart saved -> {path.name}")


def _style_axis(ax) -> None:
    ax.tick_params(colors=COLORS["grey"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.xaxis.label.set_color(COLORS["grey"])
    ax.yaxis.label.set_color(COLORS["grey"])


def _text(ax, x, y, text, color, size, weight="normal") -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=size, color=color, fontweight=weight)


# ==============================================================
def plot_match_prediction(prediction: dict, save_path: Optional[Path] = None,
                          show: bool = True):
    """4-panel chart: probability bars, Monte Carlo, Elo comparison, summary card."""
    m = prediction["match"]
    home_team, away_team = m["team_a"], m["team_b"]
    mc = prediction["monte_carlo"]
    h_flag, a_flag = _flag(home_team), _flag(away_team)

    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"{h_flag} {home_team}   vs   {away_team} {a_flag}\n"
                 f"FIFA World Cup 2026  ·  {m['date']}",
                 fontsize=15, color=COLORS["text"], fontweight="bold", y=0.98)

    bar_colors = [COLORS["home"], COLORS["draw"], COLORS["away"]]
    probs_pct = [prediction["home_win_prob"]*100, prediction["draw_prob"]*100,
                 prediction["away_win_prob"]*100]
    labels = [f"{home_team}\nWin", "Draw", f"{away_team}\nWin"]

    # Panel 1: probability bars
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG_PANEL)
    bars = ax1.barh(labels, probs_pct, color=bar_colors, height=0.55,
                    edgecolor="#ffffff22", linewidth=0.5)
    for bar, pct in zip(bars, probs_pct):
        ax1.text(bar.get_width()+1.0, bar.get_y()+bar.get_height()/2, f"{pct:.1f}%",
                 va="center", fontsize=12, color=COLORS["text"], fontweight="bold")
    outcome = prediction["predicted_outcome"]
    hi = 0 if "Home" in outcome else (2 if "Away" in outcome else 1)
    bars[hi].set_edgecolor(COLORS["gold"]); bars[hi].set_linewidth(2.5)
    ax1.set_xlim(0, 105)
    ax1.set_title("Outcome Probabilities", color=COLORS["accent"], fontsize=12, pad=8)
    ax1.set_xlabel("Probability (%)", color=COLORS["grey"]); _style_axis(ax1)

    # Panel 2: Monte Carlo
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG_PANEL)
    mc_pcts = [mc["home_win_pct"], mc["draw_pct"], mc["away_win_pct"]]
    mc_cnts = [mc["home_win_count"], mc["draw_count"], mc["away_win_count"]]
    mc_cis  = [mc["home_win_95ci"], mc["draw_95ci"], mc["away_win_95ci"]]
    mc_lbls = [f"{home_team} Win", "Draw", f"{away_team} Win"]
    bars2 = ax2.bar(mc_lbls, mc_cnts, color=bar_colors, edgecolor="#ffffff22",
                    linewidth=0.5, alpha=0.88, width=0.55)
    for bar, pct, ci in zip(bars2, mc_pcts, mc_cis):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
                 f"{pct}%\n[{ci[0]}–{ci[1]}%]", ha="center", fontsize=8.5, color=COLORS["text"])
    ax2.set_title(f"Monte Carlo  ({mc['n_simulations']:,} simulations)",
                  color=COLORS["accent"], fontsize=12, pad=8)
    ax2.set_ylabel("Simulated Outcomes", color=COLORS["grey"])
    ax2.text(0.99, 0.01, "[ ] = 95% Wilson CI", transform=ax2.transAxes,
             fontsize=7, color=COLORS["grey"], ha="right", va="bottom")
    _style_axis(ax2)

    # Panel 3: Elo comparison
    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(BG_PANEL)
    elo_h, elo_a = m["elo_home"], m["elo_away"]
    elo_c = [COLORS["home"] if elo_h >= elo_a else COLORS["away"],
             COLORS["away"] if elo_h >= elo_a else COLORS["home"]]
    bars3 = ax3.barh([home_team, away_team], [elo_h, elo_a], color=elo_c,
                     height=0.45, edgecolor="#ffffff22")
    for bar, val in zip(bars3, [elo_h, elo_a]):
        ax3.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, f"{val:.0f}",
                 va="center", fontsize=12, color=COLORS["text"], fontweight="bold")
    ax3.set_xlim(max(1100, min(elo_h, elo_a)-150), max(elo_h, elo_a)+150)
    ax3.set_title(f"Elo Ratings  (Δ = {m['elo_diff']:+.0f})",
                  color=COLORS["accent"], fontsize=12, pad=8)
    _style_axis(ax3)

    # Panel 4: summary card
    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(BG_PANEL); ax4.axis("off")
    conf_pct = prediction["confidence"]*100
    xgb_p = prediction["xgboost_prediction"]; adj = prediction["adjustments_applied"]
    _text(ax4, 0.5, 0.94, "PREDICTION", COLORS["accent"], 11, "bold")
    _text(ax4, 0.5, 0.82, outcome.upper(), COLORS["gold"], 17, "bold")
    _text(ax4, 0.5, 0.70, f"Confidence: {conf_pct:.1f}%", COLORS["text"], 11)
    _text(ax4, 0.5, 0.58, "Model Breakdown (H / D / A):", COLORS["grey"], 9, "bold")
    _text(ax4, 0.5, 0.49,
          f"XGBoost: {xgb_p['home_win']*100:.1f}% / {xgb_p['draw']*100:.1f}% / {xgb_p['away_win']*100:.1f}%",
          COLORS["grey"], 8.5)
    active = {k: v for k, v in adj.items() if v < 0.999}
    if active:
        adj_str = "Adjustments applied:\n" + "\n".join(
            f"  {k.replace('_',' ').title()}: {v:.0%}" for k, v in active.items())
        _text(ax4, 0.5, 0.32, adj_str, "#f39c12", 8)
    _text(ax4, 0.5, 0.06, "Probabilistic estimate. Football is uncertain.", COLORS["grey"], 7)

    if save_path is None:
        save_path = OUTPUTS_DIR / "predictions" / \
            f"{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}.png"
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ==============================================================
def plot_championship_probabilities(probs: dict[str, float], top_n: int = 20,
                                    save_path: Optional[Path] = None, show: bool = True):
    """Horizontal bar chart of the top-N championship probabilities (green=high, red=low)."""
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    teams = [f"{_flag(t)} {t}" for t, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DARK); ax.set_facecolor(BG_PANEL)
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=0, vmax=max(values) if values else 1)
    colors = [cmap(norm(v)) for v in values[::-1]]
    bars = ax.barh(teams[::-1], values[::-1], color=colors, height=0.65, edgecolor="#ffffff15")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width()+0.15, bar.get_y()+bar.get_height()/2, f"{val:.1f}%",
                va="center", fontsize=9.5, color=COLORS["text"], fontweight="bold")
    ax.set_xlim(0, (max(values) if values else 1)*1.25)
    ax.set_title("WC2026 Championship Probability Estimates\n"
                 f"Based on {MODEL_CONFIG['monte_carlo']['tournament_sims']:,} "
                 f"full-tournament Monte Carlo simulations",
                 color=COLORS["text"], fontsize=14, pad=12)
    ax.set_xlabel("Championship Probability (%)", color=COLORS["grey"]); _style_axis(ax)
    fig.text(0.5, 0.01,
             "Probabilistic model. Best football models reach ~55-60% accuracy. Wide error bars apply.",
             ha="center", color=COLORS["grey"], fontsize=8.5)
    if save_path is None:
        save_path = OUTPUTS_DIR / "predictions" / "championship_probabilities.png"
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ==============================================================
def plot_group_standings(group_pred: dict, save_path: Optional[Path] = None, show: bool = True):
    """Projected group standings; advancing teams green, eliminated red."""
    group = group_pred["group"]
    standings = group_pred["projected_standings"]
    advance = group_pred["advance"]
    teams_disp = [f"{_flag(s['team'])} {s['team']}" for s in standings]
    points = [s["points"] for s in standings]
    gdiffs = [s["goal_diff"] for s in standings]
    adv_set = set(advance)
    colors = [COLORS["home"] if s["team"] in adv_set else COLORS["away"] for s in standings]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG_DARK); ax.set_facecolor(BG_PANEL)
    bars = ax.barh(teams_disp[::-1], points[::-1], color=colors[::-1], height=0.5,
                   edgecolor="#ffffff15")
    for bar, pts, gd in zip(bars, points[::-1], gdiffs[::-1]):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f"{pts} pts  (GD {gd:+d})", va="center", fontsize=10, color=COLORS["text"])
    ax.set_xlim(0, 12)
    ax.set_title(f"Group {group}  —  Projected Standings\n"
                 + "  ".join(f"{t}" for t in advance) + "  advance",
                 color=COLORS["text"], fontsize=13, pad=10)
    ax.set_xlabel("Projected Points", color=COLORS["grey"])
    legend = [mpatches.Patch(color=COLORS["home"], label="Advances"),
              mpatches.Patch(color=COLORS["away"], label="Eliminated")]
    ax.legend(handles=legend, loc="lower right", facecolor="#21262d",
              edgecolor="#30363d", labelcolor=COLORS["text"])
    _style_axis(ax)
    if save_path is None:
        save_path = OUTPUTS_DIR / "predictions" / f"group_{group}_standings.png"
    _save(fig, save_path)
    if show:
        plt.show()
    return fig
