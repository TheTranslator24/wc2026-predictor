# ==============================================================
# FILE: src/config.py
# PURPOSE: Single source of truth for every project setting.
# SECURITY: Contains NO secrets. Secrets live in .env (gitignored).
#           This file is SAFE to commit to GitHub.
#
# WHAT LIVES HERE:
#   - All 48 WC2026 teams in their 12 groups
#   - FIFA Rankings (top-4 verified vs official ranking, June 2026)
#   - Confederation membership + strength priors
#   - The (whitelisted) data source URLs
#   - Model hyperparameters (XGBoost / LSTM / Monte Carlo / ensemble)
#   - Elo rating system parameters
#   - Security settings
#   - Logging setup
#
# WHY ONE FILE: a "single source of truth" means you change a
# number in exactly one place and the whole pipeline obeys it.
# That is both a maintainability and a security property (no
# divergent copies of a setting drifting out of sync).
# ==============================================================

import os                          # stdlib: read environment variables
import logging                     # stdlib: structured audit logging
from pathlib import Path           # stdlib: safe cross-platform file paths
from dotenv import load_dotenv     # python-dotenv: load .env without exposing secrets

# ── Load environment variables ────────────────────────────────
# load_dotenv() reads a local .env file if present. If it is
# absent it silently does nothing (no crash). Every secret stays
# in .env, which .gitignore keeps off GitHub.
load_dotenv()

# ── Project directory layout ──────────────────────────────────
# Path(__file__)        -> absolute path to THIS file (src/config.py)
# .parent               -> the src/ directory
# .parent.parent        -> the project root (wc2026-predictor/)
BASE_DIR = Path(__file__).parent.parent     # project root

DATA_DIR      = BASE_DIR / "data"           # all data lives under here
RAW_DATA_DIR  = DATA_DIR / "raw"            # downloaded, never edited by hand
PROCESSED_DIR = DATA_DIR / "processed"      # engineered features, cached
MODELS_DIR    = BASE_DIR / "models"         # saved trained model files
OUTPUTS_DIR   = BASE_DIR / "outputs"        # charts + prediction results
LOGS_DIR      = BASE_DIR / "logs"           # the security/audit log trail

# Auto-create each directory on import. exist_ok=True means "do not
# error if it already exists"; parents=True creates intermediate dirs.
for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ==============================================================
# WC2026 TOURNAMENT DATA
# Source: official FIFA World Cup 2026 final draw (Dec 5, 2025).
# 12 groups of 4 teams = 48 teams. Host nations: USA, Mexico, Canada.
# ==============================================================

WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Reverse lookup so you can ask "what group is Spain in?" in O(1).
# Built once from WC2026_GROUPS so the two can never disagree.
TEAM_TO_GROUP: dict[str, str] = {
    team: group
    for group, teams in WC2026_GROUPS.items()
    for team in teams
}

# All 48 teams as a frozenset. frozenset = immutable set: fast
# membership tests (the security whitelist) and it cannot be
# mutated by accident at runtime.
ALL_WC2026_TEAMS: frozenset[str] = frozenset(
    team for teams in WC2026_GROUPS.values() for team in teams
)

# ── FIFA Rankings (June 2026) ─────────────────────────────────
# Lower number = stronger team. Top-4 (Spain, Argentina, France,
# England) verified against the official June 2026 ranking.
# UPDATE the rest from fifa.com/fifa-world-ranking before a run.
FIFA_RANKINGS: dict[str, int] = {
    "Spain": 1,          "Argentina": 2,      "France": 3,
    "England": 4,        "Germany": 5,        "Portugal": 6,
    "Brazil": 7,         "Netherlands": 8,    "Belgium": 9,
    "Morocco": 10,       "Japan": 11,         "United States": 12,
    "Uruguay": 13,       "Croatia": 14,       "Colombia": 15,
    "South Korea": 16,   "Mexico": 17,        "Ecuador": 18,
    "Norway": 19,        "Sweden": 20,        "Senegal": 21,
    "Canada": 22,        "Austria": 23,       "Switzerland": 24,
    "Australia": 25,     "Turkey": 26,        "Ivory Coast": 27,
    "Tunisia": 28,       "Scotland": 29,      "Iran": 30,
    "Algeria": 31,       "Egypt": 32,         "Paraguay": 33,
    "Ghana": 34,         "Cape Verde": 35,    "South Africa": 36,
    "Saudi Arabia": 37,  "DR Congo": 38,      "Uzbekistan": 39,
    "Iraq": 40,          "Czechia": 41,       "Bosnia and Herzegovina": 42,
    "Panama": 43,        "Qatar": 44,         "New Zealand": 45,
    "Jordan": 46,        "Haiti": 47,         "Curacao": 48,
}

# ── Confederation membership ──────────────────────────────────
# Used for a "confederation strength" prior feature and for
# grouping analysis. Six confederations, all 48 teams mapped.
CONFEDERATION: dict[str, str] = {
    # UEFA — Europe
    "Spain": "UEFA",    "France": "UEFA",     "England": "UEFA",
    "Germany": "UEFA",  "Portugal": "UEFA",   "Netherlands": "UEFA",
    "Belgium": "UEFA",  "Croatia": "UEFA",    "Norway": "UEFA",
    "Sweden": "UEFA",   "Scotland": "UEFA",   "Austria": "UEFA",
    "Switzerland": "UEFA", "Czechia": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Turkey": "UEFA",
    # CONMEBOL — South America
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL",  "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    # CAF — Africa
    "Morocco": "CAF",    "Senegal": "CAF",    "Ivory Coast": "CAF",
    "Tunisia": "CAF",    "Egypt": "CAF",      "Ghana": "CAF",
    "South Africa": "CAF", "Cape Verde": "CAF", "DR Congo": "CAF",
    "Algeria": "CAF",
    # AFC — Asia
    "Japan": "AFC",      "South Korea": "AFC",  "Iran": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC",  "Iraq": "AFC",
    "Jordan": "AFC",     "Qatar": "AFC",        "Uzbekistan": "AFC",
    # CONCACAF — North/Central America + Caribbean
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF",        "Haiti": "CONCACAF",  "Curacao": "CONCACAF",
    # OFC — Oceania
    "New Zealand": "OFC",
}

# Historical average confederation strength (0.0–1.0), a soft prior
# derived from FIFA rankings + World Cup performance. Only a nudge —
# the learned models do the heavy lifting.
CONFEDERATION_STRENGTH: dict[str, float] = {
    "UEFA":     0.85,
    "CONMEBOL": 0.82,
    "CAF":      0.65,
    "AFC":      0.62,
    "CONCACAF": 0.60,
    "OFC":      0.45,
}

# ── Data sources (SECURITY: strict whitelist) ─────────────────
# ONLY these pre-approved HTTPS URLs may be downloaded. collector.py
# refuses any URL not in this dict. This prevents a tampered config
# or argument from pointing the downloader at a malicious host.
#
# CORRECTED: the repository slug is "international_results"
# (underscore). The old "international-football" URL 404s.
# License: CC0 Public Domain (free to use, no attribution required).
DATA_SOURCES: dict[str, dict] = {
    "historical_results": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "local_path": RAW_DATA_DIR / "international_results.csv",
        "description": "International football results, 1872–2026 (CC0)",
        "sha256": None,   # auto-filled after first verified download
    },
    "goalscorers": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
        "local_path": RAW_DATA_DIR / "goalscorers.csv",
        "description": "International football goalscorer data (CC0)",
        "sha256": None,
    },
}

# ── Model hyperparameters ─────────────────────────────────────
MODEL_CONFIG: dict = {
    "xgboost": {
        "n_estimators":          1000,   # max trees (early stopping cuts this)
        "learning_rate":         0.05,   # smaller = more conservative steps
        "max_depth":             6,      # tree depth; 6 is a sports-data sweet spot
        "min_child_weight":      5,      # min samples per leaf -> less overfit
        "subsample":             0.8,    # row sampling per tree -> regularization
        "colsample_bytree":      0.8,    # feature sampling per tree
        "gamma":                 0.1,    # min loss reduction to split
        "reg_alpha":             0.1,    # L1 regularization
        "reg_lambda":            1.0,    # L2 regularization
        "objective":      "multi:softprob",  # output probabilities for 3 classes
        "num_class":             3,      # 0=Away Win, 1=Draw, 2=Home Win
        "eval_metric":      "mlogloss",  # multiclass log loss
        "tree_method":         "hist",   # fastest CPU method; native-fast on M1
        "n_jobs":                -1,     # use all CPU cores
        "random_state":          42,     # reproducibility seed
        "early_stopping_rounds": 50,     # stop if val loss flat for 50 rounds
        # NOTE: class imbalance (draws are the minority) is handled at
        # fit() time via sklearn's balanced sample weights — see
        # xgboost_model.py in Batch 4. That is the fix for the
        # "draw recall = 0.01" collapse.
    },
    "lstm": {
        "sequence_length":  10,     # look back over each team's last 10 matches
        "hidden_size":      128,    # LSTM hidden-state width
        "num_layers":       2,      # stacked LSTM layers
        "dropout":          0.3,    # 30% dropout between layers (regularization)
        "output_size":      3,      # Away/Draw/Home logits
        "learning_rate":    0.001,  # Adam step size
        "epochs":           100,    # max epochs (early stopping ends it sooner)
        "batch_size":       64,     # samples per gradient update
        "patience":         15,     # early-stopping patience
        # device is auto-detected at runtime (MPS on M1, else CPU) by
        # lstm_model.py — do not hard-code it here.
    },
    "monte_carlo": {
        "n_simulations":    10000,  # simulations per single-match prediction
        "tournament_sims":  1000,   # full-bracket simulations
        "random_state":     42,
    },
    "ensemble": {
        # Final probability = weighted blend of the two models.
        # XGBoost gets more weight: it is stronger on tabular features.
        "xgboost_weight":   0.65,
        "lstm_weight":      0.35,
    },
}

# ── Elo rating system ─────────────────────────────────────────
# Chess-style dynamic strength rating, updated after each match by
# (actual result − expected result) scaled by K and a tournament
# importance multiplier (a World Cup result moves Elo more than a
# friendly does).
ELO_CONFIG: dict = {
    "initial_rating": 1500,   # starting Elo for an unseen team
    "k_factor":       40,     # max rating points exchanged per match
    "tournament_multipliers": {
        "FIFA World Cup":               4.0,
        "UEFA Euro":                    3.0,
        "Copa America":                 3.0,
        "AFC Asian Cup":                3.0,
        "Africa Cup of Nations":        3.0,
        "CONCACAF Gold Cup":            2.5,
        "FIFA World Cup qualification": 2.5,
        "UEFA Nations League":          2.0,
        "Friendly":                     1.0,
    },
}

# ── Feature engineering settings ──────────────────────────────
FEATURE_CONFIG: dict = {
    "recent_form_window":      10,    # matches used for "current form"
    "h2h_lookback_years":      20,    # head-to-head history window
    "form_decay_factor":       0.9,   # most-recent match weighted highest
    "min_matches_for_feature": 3,     # below this, fall back to priors
    "training_start_year":     2000,  # modern era only; older data less relevant
}

# ── Security settings ─────────────────────────────────────────
# Read overridable values from the environment first, fall back to
# safe defaults. os.getenv returns a string, so cast where needed.
SECURITY_CONFIG: dict = {
    "max_file_size_mb":   int(os.getenv("MAX_FILE_SIZE_MB", "50")),  # DoS guard
    "allowed_extensions": [".csv", ".json"],   # download type whitelist
    "ssl_verify":         True,    # ALWAYS verify TLS certs — never set False
    "request_timeout_seconds": 30, # kill stalled requests
    "max_retries":        3,       # retry transient download failures
    "log_all_data_access": True,   # audit every data read/write
    "hash_algorithm":     "sha256",# integrity hash function
}

# ── Logging configuration (the audit trail) ───────────────────
# Logs go to BOTH a persistent file (security trail) and your
# terminal (live feedback). LOG_LEVEL is overridable via .env.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / "predictor.log", encoding="utf-8"),
        logging.StreamHandler(),  # also print to the VS Code terminal
    ],
)

_logger = logging.getLogger("wc2026.config")
_logger.info(
    "Config loaded | WC2026 Predictor | Python 3.13 | M1 Mac | "
    f"teams={len(ALL_WC2026_TEAMS)} | groups={len(WC2026_GROUPS)} | "
    f"log_level={_LOG_LEVEL}"
)
