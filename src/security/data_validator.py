# ==============================================================
# FILE: src/security/data_validator.py
# PURPOSE: Validate and sanitize EVERY input before it reaches a model.
#
# SECURITY PRINCIPLE: "Never trust external data."
#   - Team names must match the official 48-team whitelist (exact).
#   - Adjustment factors are key-whitelisted, type-checked, range-clamped.
#   - DataFrames are schema- and size-checked before any processing.
#   - Rejections are always explicit and logged — never silent.
#
# MENTAL MODEL (for a cybersecurity background): this is your input
# sanitization boundary, the same role prepared statements play against
# SQL injection. Here it blocks type confusion, out-of-range values,
# truncated/corrupt CSVs, and stray strings (e.g. a path like
# "../../etc/passwd") sneaking into a feature dictionary as a "team".
# ==============================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ALL_WC2026_TEAMS   # the authoritative 48-team whitelist

logger = logging.getLogger("wc2026.security.validator")

# ── Validation constants ──────────────────────────────────────

# Only these optional adjustment keys are accepted; anything else is rejected.
VALID_ADJUSTMENT_KEYS: frozenset[str] = frozenset({
    "weather_factor",
    "pitch_condition",
    "lineup_stability",
})

# Columns the historical results CSV MUST contain. Missing any of these
# means the wrong file or a corrupt download — fail fast.
RESULTS_REQUIRED_COLS: list[str] = [
    "date", "home_team", "away_team",
    "home_score", "away_score", "tournament", "neutral",
]

# Upper bound on any single string field — a cheap DoS / abuse guard.
MAX_STRING_LENGTH: int = 200


def validate_team_name(name: object, context: str = "unknown") -> str:
    """
    Validate a team name against the official WC2026 whitelist.

    Blocks: wrong types, pathologically long strings, and ANY value not
    in the exact 48-team roster (typos, injected paths, junk keys).

    Args:
        name:    value to validate (any type — we type-check first)
        context: caller label, included in logs for traceability
    Returns:
        the cleaned, whitespace-stripped, whitelisted team name
    Raises:
        TypeError:  name is not a string
        ValueError: name too long, or not a recognized WC2026 team
    """
    # 1) Type gate — rejects None, int, list, dict, etc. up front.
    if not isinstance(name, str):
        raise TypeError(
            f"Team name must be str, got {type(name).__name__} (context={context})"
        )

    # 2) Normalize — strip whitespace (a very common CSV quality issue).
    name = name.strip()

    # 3) Length gate — refuse absurdly long inputs before the set lookup.
    if len(name) > MAX_STRING_LENGTH:
        raise ValueError(
            f"Team name too long ({len(name)} > {MAX_STRING_LENGTH}) (context={context})"
        )

    # 4) Whitelist gate — ONLY exact members of the official roster pass.
    if name not in ALL_WC2026_TEAMS:
        logger.warning(f"REJECTED team name '{name}' (context={context})")
        raise ValueError(
            f"Unknown team: '{name}'. Must be one of the 48 WC2026 teams.\n"
            f"Spelling tips: 'United States' not 'USA'; 'DR Congo' not 'Congo'; "
            f"'Turkey' not 'Türkiye'; 'South Korea' not 'Korea Republic'."
        )

    return name


def validate_match_input(
    team_a: str,
    team_b: str,
    adjustments: Optional[dict] = None,
) -> dict:
    """
    Validate everything needed for one match prediction request.

    The single gate every prediction passes through: both teams are
    whitelisted, they differ, and any adjustment factors are clean.

    Returns a ready-to-use dict:
        {
          "home_team": str,
          "away_team": str,
          "adjustments": {weather_factor, pitch_condition, lineup_stability}
        }
    Raises TypeError / ValueError on any invalid input.
    """
    home_team = validate_team_name(team_a, context="match_input.team_a")
    away_team = validate_team_name(team_b, context="match_input.team_b")

    # A team cannot play itself.
    if home_team == away_team:
        raise ValueError(f"Home and away teams cannot be identical: '{home_team}'")

    # Safe defaults: 1.0 == "ideal / no adjustment".
    clean_adjustments: dict[str, float] = {
        "weather_factor":   1.0,
        "pitch_condition":  1.0,
        "lineup_stability": 1.0,
    }

    if adjustments is not None:
        if not isinstance(adjustments, dict):
            raise TypeError(f"adjustments must be dict, got {type(adjustments).__name__}")

        for key, value in adjustments.items():
            # Key whitelist — strictly reject anything unexpected.
            if key not in VALID_ADJUSTMENT_KEYS:
                raise ValueError(
                    f"Unknown adjustment key '{key}'. "
                    f"Valid: {sorted(VALID_ADJUSTMENT_KEYS)}"
                )
            # Type gate.
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                # note: bool is a subclass of int in Python — exclude it explicitly
                raise TypeError(f"Adjustment '{key}' must be a real number, got {value!r}")
            value = float(value)
            # Range gate — clamp domain is [0.0, 1.0].
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Adjustment '{key}'={value} out of range [0.0, 1.0]."
                )
            clean_adjustments[key] = value

    logger.info(
        f"Match input validated | {home_team} vs {away_team} | adj={clean_adjustments}"
    )
    return {
        "home_team":   home_team,
        "away_team":   away_team,
        "adjustments": clean_adjustments,
    }


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    min_rows: int = 100,
    name: str = "dataframe",
) -> pd.DataFrame:
    """
    Confirm a DataFrame has the expected schema and a plausible size.

    Catches truncated downloads, the wrong file entirely, and upstream
    schema drift — before any of it can poison feature engineering.

    Args:
        df:               the frame to check
        required_columns: columns that MUST be present (e.g. RESULTS_REQUIRED_COLS)
        min_rows:         smallest acceptable row count (corruption guard)
        name:             label used in log lines
    Returns:
        the same DataFrame, unchanged, once it passes.
    Raises TypeError / ValueError on failure.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for '{name}', got {type(df).__name__}")

    if len(df) < min_rows:
        raise ValueError(
            f"'{name}' has only {len(df):,} rows (min {min_rows:,}). "
            "Likely truncated or the wrong file."
        )

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"'{name}' is missing required columns: {sorted(missing)}. "
            f"Present: {sorted(df.columns.tolist())}"
        )

    # Warn (don't fail) on suspiciously high null rates — soft corruption signal.
    for col in required_columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            logger.warning(f"Column '{col}' in '{name}' is {null_pct:.1%} null — check source.")

    logger.info(f"DataFrame validated | name={name} | rows={len(df):,} | cols={len(df.columns)}")
    return df


def sanitize_float(
    value: object,
    name: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    default: float = 0.5,
) -> float:
    """
    Last-line-of-defense numeric sanitizer.

    Converts any value to a float clamped to [min_val, max_val], turning
    NaN/Inf/garbage into a safe default instead of letting it propagate
    into a feature vector (where a single NaN can poison a whole batch).

    Always returns a usable float; never raises.
    """
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            logger.warning(f"NaN/Inf in '{name}' -> default {default}")
            return default
        clamped = float(np.clip(f, min_val, max_val))
        if clamped != f:
            logger.warning(f"'{name}' clamped {f:.4f} -> {clamped:.4f}")
        return clamped
    except (TypeError, ValueError):
        logger.warning(f"Cannot convert '{name}'={value!r} to float -> default {default}")
        return default
