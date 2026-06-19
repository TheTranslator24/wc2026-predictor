# ==============================================================
# FILE: tests/test_security.py
# PURPOSE: Prove every security control works. Run before trusting any output.
#
#   pytest tests/test_security.py -v        # verbose
#   pytest tests/test_security.py -q        # quiet
#
# These are the gate: if any test here fails, the security layer has a
# regression and predictions should not be trusted until it's fixed.
# ==============================================================

import hashlib
import json

import pandas as pd
import pytest

from src.security.integrity_check import (
    compute_sha256, register_file, verify_file, load_checksum_registry,
)
from src.security.data_validator import (
    validate_team_name, validate_match_input, validate_dataframe, sanitize_float,
)


# ==============================================================
# SHA-256 INTEGRITY
# ==============================================================
class TestSHA256Integrity:

    def test_compute_sha256_known_value(self, tmp_path):
        """Our hash must equal Python's reference hashlib value (correctness)."""
        f = tmp_path / "test.txt"; f.write_bytes(b"WC2026")
        assert compute_sha256(f) == hashlib.sha256(b"WC2026").hexdigest()

    def test_compute_sha256_different_content(self, tmp_path):
        """Different bytes -> different hash (collision sanity)."""
        a = tmp_path / "a.txt"; a.write_bytes(b"Spain")
        b = tmp_path / "b.txt"; b.write_bytes(b"Germany")
        assert compute_sha256(a) != compute_sha256(b)

    def test_compute_sha256_file_not_found(self, tmp_path):
        """Hashing a missing file must raise, not return garbage."""
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nope.csv")

    def test_register_and_verify_clean(self, tmp_path):
        """A registered, untouched file must verify True."""
        f = tmp_path / "data.csv"; f.write_bytes(b"date,home_team\n2026-06-11,Spain")
        reg = tmp_path / "checksums.json"
        register_file(f, reg)
        assert verify_file(f, reg) is True

    def test_verify_detects_tampering(self, tmp_path):
        """The core guarantee: a changed file must verify False."""
        f = tmp_path / "data.csv"; f.write_bytes(b"original")
        reg = tmp_path / "checksums.json"
        register_file(f, reg)
        f.write_bytes(b"MODIFIED malicious content")   # tamper after registration
        assert verify_file(f, reg) is False

    def test_registry_is_valid_json(self, tmp_path):
        """The checksum registry must be well-formed JSON keyed by filename."""
        f = tmp_path / "data.csv"; f.write_bytes(b"test data")
        reg = tmp_path / "checksums.json"
        register_file(f, reg)
        with open(reg) as fh:
            data = json.load(fh)
        assert isinstance(data, dict) and "data.csv" in data

    def test_empty_registry_returns_dict(self, tmp_path):
        """First run (no registry yet) is a normal empty-dict state, not an error."""
        assert load_checksum_registry(tmp_path / "checksums.json") == {}


# ==============================================================
# TEAM NAME WHITELIST
# ==============================================================
class TestTeamNameValidation:

    def test_valid_teams_pass(self):
        """Exact WC2026 names (including tricky ones) must be accepted."""
        for name in ["Spain", "United States", "DR Congo",
                     "Bosnia and Herzegovina", "Cape Verde", "Czechia", "Curacao"]:
            assert validate_team_name(name, "test") == name

    def test_strips_whitespace(self):
        assert validate_team_name("  Spain  ", "test") == "Spain"

    def test_invalid_team_raises(self):
        """Typos, wrong names, injections, and over-length must all be rejected."""
        bad = ["USA", "Türkiye", "Congo", "Italy",
               "../etc/passwd", "'; DROP TABLE;--", "", "x" * 201]
        for name in bad:
            with pytest.raises((ValueError, TypeError)):
                validate_team_name(name, "test")

    def test_non_string_raises_type_error(self):
        for bad in [None, 42, 3.14, ["Spain"], {"team": "Spain"}]:
            with pytest.raises(TypeError):
                validate_team_name(bad, "test")


# ==============================================================
# MATCH INPUT VALIDATION
# ==============================================================
class TestMatchInputValidation:

    def test_valid_match_passes(self):
        r = validate_match_input("Spain", "Germany")
        assert r["home_team"] == "Spain" and r["away_team"] == "Germany"
        assert r["adjustments"]["weather_factor"] == 1.0

    def test_valid_adjustments_pass(self):
        r = validate_match_input("France", "Brazil",
                                 adjustments={"weather_factor": 0.7,
                                              "pitch_condition": 0.85,
                                              "lineup_stability": 0.9})
        assert r["adjustments"]["weather_factor"] == 0.7

    def test_same_team_raises(self):
        # validator message says "...cannot be identical..." -> match "identical"
        with pytest.raises(ValueError, match="identical"):
            validate_match_input("Spain", "Spain")

    def test_unknown_adjustment_key_raises(self):
        with pytest.raises(ValueError, match="Unknown adjustment key"):
            validate_match_input("Spain", "Germany", adjustments={"malicious_key": 0.5})

    def test_out_of_range_adjustment_raises(self):
        for bad in [-0.1, 1.01, 2.0, -1.0, 100.0]:
            with pytest.raises(ValueError):
                validate_match_input("Spain", "Germany", adjustments={"weather_factor": bad})

    def test_non_numeric_adjustment_raises(self):
        with pytest.raises(TypeError):
            validate_match_input("Spain", "Germany", adjustments={"weather_factor": "heavy rain"})

    def test_bool_adjustment_rejected(self):
        # bool is a subclass of int in Python; the validator must still reject it
        with pytest.raises(TypeError):
            validate_match_input("Spain", "Germany", adjustments={"weather_factor": True})


# ==============================================================
# DATAFRAME SCHEMA VALIDATION
# ==============================================================
class TestDataFrameValidation:

    REQUIRED = ["date", "home_team", "away_team", "home_score",
                "away_score", "tournament", "neutral"]

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=200),
            "home_team": ["Spain"] * 200, "away_team": ["Germany"] * 200,
            "home_score": [1.0] * 200, "away_score": [0.0] * 200,
            "tournament": ["FIFA World Cup"] * 200, "neutral": [True] * 200,
        })
        assert len(validate_dataframe(df, required_columns=self.REQUIRED)) == 200

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"date": [1, 2, 3], "home_team": ["Spain"] * 3})
        with pytest.raises(ValueError, match="rows"):
            validate_dataframe(df, required_columns=["date", "home_team"], min_rows=100)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"date": range(200), "home_team": ["Spain"] * 200})
        with pytest.raises(ValueError, match="missing"):
            validate_dataframe(df, required_columns=["date", "home_team", "MISSING_COL"])


# ==============================================================
# FLOAT SANITIZER
# ==============================================================
class TestSanitizeFloat:

    def test_valid_values_unchanged(self):
        for v in [0.0, 0.5, 1.0, 0.75]:
            assert sanitize_float(v, "t") == pytest.approx(v)

    def test_clamps_below_zero(self):
        assert sanitize_float(-0.1, "t") == pytest.approx(0.0)

    def test_clamps_above_one(self):
        assert sanitize_float(1.5, "t") == pytest.approx(1.0)

    def test_nan_returns_default(self):
        assert sanitize_float(float("nan"), "t", default=0.5) == pytest.approx(0.5)

    def test_inf_returns_default(self):
        assert sanitize_float(float("inf"), "t", default=0.5) == pytest.approx(0.5)

    def test_non_numeric_returns_default(self):
        assert sanitize_float("not a number", "t", default=0.5) == pytest.approx(0.5)


# ==============================================================
# CONFIG INVARIANTS (the data the whole system depends on)
# ==============================================================
class TestSecurityInvariants:

    def test_all_48_teams_in_whitelist(self):
        from src.config import ALL_WC2026_TEAMS, WC2026_GROUPS
        group_teams = {t for teams in WC2026_GROUPS.values() for t in teams}
        assert group_teams == ALL_WC2026_TEAMS
        assert len(ALL_WC2026_TEAMS) == 48

    def test_groups_have_exactly_4_teams(self):
        from src.config import WC2026_GROUPS
        for g, teams in WC2026_GROUPS.items():
            assert len(teams) == 4, f"Group {g} has {len(teams)}"

    def test_twelve_groups(self):
        from src.config import WC2026_GROUPS
        assert len(WC2026_GROUPS) == 12
        assert set(WC2026_GROUPS) == set("ABCDEFGHIJKL")

    def test_team_to_group_complete(self):
        from src.config import TEAM_TO_GROUP, ALL_WC2026_TEAMS
        assert set(TEAM_TO_GROUP) == ALL_WC2026_TEAMS

    def test_fifa_rankings_cover_all_teams(self):
        from src.config import FIFA_RANKINGS, ALL_WC2026_TEAMS
        assert set(FIFA_RANKINGS) == ALL_WC2026_TEAMS
        for team, rank in FIFA_RANKINGS.items():
            assert 1 <= rank <= 48, f"{team}: {rank} out of [1,48]"


if __name__ == "__main__":
    import sys, subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))
