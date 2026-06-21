# ==============================================================
# FILE: src/visualization/team_identity.py
# PURPOSE: Visual identity for all 48 teams — colors + ISO codes + flags.
#
# WHY THIS EXISTS:
#   - Emoji flags render as empty boxes ("tofu") in matplotlib's default font.
#   - Team CRESTS/LOGOS are trademarked IP and must NOT be reproduced.
#   - National FLAG DESIGNS are public domain and safe to use.
#   - Team COLORS are not protected and are instantly recognizable.
#   So we identify teams by curated color palettes + ISO codes, with an
#   OPTIONAL real-flag-image overlay you can enable by downloading PNGs.
#
# OUTPUT: a TEAMS dict {team: {"primary","secondary","iso"}} plus helpers.
# ==============================================================

from pathlib import Path

from src.config import BASE_DIR

# Where downloaded flag PNGs are cached (see flag_loader / download step).
FLAG_DIR = BASE_DIR / "assets" / "flags"

# ── Curated identity for every WC2026 team ────────────────────
# primary/secondary = recognizable national kit/flag colors (hex).
# iso = ISO 3166-1 alpha-2 (or GB subdivision) used for flag file names.
TEAMS: dict[str, dict] = {
    # Group A
    "Mexico":        {"primary": "#006847", "secondary": "#CE1126", "iso": "mx"},
    "South Korea":   {"primary": "#0047A0", "secondary": "#CD2E3A", "iso": "kr"},
    "South Africa":  {"primary": "#007A4D", "secondary": "#FFB915", "iso": "za"},
    "Czechia":       {"primary": "#11457E", "secondary": "#D7141A", "iso": "cz"},
    # Group B
    "Canada":        {"primary": "#FF0000", "secondary": "#FFFFFF", "iso": "ca"},
    "Switzerland":   {"primary": "#D52B1E", "secondary": "#FFFFFF", "iso": "ch"},
    "Qatar":         {"primary": "#8A1538", "secondary": "#FFFFFF", "iso": "qa"},
    "Bosnia and Herzegovina": {"primary": "#002395", "secondary": "#FFD700", "iso": "ba"},
    # Group C
    "Brazil":        {"primary": "#FFDF00", "secondary": "#009C3B", "iso": "br"},
    "Morocco":       {"primary": "#C1272D", "secondary": "#006233", "iso": "ma"},
    "Scotland":      {"primary": "#0065BF", "secondary": "#FFFFFF", "iso": "gb-sct"},
    "Haiti":         {"primary": "#00209F", "secondary": "#D21034", "iso": "ht"},
    # Group D
    "United States": {"primary": "#0A3161", "secondary": "#B31942", "iso": "us"},
    "Australia":     {"primary": "#00843D", "secondary": "#FFCD00", "iso": "au"},
    "Paraguay":      {"primary": "#D52B1E", "secondary": "#0038A8", "iso": "py"},
    "Turkey":        {"primary": "#E30A17", "secondary": "#FFFFFF", "iso": "tr"},
    # Group E
    "Germany":       {"primary": "#000000", "secondary": "#DD0000", "iso": "de"},
    "Ecuador":       {"primary": "#FFD100", "secondary": "#0072CE", "iso": "ec"},
    "Ivory Coast":   {"primary": "#FF8200", "secondary": "#009A44", "iso": "ci"},
    "Curacao":       {"primary": "#002B7F", "secondary": "#F9D616", "iso": "cw"},
    # Group F
    "Netherlands":   {"primary": "#FF7900", "secondary": "#21468B", "iso": "nl"},
    "Japan":         {"primary": "#0A1E5E", "secondary": "#BC002D", "iso": "jp"},
    "Tunisia":       {"primary": "#E70013", "secondary": "#FFFFFF", "iso": "tn"},
    "Sweden":        {"primary": "#006AA7", "secondary": "#FECC02", "iso": "se"},
    # Group G
    "Belgium":       {"primary": "#E30613", "secondary": "#FFD90C", "iso": "be"},
    "Iran":          {"primary": "#239F40", "secondary": "#DA0000", "iso": "ir"},
    "Egypt":         {"primary": "#CE1126", "secondary": "#000000", "iso": "eg"},
    "New Zealand":   {"primary": "#000000", "secondary": "#FFFFFF", "iso": "nz"},
    # Group H
    "Spain":         {"primary": "#C60B1E", "secondary": "#FFC400", "iso": "es"},
    "Uruguay":       {"primary": "#0038A8", "secondary": "#FCD116", "iso": "uy"},
    "Saudi Arabia":  {"primary": "#006C35", "secondary": "#FFFFFF", "iso": "sa"},
    "Cape Verde":    {"primary": "#003893", "secondary": "#CF2027", "iso": "cv"},
    # Group I
    "France":        {"primary": "#0055A4", "secondary": "#EF4135", "iso": "fr"},
    "Senegal":       {"primary": "#00853F", "secondary": "#FDEF42", "iso": "sn"},
    "Norway":        {"primary": "#BA0C2F", "secondary": "#00205B", "iso": "no"},
    "Iraq":          {"primary": "#007A3D", "secondary": "#CE1126", "iso": "iq"},
    # Group J
    "Argentina":     {"primary": "#75AADB", "secondary": "#FFFFFF", "iso": "ar"},
    "Austria":       {"primary": "#ED2939", "secondary": "#FFFFFF", "iso": "at"},
    "Algeria":       {"primary": "#006233", "secondary": "#D21034", "iso": "dz"},
    "Jordan":        {"primary": "#007A3D", "secondary": "#CE1126", "iso": "jo"},
    # Group K
    "Portugal":      {"primary": "#006600", "secondary": "#FF0000", "iso": "pt"},
    "Colombia":      {"primary": "#FCD116", "secondary": "#003893", "iso": "co"},
    "Uzbekistan":    {"primary": "#1EB53A", "secondary": "#0099B5", "iso": "uz"},
    "DR Congo":      {"primary": "#007FFF", "secondary": "#F7D618", "iso": "cd"},
    # Group L
    "England":       {"primary": "#FFFFFF", "secondary": "#CF081F", "iso": "gb-eng"},
    "Croatia":       {"primary": "#FF0000", "secondary": "#171796", "iso": "hr"},
    "Panama":        {"primary": "#005293", "secondary": "#D21034", "iso": "pa"},
    "Ghana":         {"primary": "#006B3F", "secondary": "#FCD116", "iso": "gh"},
}

# Neutral fallback for any unexpected name (keeps charts from crashing).
_FALLBACK = {"primary": "#888888", "secondary": "#cccccc", "iso": "un"}


def color(team: str, which: str = "primary") -> str:
    """Hex color for a team. which='primary' or 'secondary'."""
    return TEAMS.get(team, _FALLBACK).get(which, _FALLBACK[which])


def iso(team: str) -> str:
    """ISO 3166-1 alpha-2 (or GB subdivision) code for a team."""
    return TEAMS.get(team, _FALLBACK)["iso"]


def label(team: str) -> str:
    """
    Clean text label for charts: 'ESP · Spain'. Uses the ISO code uppercased
    (the GB subdivisions collapse to ENG/SCT) — readable everywhere, no emoji,
    no tofu boxes.
    """
    code = iso(team).split("-")[-1].upper()   # 'gb-eng' -> 'ENG', 'es' -> 'ES'
    return f"{code} · {team}"


def flag_path(team: str) -> Path:
    """Where this team's flag PNG would live (may or may not exist)."""
    return FLAG_DIR / f"{iso(team)}.png"


def has_flag(team: str) -> bool:
    """True if a downloaded flag image is available for this team."""
    return flag_path(team).exists()
