#!/usr/bin/env python3
# ==============================================================
# FILE: scripts/verify_setup.py
# PURPOSE: One-command audit of everything built in Batches 1-4.
#
# It checks four things and prints a clear PASS/FAIL for each:
#   1. INTERPRETER  — are you on the right venv (3.11, project venv)?
#   2. FILES        — does every expected source file exist?
#   3. PACKAGES     — is every third-party dependency importable?
#   4. MODULES      — does every project module import cleanly?
#   5. SMOKE TEST   — do the core pieces actually function?
#
# RUN FROM THE PROJECT ROOT, venv active:
#   python3 scripts/verify_setup.py
#
# Nothing here changes your project — it only inspects and reports.
# ==============================================================

import sys
import importlib
from pathlib import Path

# Resolve project root = the folder ABOVE scripts/, and make it importable
# so "src.*" resolves no matter where you launch from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN, RED, YEL, DIM, RST = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
OK   = f"{GREEN}✓{RST}"
BAD  = f"{RED}✗{RST}"
WARN = f"{YEL}!{RST}"

# Counters used for the final verdict.
fails = 0
warns = 0


def header(text: str) -> None:
    print(f"\n{text}\n" + "─" * len(text))


# ── 1. INTERPRETER ────────────────────────────────────────────
header("1. INTERPRETER")
py = sys.version_info
print(f"  {OK} Python {py.major}.{py.minor}.{py.micro}")
in_venv = (Path(sys.prefix) != Path(sys.base_prefix))
if in_venv and "wc2026-predictor" in sys.prefix:
    print(f"  {OK} Using project venv: {DIM}{sys.prefix}{RST}")
elif in_venv:
    print(f"  {WARN} In a venv, but not obviously the project one: {sys.prefix}"); warns += 1
else:
    print(f"  {BAD} NOT in a virtual environment — run 'source venv/bin/activate'"); fails += 1

# ── 2. FILES ──────────────────────────────────────────────────
header("2. FILES (Batches 1-4)")
EXPECTED_FILES = [
    "requirements.txt", ".gitignore", ".env.example",
    "src/__init__.py", "src/config.py",
    "src/security/__init__.py",
    "src/security/integrity_check.py", "src/security/data_validator.py",
    "src/data/__init__.py", "src/data/collector.py", "src/data/features.py",
    "src/models/__init__.py", "src/models/xgboost_model.py", "src/models/lstm_model.py",
    # package folders that exist now but are filled in Batch 5:
    "src/predictor/__init__.py", "src/visualization/__init__.py",
]
for rel in EXPECTED_FILES:
    if (ROOT / rel).exists():
        print(f"  {OK} {rel}")
    else:
        print(f"  {BAD} {rel}  {DIM}<-- MISSING{RST}"); fails += 1

# ── 3. PACKAGES (pip dependencies) ────────────────────────────
header("3. PACKAGES")
# import name : pip name (where they differ)
PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "xgboost": "xgboost", "shap": "shap",
    "torch": "torch", "requests": "requests", "certifi": "certifi",
    "dotenv": "python-dotenv", "psutil": "psutil", "matplotlib": "matplotlib",
}
for mod, pip_name in PACKAGES.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {OK} {pip_name:<14} {DIM}{ver}{RST}")
    except Exception as e:
        print(f"  {BAD} {pip_name:<14} {DIM}pip install {pip_name}{RST}  ({type(e).__name__})"); fails += 1

# ── 4. PROJECT MODULES ────────────────────────────────────────
header("4. PROJECT MODULES")
MODULES = [
    "src.config",
    "src.security.integrity_check", "src.security.data_validator",
    "src.data.collector", "src.data.features",
    "src.models.xgboost_model", "src.models.lstm_model",
]
for mod in MODULES:
    try:
        importlib.import_module(mod)
        print(f"  {OK} {mod}")
    except Exception as e:
        print(f"  {BAD} {mod}  {DIM}{type(e).__name__}: {e}{RST}"); fails += 1

# ── 5. SMOKE TEST (do the core pieces work?) ──────────────────
header("5. SMOKE TEST")
try:
    from src.config import ALL_WC2026_TEAMS
    assert len(ALL_WC2026_TEAMS) == 48
    print(f"  {OK} config: 48 teams loaded")
except Exception as e:
    print(f"  {BAD} config smoke test: {e}"); fails += 1

try:
    from src.security.data_validator import validate_match_input, validate_team_name
    validate_match_input("Mexico", "South Africa")          # should pass
    try:
        validate_team_name("USA"); raise AssertionError("'USA' should have been rejected")
    except ValueError:
        pass
    print(f"  {OK} validator: accepts real teams, rejects 'USA'")
except Exception as e:
    print(f"  {BAD} validator smoke test: {e}"); fails += 1

try:
    import tempfile
    from src.security.integrity_check import register_file, verify_file
    d = Path(tempfile.mkdtemp()); f = d / "t.csv"; f.write_text("a,b\n1,2\n"); reg = d / "r.json"
    register_file(f, reg)
    ok_clean = verify_file(f, reg)
    f.write_text("tampered")
    bad_tamper = verify_file(f, reg)
    assert ok_clean and not bad_tamper
    print(f"  {OK} integrity: verifies clean file, catches tampering")
except Exception as e:
    print(f"  {BAD} integrity smoke test: {e}"); fails += 1

# ── VERDICT ───────────────────────────────────────────────────
header("VERDICT")
if fails == 0 and warns == 0:
    print(f"  {GREEN}ALL CHECKS PASSED — Batches 1-4 are complete and healthy.{RST}")
    print(f"  {DIM}Ready for Batch 5 (ensemble, predictor, charts, main).{RST}")
    sys.exit(0)
elif fails == 0:
    print(f"  {YEL}PASSED with {warns} warning(s) — safe to continue.{RST}")
    sys.exit(0)
else:
    print(f"  {RED}{fails} problem(s) found above — fix those before Batch 5.{RST}")
    sys.exit(1)
