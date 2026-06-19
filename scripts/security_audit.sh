#!/bin/bash
# ==============================================================
# scripts/security_audit.sh
# PURPOSE: Run the full security scan before any GitHub push.
#
# RUNS:
#   1. bandit    — Python AST security linter (finds insecure code patterns)
#   2. pip-audit — checks installed packages against known CVE databases
#   3. Git safety — confirms .env is never staged and is in .gitignore
#   4. Secret scan — greps source for hardcoded password/key/token literals
#
# USAGE:
#   chmod +x scripts/security_audit.sh   # first time only
#   ./scripts/security_audit.sh
#
# Exit codes: 0 = all clear | 1 = issues found
#
# NOTE: we deliberately do NOT use `set -e`. An audit must run EVERY check
# even when an earlier one fails, then report a single verdict at the end.
# (Also, `set -e` + `((PASS++))` is a known bash trap: the post-increment
# returns the old value 0, which bash reads as failure and aborts the script.
# We use PASS=$((PASS+1)) instead, which never affects the exit status.)
# ==============================================================

GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[1;33m"; NC="\033[0m"
PASS=0
FAIL=0

pass() { echo -e "  ${GREEN}OK${NC}  $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}!!${NC}  $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  ${YELLOW}~~${NC}  $1"; }

echo ""
echo "======================================================"
echo "  WC2026 Predictor — Security Audit"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# ── 1. bandit ─────────────────────────────────────────────────
echo ""
echo "  [1/4] bandit — static security analysis"
if command -v bandit &> /dev/null; then
    # -ll = report MEDIUM and HIGH only; -q = quiet; -r = recurse src/
    if bandit -r src/ -ll -q 2>/dev/null; then
        pass "bandit: no MEDIUM/HIGH severity issues"
    else
        fail "bandit: issues detected (review output above)"
    fi
else
    warn "bandit not installed — pip install bandit"
fi

# ── 2. pip-audit ──────────────────────────────────────────────
echo ""
echo "  [2/4] pip-audit — dependency CVE scan"
if command -v pip-audit &> /dev/null; then
    # ACCEPTED VULN: GHSA-rrmf-rvhw-rf47 (CVE-2025-3000) affects torch.jit.script,
    # which this project never calls; the attack vector is local-only; and PyTorch
    # has published no fixed version (nothing to upgrade to). Reviewed and accepted
    # here — NOT silenced. Re-check when a torch fix ships.
    #
    # We don't rely on pip-audit's --ignore-vuln (its ID matching is inconsistent
    # across versions). Instead we run the scan, and PASS only if the single
    # advisory left is exactly the one we've accepted — anything else fails.
    ACCEPTED="GHSA-rrmf-rvhw-rf47"
    PA_OUT="$(pip-audit 2>/dev/null)"; PA_RC=$?
    if [ $PA_RC -eq 0 ]; then
        pass "pip-audit: no known CVEs in installed packages"
    else
        # Extract every advisory ID found, then drop the accepted one.
        REMAINING="$(printf '%s\n' "$PA_OUT" | grep -oE '(GHSA|CVE|PYSEC)-[A-Za-z0-9-]+' | grep -v "$ACCEPTED" | sort -u)"
        if [ -z "$REMAINING" ]; then
            pass "pip-audit: only the reviewed+accepted $ACCEPTED remains (unused torch.jit.script; no fix released)"
        else
            fail "pip-audit: unexpected vulnerable packages found (try: pip-audit --fix)"
            printf '%s\n' "$REMAINING" | sed 's/^/        /'
        fi
    fi
else
    warn "pip-audit not installed — pip install pip-audit"
fi

# ── 3. Git safety: .env must never be committed ───────────────
echo ""
echo "  [3/4] git safety — secret-file checks"
if [ -f ".env" ]; then
    if git diff --cached --name-only 2>/dev/null | grep -q "^\.env$"; then
        fail ".env IS STAGED! run: git restore --staged .env"
    else
        pass ".env exists and is NOT staged"
    fi
else
    pass ".env not present (create from .env.example when needed)"
fi
if grep -qE "^\.env([[:space:]]|#|$)" .gitignore 2>/dev/null; then
    pass ".gitignore contains .env"
else
    fail ".gitignore is missing a .env entry — add it now"
fi

# ── 4. Hardcoded-secret pattern scan ──────────────────────────
echo ""
echo "  [4/4] hardcoded-secret pattern scan"
PATTERNS=(
    "password[[:space:]]*=[[:space:]]*['\"][^'\"]{4,}"
    "api_key[[:space:]]*=[[:space:]]*['\"][^'\"]{8,}"
    "secret[[:space:]]*=[[:space:]]*['\"][^'\"]{8,}"
    "token[[:space:]]*=[[:space:]]*['\"][^'\"]{8,}"
)
SECRET_FOUND=0
for pattern in "${PATTERNS[@]}"; do
    # -E extended regex; exclude comment lines so explanatory text doesn't trip it
    if grep -rniE --include="*.py" "$pattern" src/ main.py 2>/dev/null | grep -v "#" | grep -q .; then
        fail "possible hardcoded secret matching: $pattern"
        grep -rniE --include="*.py" "$pattern" src/ main.py 2>/dev/null | grep -v "#"
        SECRET_FOUND=1
    fi
done
[ $SECRET_FOUND -eq 0 ] && pass "no hardcoded-secret patterns in Python source"

# ── Verdict ───────────────────────────────────────────────────
echo ""
echo "======================================================"
if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}ALL CHECKS PASSED${NC} | $PASS passed | $FAIL failed"
    echo "  Safe to commit and push."
    exit 0
else
    echo -e "  ${RED}AUDIT FAILED${NC} | $PASS passed | $FAIL failed"
    echo "  Fix all failures before pushing."
    exit 1
fi