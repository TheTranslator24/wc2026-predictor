#!/bin/bash
# ==============================================================
# scripts/make_release.sh
# PURPOSE: Produce the integrity manifest for a GitHub Release.
#
# Trained models are too large (and binary) to commit to git — they're in
# .gitignore. Instead you attach them to a GitHub *Release* alongside a
# SHA256SUMS file so anyone downloading them can verify they are exactly the
# artifacts you produced (integrity + provenance). This script generates that
# manifest. The same SHA-256 mechanism your integrity_check.py uses for data.
#
# USAGE (after training, from project root):
#   chmod +x scripts/make_release.sh   # first time only
#   ./scripts/make_release.sh
#
# Then: create a GitHub Release for your signed tag and upload the files in
# release_artifacts/ (the two models + SHA256SUMS.txt).
# ==============================================================

set -u   # error on unset variables (safe here; no counters involved)

OUT="release_artifacts"
mkdir -p "$OUT"

echo "Collecting trained model artifacts..."
COPIED=0
for f in models/xgboost_wc2026.pkl models/lstm_wc2026.pt; do
    if [ -f "$f" ]; then
        cp "$f" "$OUT/"
        echo "  + $f"
        COPIED=$((COPIED+1))
    else
        echo "  - $f not found (train first, or LSTM ran XGBoost-only)"
    fi
done

if [ "$COPIED" -eq 0 ]; then
    echo "No model artifacts found. Run 'python3 main.py' to train first."
    exit 1
fi

# Generate the checksum manifest. macOS uses 'shasum -a 256'; Linux 'sha256sum'.
echo "Generating SHA256SUMS.txt..."
cd "$OUT"
if command -v sha256sum &> /dev/null; then
    sha256sum ./* > SHA256SUMS.txt 2>/dev/null
else
    shasum -a 256 ./* > SHA256SUMS.txt 2>/dev/null
fi
# Don't hash the manifest itself.
grep -v "SHA256SUMS.txt" SHA256SUMS.txt > .tmp && mv .tmp SHA256SUMS.txt
cd ..

echo ""
echo "Release artifacts ready in $OUT/ :"
ls -lh "$OUT"
echo ""
echo "Verify later with:   cd $OUT && shasum -a 256 -c SHA256SUMS.txt"
echo "Next: attach these files to your GitHub Release for the signed tag."
