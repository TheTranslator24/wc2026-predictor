# WC2026 Match Predictor

A machine-learning system that predicts **win / draw / loss** outcomes for FIFA World Cup 2026 matches, built with a security-first engineering approach. It combines a gradient-boosted-tree model and a recurrent neural network into a weighted ensemble, then quantifies uncertainty with Monte Carlo simulation.

> ⚠️ **Disclaimer:** This is an educational machine-learning project. Its forecasts are
> simulated model outputs — **not betting advice** and not a prediction of any specific
> outcome. See [DISCLAIMER.md](DISCLAIMER.md). Football is uncertain.

> **Disclaimer.** Football is inherently unpredictable. The best public match-outcome models reach roughly **55–60% accuracy**, and draws are the hardest outcome to call. This project is a portfolio and learning exercise, not betting advice. Treat every number as a probability with wide error bars, not a guarantee.

---

## What it does

- Predicts the outcome of any single WC2026 match with calibrated probabilities and 95% confidence intervals.
- Projects full group standings (who advances) and runs a full-tournament championship simulation.
- Updates team strength ratings (Elo) as real results come in during the tournament.
- Explains its reasoning: SHAP attributions show which features drove each prediction.

## How it works

The pipeline runs in stages, each leak-free (every feature uses only information available *before* a match):

1. **Secure data ingestion** — downloads the public international-results dataset over verified HTTPS, checks its size, and confirms a SHA-256 hash against a trusted baseline before use.
2. **Elo ratings** — a chronological pass computes a dynamic strength rating for every team, scaled by match importance (a World Cup result moves Elo more than a friendly).
3. **Feature engineering** — 25 features per match: Elo, FIFA ranking, recent form (exponentially weighted), goal averages, head-to-head history, and confederation strength.
4. **Two models, two perspectives**
   - **XGBoost** reads the static snapshot. Balanced class weights ensure draws are predicted at a realistic rate rather than ignored.
   - **LSTM** reads each team's last 10 matches as an ordered sequence, capturing momentum that averages miss.
5. **Ensemble + Monte Carlo** — a weighted blend (XGBoost 65% / LSTM 35%) produces the final distribution; 10,000 simulations wrap it in Wilson confidence intervals.

## Security features

Security is a first-class concern, not an afterthought:

- **Integrity verification** — every data file is SHA-256 hashed and checked against a committed registry; tampered or truncated files are rejected, not silently used.
- **Input whitelisting** — team names must match the exact 48-team roster; adjustment factors are key-whitelisted, type-checked, and range-clamped. This blocks type confusion and injection-style inputs at the boundary.
- **Hardened downloads** — TLS certificate verification (never disabled), request timeouts, streaming with a size cap, and full audit logging.
- **No secrets in code** — configuration is secret-free and safe to commit; any credentials live in a git-ignored `.env`.
- **Tested + scanned** — a pytest security suite plus a `bandit` + `pip-audit` audit script gate every push.

## Quick start

Requires **Python 3.11+** and works comfortably on an Apple-Silicon Mac with 16 GB RAM.

```bash
# 1. Clone and enter
git clone git@github.com:<your-username>/wc2026-predictor.git
cd wc2026-predictor

# 2. Virtual environment + dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio      # Apple-Silicon build

# 3. Configure (no secrets required to run)
cp .env.example .env

# 4. Verify the install
python3 scripts/verify_setup.py

# 5. Run — first time trains + saves both models (~6-12 min), then predicts
python3 main.py
```

## Usage

```bash
python3 main.py                 # predict the fixtures listed in main.py
python3 main.py --group H       # one group: 6 matches + projected table
python3 main.py --all-groups    # all 12 groups
python3 main.py --tournament    # championship probabilities for all 48 teams
python3 main.py --retrain       # force both models to re-train
python3 main.py --no-charts     # text only (faster)
```

In code:

```python
from src.predictor.match_predictor import WC2026Predictor

p = WC2026Predictor().setup()
result = p.predict("Spain", "Germany")
print(result["predicted_outcome"], result["confidence"])

# Feed a real result back in to keep ratings current:
p.update_elo("Spain", "Germany", 2, 1)
```

## Project structure

```
wc2026-predictor/
├── main.py                       # entry point
├── requirements.txt
├── src/
│   ├── config.py                 # single source of truth (teams, params, sources)
│   ├── security/
│   │   ├── integrity_check.py    # SHA-256 verification
│   │   └── data_validator.py     # input whitelisting + sanitization
│   ├── data/
│   │   ├── collector.py          # hardened secure downloader
│   │   └── features.py           # Elo + feature engineering
│   ├── models/
│   │   ├── xgboost_model.py      # gradient-boosted trees
│   │   ├── lstm_model.py         # recurrent sequence model
│   │   └── ensemble.py           # blend + Monte Carlo + tournament sim
│   ├── predictor/
│   │   └── match_predictor.py    # the high-level API
│   └── visualization/
│       └── probability_charts.py # dark-themed PNG charts
├── tests/
│   └── test_security.py          # security regression suite
└── scripts/
    ├── verify_setup.py           # install/health audit
    ├── security_audit.sh         # bandit + pip-audit + secret scan
    └── make_release.sh           # SHA256SUMS for release artifacts
```

## Testing

```bash
pytest tests/test_security.py -v     # all security controls must pass
./scripts/security_audit.sh          # static + dependency security scan
```

## Tech stack

Python · XGBoost · PyTorch · scikit-learn · SHAP · pandas / NumPy · matplotlib · pytest · bandit · pip-audit

## Data & license

- **Data:** international football results, 1872–2026, from the public `martj42/international_results` dataset (**CC0 Public Domain**).
- **Code:** released under the MIT License (see `LICENSE`).

## Limitations

- Match-outcome accuracy is bounded by the sport itself (~55–60% is strong); this model is in that range, not above it.
- Draw recall realistically tops out around 0.20–0.35 — draws are genuinely hard to call.
- The full-tournament simulation uses simplified Elo-only odds for speed and a simplified bracket (it does not model the best-third-place qualification rule exactly).
- FIFA rankings in `config.py` are a snapshot; update them before a run for best results.
