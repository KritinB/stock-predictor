# Stock Direction Predictor — Project Reference

## Location
`/Users/kritinbysani/Documents/Programing/Claude/stock-predictor/`

## What it does
Predicts whether each stock will close **UP or DOWN tomorrow** using XGBoost + technical indicators + market context. Runs as a terminal UI with colour-coded output.

## Files
| File | Purpose |
|---|---|
| `predictor.py` | Main script — run this daily with `python predictor.py` |
| `predictor.ipynb` | Jupyter notebook version with charts |
| `requirements.txt` | Dependencies |
| `history.json` | Persistent log of every run, prediction, and verification |
| `models/` | Saved XGBoost models per ticker (joblib) |

## Stocks tracked
**US** (market context: S&P 500 + VIX)
- AAPL — Apple
- TSLA — Tesla
- NVDA — Nvidia
- ^GSPC — S&P 500

**India** (market context: Nifty 50 + VIX)
- RELIANCE.NS — Reliance Industries
- TCS.NS — Tata Consultancy Services
- HDFCBANK.NS — HDFC Bank
- ^NSEI — Nifty 50

## Model
- **Algorithm:** XGBoost classifier
- **Validation:** 5-fold walk-forward cross-validation (no lookahead bias)
- **Horizon:** 1 trading day (HORIZON = 1 in predictor.py)
- **Training data:** 5 years of daily OHLCV (PERIOD = "5y")

## Features engineered
- Moving averages: SMA 5/20/50, EMA 12/26
- MACD, RSI (14), Bollinger Bands (position + width)
- Daily range, momentum (10-day)
- Volume ratio, volume change (1d, 5d)
- Lag returns: 1/2/3/5/10 day
- Open gap
- Market context: index return (1d, 5d), index vs SMA20
- VIX level + 1-day VIX change

## Metrics displayed
- **Model Accuracy** — historical hit rate from walk-forward CV
- **Confidence** — XGBoost probability output for today's prediction
- **Overall Prediction Accuracy** — `accuracy × 0.7 + confidence × 0.3`
  - ★ High conviction: ≥68%
  - ◆ Moderate conviction: 58–68%
  - ◇ Low conviction: <58%

## Memory system
- Every run logs date, predictions, prices, confidence to `history.json`
- Next day: auto-verifies yesterday's predictions against actual prices
- Builds a **live scorecard** of real-world accuracy over time
- Saved models compared against fresh model on most recent fold — best kept

## Known accuracy (walk-forward CV)
- US stocks: ~40–55% (genuinely hard to predict next-day)
- Indian stocks: ~47–64% (Reliance + Nifty tend to score higher)
- One-day horizon is noisier than 3-day; 3-day was the original setting

## Key decisions & history
- Started as Random Forest, switched to XGBoost for better accuracy
- Horizon changed from 3 days → 1 day at user request
- Overall score formula (70/30) is acknowledged as somewhat arbitrary — confidence tends to be inflated (90%+) regardless of accuracy, which inflates the overall
- German Hub project deliberately excluded from portfolio — same user owns both
- Must run daily to accumulate live scorecard; retraining on 5yr data means single days don't shift the model much

## How to run
```bash
cd /Users/kritinbysani/Documents/Programing/Claude/stock-predictor
python predictor.py
```

## Dependencies
```
yfinance, pandas, numpy, xgboost, scikit-learn, joblib, matplotlib, notebook
```
Install: `pip install -r requirements.txt`

## Potential next improvements discussed
1. Sentiment signals (Reddit, news headlines)
2. SEC insider trading data (free, Form 4 filings)
3. Earnings surprise history
4. Regime detection (separate models for bull/bear/volatile)
5. Ensemble voting (XGBoost + LightGBM + Logistic Regression)
6. Sharpe ratio optimisation instead of raw accuracy
7. Rethink overall score formula — penalise high confidence when accuracy is low
