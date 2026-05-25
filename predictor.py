import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import uuid
import joblib
from datetime import datetime, date
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix

PERIOD   = "5y"
HORIZON  = 1
N_SPLITS = 5

US_TICKERS    = ["AAPL", "TSLA", "NVDA", "^GSPC"]
INDIA_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "^NSEI"]

DISPLAY_NAMES = {
    "AAPL":        "Apple (AAPL)",
    "TSLA":        "Tesla (TSLA)",
    "NVDA":        "Nvidia (NVDA)",
    "^GSPC":       "S&P 500",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":      "Tata Consultancy Svcs",
    "HDFCBANK.NS": "HDFC Bank",
    "^NSEI":       "Nifty 50",
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
WHITE  = "\033[97m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREY   = "\033[90m"

def clr(text, *codes):
    return "".join(codes) + str(text) + RESET

def banner():
    print()
    print(clr("┌─────────────────────────────────────────────┐", CYAN))
    print(clr("│        STOCK DIRECTION PREDICTOR  v3        │", CYAN, BOLD))
    print(clr("│   US · India  ·  Memory & Live Scorecard    │", CYAN, DIM))
    print(clr("└─────────────────────────────────────────────┘", CYAN))
    print()

def section(title):
    print(clr(f"\n  ▸ {title}", YELLOW, BOLD))
    print(clr("  " + "─" * 43, GREY))

def market_header(label):
    print()
    print(clr(f"  {'━' * 43}", WHITE))
    print(clr(f"  {label.center(43)}", WHITE, BOLD))
    print(clr(f"  {'━' * 43}", WHITE))

def explain(text):
    print(clr(f"    → {text}", GREY))

def progress(msg):
    print(clr(f"  ⟳  {msg}...", DIM), end="\r", flush=True)

def done(msg):
    print(clr(f"  ✓  {msg}          ", GREEN))

def accuracy_label(acc):
    if acc >= 0.62:
        return clr(f"{acc:.1%}  (strong)", GREEN, BOLD)
    if acc >= 0.55:
        return clr(f"{acc:.1%}  (decent)", YELLOW, BOLD)
    return clr(f"{acc:.1%}  (coin-flip territory)", RED, BOLD)

def overall_score(acc, confidence):
    # model accuracy carries more weight — it's the proven track record
    return acc * 0.7 + confidence * 0.3

def overall_label(score):
    if score >= 0.68:
        return clr(f"{score:.1%}  ★ High", GREEN, BOLD)
    if score >= 0.58:
        return clr(f"{score:.1%}  ◆ Moderate", YELLOW, BOLD)
    return clr(f"{score:.1%}  ◇ Low", RED, BOLD)

def confidence_label(conf):
    if conf >= 0.70:
        return clr(f"{conf:.1%}  (high)", GREEN, BOLD)
    if conf >= 0.55:
        return clr(f"{conf:.1%}  (moderate)", YELLOW, BOLD)
    return clr(f"{conf:.1%}  (low — treat with caution)", RED, BOLD)


# ── History ───────────────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(clr("  ⚠  History file corrupted — starting fresh.", YELLOW))
    return {"runs": [], "live_scores": {}}


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def business_days_between(d1_str, d2_str):
    d1 = pd.Timestamp(d1_str)
    d2 = pd.Timestamp(d2_str)
    return len(pd.bdate_range(d1, d2)) - 1


def verify_past_predictions(history, raw):
    """Check old predictions where HORIZON business days have passed."""
    today_str = date.today().isoformat()
    any_verified = False

    for run in history["runs"]:
        for ticker, entry in run["tickers"].items():
            if entry.get("verified"):
                continue
            if business_days_between(run["date"], today_str) < HORIZON:
                continue

            # Enough time has passed — check actual outcome
            col = f"Close_{ticker}"
            if col not in raw.columns:
                continue

            price_then = entry["price"]
            # find closest date >= prediction date + HORIZON in the data
            pred_date  = pd.Timestamp(run["date"])
            future_idx = raw.index[raw.index >= pred_date]
            if len(future_idx) <= HORIZON:
                continue

            price_now = float(raw[col].loc[future_idx[HORIZON]])
            actual    = "UP" if price_now > price_then else "DOWN"
            correct   = actual == entry["prediction"]

            entry["verified"]     = True
            entry["actual"]       = actual
            entry["price_then"]   = price_then
            entry["price_now"]    = price_now
            entry["correct"]      = correct

            # update live score
            if ticker not in history["live_scores"]:
                history["live_scores"][ticker] = {"correct": 0, "total": 0}
            history["live_scores"][ticker]["total"]   += 1
            history["live_scores"][ticker]["correct"] += int(correct)

            any_verified = True

    return history, any_verified


def print_scorecard(history):
    """Print live prediction accuracy from past runs."""
    scores = history.get("live_scores", {})
    if not scores:
        return

    section("LIVE SCORECARD — Past Prediction Results")
    explain("These are real outcomes — did our past predictions actually come true?")
    print()
    print(clr(f"  {'Asset':<25}{'Correct':>10}{'Total':>8}{'Live Accuracy':>16}", GREY))
    print(clr("  " + "─" * 59, GREY))

    all_correct = all_total = 0
    for ticker in US_TICKERS + INDIA_TICKERS:
        s = scores.get(ticker)
        if not s or s["total"] == 0:
            continue
        acc = s["correct"] / s["total"]
        acc_color = GREEN if acc >= 0.55 else (YELLOW if acc >= 0.50 else RED)
        trend = "↑" if acc >= 0.55 else ("→" if acc >= 0.50 else "↓")
        print(
            clr(f"  {DISPLAY_NAMES.get(ticker, ticker):<25}", WHITE) +
            clr(f"{s['correct']:>10}", GREEN) +
            clr(f"{s['total']:>8}", GREY) +
            clr(f"  {trend}  {acc:.1%}", acc_color, BOLD)
        )
        all_correct += s["correct"]
        all_total   += s["total"]

    if all_total > 0:
        overall = all_correct / all_total
        print(clr("  " + "─" * 59, GREY))
        print(
            clr(f"  {'OVERALL':<25}", WHITE, BOLD) +
            clr(f"{all_correct:>10}", GREEN) +
            clr(f"{all_total:>8}", GREY) +
            clr(f"  {'↑' if overall >= 0.55 else '→' if overall >= 0.5 else '↓'}  {overall:.1%}", GREEN if overall >= 0.55 else YELLOW if overall >= 0.50 else RED, BOLD)
        )


def print_recent_verifications(history):
    """Show the most recently verified predictions."""
    verified = []
    for run in history["runs"]:
        for ticker, entry in run["tickers"].items():
            if entry.get("verified"):
                verified.append({
                    "date": run["date"], "ticker": ticker,
                    "name": DISPLAY_NAMES.get(ticker, ticker),
                    "prediction": entry["prediction"],
                    "actual": entry["actual"],
                    "correct": entry["correct"],
                    "price_then": entry["price_then"],
                    "price_now": entry["price_now"],
                })

    if not verified:
        return

    recent = sorted(verified, key=lambda x: x["date"], reverse=True)[:6]

    section("RECENT PREDICTION OUTCOMES")
    explain(f"Checking what actually happened to our past {HORIZON}-day predictions.")
    print()
    print(clr(f"  {'Date':<12}{'Asset':<25}{'Predicted':>10}{'Actual':>8}{'Result':>10}", GREY))
    print(clr("  " + "─" * 65, GREY))

    for v in recent:
        result_str = clr("  ✓ CORRECT", GREEN, BOLD) if v["correct"] else clr("  ✗ WRONG", RED, BOLD)
        pred_clr   = GREEN if v["prediction"] == "UP" else RED
        actual_clr = GREEN if v["actual"] == "UP" else RED
        print(
            clr(f"  {v['date']:<12}", GREY) +
            clr(f"{v['name']:<25}", WHITE) +
            clr(f"{v['prediction']:>10}", pred_clr) +
            clr(f"{v['actual']:>8}", actual_clr) +
            result_str
        )


# ── Model persistence ─────────────────────────────────────────────────────────
def model_path(ticker):
    safe = ticker.replace("^", "").replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}.joblib")


def save_model(model, ticker):
    joblib.dump(model, model_path(ticker))


def load_model(ticker):
    path = model_path(ticker)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ── Data ──────────────────────────────────────────────────────────────────────
def fetch_all(period):
    all_symbols = list(set(US_TICKERS + INDIA_TICKERS + ["^VIX"]))
    raw = yf.download(all_symbols, period=period, auto_adjust=True, progress=False)
    raw.columns = ["_".join(c).strip() for c in raw.columns]
    return raw


def add_features(raw, ticker, market_index):
    c   = raw[f"Close_{ticker}"]
    vol = raw.get(f"Volume_{ticker}", pd.Series(1, index=raw.index))
    mkt = raw[f"Close_{market_index}"]
    vix = raw["Close_^VIX"]

    out = pd.DataFrame(index=raw.index)

    out["sma_5"]   = c.rolling(5).mean()
    out["sma_20"]  = c.rolling(20).mean()
    out["sma_50"]  = c.rolling(50).mean()
    out["ema_12"]  = c.ewm(span=12).mean()
    out["ema_26"]  = c.ewm(span=26).mean()
    out["macd"]    = out["ema_12"] - out["ema_26"]

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - (100 / (1 + gain / loss))

    rolling_std        = c.rolling(20).std()
    bb_upper           = out["sma_20"] + 2 * rolling_std
    bb_lower           = out["sma_20"] - 2 * rolling_std
    out["bb_position"] = (c - bb_lower) / (bb_upper - bb_lower)
    out["bb_width"]    = (bb_upper - bb_lower) / out["sma_20"]

    hi_col = f"High_{ticker}"
    lo_col = f"Low_{ticker}"
    out["daily_range"] = (raw[hi_col] - raw[lo_col]) / c if hi_col in raw.columns else 0.0

    out["momentum_10"] = c - c.shift(10)
    out["vol_ratio"]   = vol / vol.rolling(20).mean()

    for d in [1, 2, 3, 5, 10]:
        out[f"ret_{d}d"] = c.pct_change(fill_method=None, periods=d)
    out["vol_chg_1d"] = vol.pct_change(fill_method=None, periods=1)
    out["vol_chg_5d"] = vol.pct_change(fill_method=None, periods=5)

    open_col = f"Open_{ticker}"
    out["gap"] = (raw[open_col] - c.shift(1)) / c.shift(1) if open_col in raw.columns else 0.0

    out["mkt_ret_1d"]   = mkt.pct_change(fill_method=None, periods=1)
    out["mkt_ret_5d"]   = mkt.pct_change(fill_method=None, periods=5)
    out["mkt_vs_sma20"] = mkt / mkt.rolling(20).mean() - 1
    out["vix"]          = vix
    out["vix_chg_1d"]   = vix.pct_change(fill_method=None, periods=1)

    out["target"] = (c.shift(-HORIZON) > c).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out, c


def train_and_predict(df, ticker):
    features = [col for col in df.columns if col != "target"]
    X = df[features].values
    y = df["target"].values

    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )

    # walk-forward CV on all but the last fold (kept for final comparison)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(X))
    fold_scores, all_preds, all_true = [], [], []

    for train_idx, test_idx in splits:
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        fold_scores.append(accuracy_score(y[test_idx], preds))
        all_preds.extend(preds)
        all_true.extend(y[test_idx])

    overall = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    # compare against last saved model on the final held-out fold only
    last_train_idx, last_test_idx = splits[-1]
    saved = load_model(ticker)
    if saved is not None:
        saved_acc = accuracy_score(y[last_test_idx], saved.predict(X[last_test_idx]))
        fresh_acc = fold_scores[-1]
        if saved_acc > fresh_acc:
            # saved model is still better on recent data — keep using it
            model = saved

    # retrain final model on all data and save
    model.fit(X, y)
    save_model(model, ticker)

    latest     = df[features].iloc[[-1]]
    prob       = model.predict_proba(latest.values)[0]
    direction  = "UP" if prob[1] >= 0.5 else "DOWN"
    confidence = prob[1] if direction == "UP" else prob[0]

    return overall, fold_scores, cm, direction, confidence, features


# ── Display helpers ───────────────────────────────────────────────────────────
def run_ticker(raw, ticker, market_index, results, run_entry):
    name = DISPLAY_NAMES[ticker]
    section(f"ANALYSING — {name}")
    progress("Building features and training model")
    df, close = add_features(raw, ticker, market_index)
    acc, fold_scores, cm, direction, confidence, features = train_and_predict(df, ticker)

    last_price = float(close.dropna().iloc[-1])
    score = overall_score(acc, confidence)
    results[ticker] = {
        "name": name, "price": last_price,
        "acc": acc, "direction": direction, "confidence": confidence,
        "score": score,
    }
    run_entry["tickers"][ticker] = {
        "price": float(last_price), "prediction": direction,
        "confidence": float(confidence), "cv_accuracy": float(acc),
        "verified": False, "actual": None, "correct": None,
    }

    dcolor = GREEN if direction == "UP" else RED
    arrow  = "↑" if direction == "UP" else "↓"
    done("Done")
    print()

    for i, s in enumerate(fold_scores, 1):
        bar   = "█" * int(s * 20)
        color = GREEN if s >= 0.55 else (YELLOW if s >= 0.50 else RED)
        print(clr(f"    Fold {i}:  {bar:<20}  {s:.1%}", color))

    print()
    print(f"  {'Model Accuracy':<28}{accuracy_label(acc)}")
    print(f"  {'Confidence':<28}{confidence_label(confidence)}")
    print(f"  {'Overall Prediction Accuracy':<28}{overall_label(score)}")
    print(f"  {'Last price':<28}{clr(f'${last_price:,.2f}', WHITE, BOLD)}")
    print(f"  {'Prediction (tomorrow)':<28}{clr(f'{arrow}  {direction}', dcolor, BOLD)}")
    explain("Model Accuracy = historical hit rate from back-testing.")
    explain("Confidence = how strongly the model leans on today's data.")
    explain("Overall = combined score (70% accuracy + 30% confidence).")


def print_summary_table(label, tickers, results, currency="$"):
    print()
    print(clr(f"  {label}", WHITE, BOLD))
    print(clr(f"  {'Asset':<25}{'Price':>12}{'Mdl Acc':>9}{'Confidence':>12}{'Overall':>10}{'Prediction':>12}", GREY))
    print(clr("  " + "─" * 80, GREY))

    for ticker in tickers:
        r = results[ticker]
        dcolor    = GREEN if r["direction"] == "UP" else RED
        arrow     = "↑" if r["direction"] == "UP" else "↓"
        acc_color = GREEN if r["acc"] >= 0.55 else (YELLOW if r["acc"] >= 0.50 else RED)
        sc        = r["score"]
        sc_color  = GREEN if sc >= 0.68 else (YELLOW if sc >= 0.58 else RED)

        print(
            clr(f"  {r['name']:<25}", WHITE) +
            clr(f"{currency}{r['price']:>11,.2f}", CYAN) +
            clr(f"{r['acc']:>8.1%}", acc_color) +
            clr(f"{r['confidence']:>12.1%}", WHITE) +
            clr(f"{sc:>9.1%}", sc_color, BOLD) +
            clr(f"  {arrow} {r['direction']:<8}", dcolor, BOLD)
        )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner()

    # ── Load history ──
    history = load_history()
    run_count = len(history["runs"])
    last_run  = history["runs"][-1]["date"] if run_count > 0 else "never"

    print(clr(f"  Total runs logged : {run_count}", GREY))
    print(clr(f"  Last run          : {last_run}", GREY))
    print(clr(f"  Today             : {date.today().isoformat()}", GREY))

    # ── Fetch data ──
    section("STEP 1 — Fetching Market Data")
    progress("Downloading all US and India stocks, indices, and VIX")
    raw = fetch_all(PERIOD)
    done(f"Downloaded {len(raw)} trading days for all assets")
    explain("US stocks use S&P 500 as market context.")
    explain("Indian stocks use Nifty 50 as market context.")

    # ── Verify past predictions ──
    history, any_verified = verify_past_predictions(history, raw)
    if any_verified:
        save_history(history)
        print_recent_verifications(history)

    print_scorecard(history)

    # ── New run entry ──
    run_entry = {
        "run_id": str(uuid.uuid4())[:8],
        "date":   date.today().isoformat(),
        "time":   datetime.now().strftime("%H:%M:%S"),
        "tickers": {},
    }

    results = {}

    # ── US Markets ──
    market_header("  US MARKETS")
    for ticker in US_TICKERS:
        run_ticker(raw, ticker, "^GSPC", results, run_entry)

    # ── India Markets ──
    market_header("  INDIA MARKETS")
    for ticker in INDIA_TICKERS:
        run_ticker(raw, ticker, "^NSEI", results, run_entry)

    # ── Save this run ──
    history["runs"].append(run_entry)
    save_history(history)

    # ── Summary ──
    section("SUMMARY — All Predictions")
    explain(f"Horizon: {HORIZON} trading days  |  Model: XGBoost  |  Walk-forward CV")
    explain(f"Predictions saved — will be verified automatically in {HORIZON} trading days.")

    print_summary_table("US Markets", US_TICKERS, results, currency="$")
    print_summary_table("India Markets", INDIA_TICKERS, results, currency="₹")

    print()
    print(clr("  ─────────────────────────────────────────────", GREY))
    print(clr("  ⚠  This is a learning project, not financial advice.", GREY))
    print(clr("     Never trade real money based on a model's output.", GREY))
    print(clr("  ─────────────────────────────────────────────", GREY))
    print()
