import json
import os
import yfinance as yf
import pandas as pd
from datetime import date, datetime
from math import floor

TRADES_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.json")
HISTORY_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")

STARTING_CAPITAL = 100_000   # simulated dollars
MAX_POS_PCT      = 0.25      # max 25% of portfolio per stock
STOP_LOSS_PCT    = 0.02      # exit if position drops 2%
TAKE_PROFIT_PCT  = 0.04      # exit if position gains 4%
MIN_SCORE        = 0.58      # only trade if overall score >= 58%

US_TICKERS = ["AAPL", "TSLA", "NVDA", "^GSPC"]

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
WHITE  = "\033[97m"; CYAN  = "\033[96m"; GREEN = "\033[92m"
RED    = "\033[91m"; YELLOW= "\033[93m"; GREY  = "\033[90m"

def clr(text, *codes): return "".join(codes) + str(text) + RESET
def section(t): print(clr(f"\n  ▸ {t}", YELLOW, BOLD)); print(clr("  " + "─"*43, GREY))
def explain(t): print(clr(f"    → {t}", GREY))
def done(t):    print(clr(f"  ✓  {t}          ", GREEN))


# ── Persistence ───────────────────────────────────────────────────────────────
def load_trades():
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {
        "cash": STARTING_CAPITAL,
        "starting_capital": STARTING_CAPITAL,
        "positions": {},
        "trade_log": [],
        "snapshots": [],
    }

def save_trades(data):
    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"runs": []}


# ── Price fetch ───────────────────────────────────────────────────────────────
def get_current_prices(tickers):
    clean = [t for t in tickers if t != "^GSPC"]
    prices = {}
    if not clean:
        return prices
    raw = yf.download(clean, period="2d", auto_adjust=True, progress=False)
    if raw.empty:
        return prices
    raw.columns = ["_".join(c).strip() if isinstance(c, tuple) else c for c in raw.columns]
    for t in clean:
        col = f"Close_{t}"
        if col in raw.columns:
            prices[t] = float(raw[col].dropna().iloc[-1])
    return prices


# ── Trade logic ───────────────────────────────────────────────────────────────
def overall_score(acc, confidence):
    return acc * 0.7 + confidence * 0.3

def portfolio_value(data, prices):
    pos_value = sum(
        pos["shares"] * prices.get(ticker, pos["entry_price"])
        for ticker, pos in data["positions"].items()
    )
    return data["cash"] + pos_value

def position_value(data, prices):
    return sum(
        pos["shares"] * prices.get(ticker, pos["entry_price"])
        for ticker, pos in data["positions"].items()
    )

def check_exits(data, prices, today):
    """Check stop loss and take profit on all open positions."""
    to_close = []
    for ticker, pos in data["positions"].items():
        price = prices.get(ticker)
        if price is None:
            continue
        change = (price - pos["entry_price"]) / pos["entry_price"]
        if change <= -STOP_LOSS_PCT:
            to_close.append((ticker, price, "STOP LOSS"))
        elif change >= TAKE_PROFIT_PCT:
            to_close.append((ticker, price, "TAKE PROFIT"))

    for ticker, price, reason in to_close:
        pos    = data["positions"][ticker]
        value  = pos["shares"] * price
        profit = value - (pos["shares"] * pos["entry_price"])
        data["cash"] += value
        data["trade_log"].append({
            "date": today, "ticker": ticker, "action": "SELL",
            "reason": reason, "shares": pos["shares"],
            "price": round(price, 2), "value": round(value, 2),
            "profit": round(profit, 2),
        })
        del data["positions"][ticker]
        pclr = GREEN if profit >= 0 else RED
        print(
            clr(f"  {reason:<14}", YELLOW, BOLD) +
            clr(f"  {ticker:<12}", WHITE) +
            clr(f"  Sold @ ${price:.2f}  ", CYAN) +
            clr(f"  P&L: ${profit:+.2f}", pclr, BOLD)
        )

    return data


def apply_predictions(data, predictions, prices, today):
    """Open/close positions based on today's predictions."""
    total_val = portfolio_value(data, prices)

    for ticker, pred in predictions.items():
        if ticker not in prices:
            continue
        price  = prices[ticker]
        score  = overall_score(pred["cv_accuracy"], pred["confidence"])
        direction = pred["prediction"]

        # close if prediction flipped to DOWN or score too low
        if ticker in data["positions"]:
            if direction == "DOWN" or score < MIN_SCORE:
                pos    = data["positions"][ticker]
                value  = pos["shares"] * price
                profit = value - (pos["shares"] * pos["entry_price"])
                data["cash"] += value
                data["trade_log"].append({
                    "date": today, "ticker": ticker, "action": "SELL",
                    "reason": "SIGNAL FLIP" if direction == "DOWN" else "LOW SCORE",
                    "shares": pos["shares"], "price": round(price, 2),
                    "value": round(value, 2), "profit": round(profit, 2),
                })
                del data["positions"][ticker]
                pclr = GREEN if profit >= 0 else RED
                print(
                    clr(f"  SELL (signal)  ", YELLOW, BOLD) +
                    clr(f"  {ticker:<12}", WHITE) +
                    clr(f"  @ ${price:.2f}  ", CYAN) +
                    clr(f"  P&L: ${profit:+.2f}", pclr, BOLD)
                )
            continue

        # open new position if UP and score high enough
        if direction == "UP" and score >= MIN_SCORE:
            allocation = total_val * MAX_POS_PCT
            if data["cash"] < allocation * 0.5:
                continue   # not enough cash
            allocation = min(allocation, data["cash"])
            shares     = floor(allocation / price)
            if shares < 1:
                continue
            cost = shares * price
            data["cash"] -= cost
            data["positions"][ticker] = {
                "shares": shares,
                "entry_price": round(price, 2),
                "entry_date": today,
                "stop_loss": round(price * (1 - STOP_LOSS_PCT), 2),
                "take_profit": round(price * (1 + TAKE_PROFIT_PCT), 2),
            }
            data["trade_log"].append({
                "date": today, "ticker": ticker, "action": "BUY",
                "reason": "SIGNAL", "shares": shares,
                "price": round(price, 2), "value": round(cost, 2),
                "profit": None,
            })
            print(
                clr(f"  BUY            ", GREEN, BOLD) +
                clr(f"  {ticker:<12}", WHITE) +
                clr(f"  {shares} shares @ ${price:.2f}  ", CYAN) +
                clr(f"  Score: {score:.1%}", YELLOW)
            )

    return data


# ── Display ───────────────────────────────────────────────────────────────────
def banner():
    print()
    print(clr("┌─────────────────────────────────────────────┐", CYAN))
    print(clr("│       SIMULATED PAPER TRADER  v1            │", CYAN, BOLD))
    print(clr("│   No real money  ·  US stocks  ·  $100k     │", CYAN, DIM))
    print(clr("└─────────────────────────────────────────────┘", CYAN))
    print()

def print_portfolio(data, prices):
    total    = portfolio_value(data, prices)
    start    = data["starting_capital"]
    total_pl = total - start
    pl_pct   = (total_pl / start) * 100
    pl_clr   = GREEN if total_pl >= 0 else RED

    section("PORTFOLIO OVERVIEW")
    print(f"  {'Starting Capital':<28}{clr(f'${start:>12,.2f}', WHITE)}")
    cash_str = f"${data['cash']:>12,.2f}"
    pos_str  = f"${position_value(data, prices):>12,.2f}"
    print(f"  {'Cash Available':<28}{clr(cash_str, CYAN)}")
    print(f"  {'Positions Value':<28}{clr(pos_str, CYAN)}")
    print(f"  {'Total Portfolio Value':<28}{clr(f'${total:>12,.2f}', WHITE, BOLD)}")
    print(f"  {'Total P&L':<28}{clr(f'${total_pl:>+12,.2f}  ({pl_pct:+.2f}%)', pl_clr, BOLD)}")

    if data["positions"]:
        section("OPEN POSITIONS")
        print(clr(f"  {'Ticker':<10}{'Shares':>8}{'Entry':>10}{'Current':>10}{'Stop':>10}{'Target':>10}{'P&L':>12}", GREY))
        print(clr("  " + "─" * 60, GREY))
        for ticker, pos in data["positions"].items():
            price  = prices.get(ticker, pos["entry_price"])
            pl     = (price - pos["entry_price"]) * pos["shares"]
            pl_pct_pos = (price - pos["entry_price"]) / pos["entry_price"] * 100
            pclr   = GREEN if pl >= 0 else RED
            print(
                clr(f"  {ticker:<10}", WHITE, BOLD) +
                clr(f"{pos['shares']:>8}", WHITE) +
                clr(f"${pos['entry_price']:>9.2f}", GREY) +
                clr(f"${price:>9.2f}", CYAN) +
                clr(f"${pos['stop_loss']:>9.2f}", RED) +
                clr(f"${pos['take_profit']:>9.2f}", GREEN) +
                clr(f"  ${pl:>+9.2f} ({pl_pct_pos:+.1f}%)", pclr, BOLD)
            )

    if data["trade_log"]:
        section("RECENT TRADES")
        recent = data["trade_log"][-8:]
        print(clr(f"  {'Date':<12}{'Ticker':<10}{'Action':<16}{'Shares':>6}{'Price':>10}{'P&L':>12}", GREY))
        print(clr("  " + "─" * 66, GREY))
        for t in reversed(recent):
            aclr  = GREEN if t["action"] == "BUY" else RED
            pl_str = clr(f"  ${t['profit']:>+9.2f}", GREEN if (t.get("profit") or 0) >= 0 else RED) if t.get("profit") is not None else clr("         —", GREY)
            print(
                clr(f"  {t['date']:<12}", GREY) +
                clr(f"{t['ticker']:<10}", WHITE) +
                clr(f"{t['action']} ({t['reason']}){'':<4}", aclr, BOLD) +
                clr(f"{t['shares']:>6}", WHITE) +
                clr(f"  ${t['price']:>8.2f}", CYAN) +
                pl_str
            )

    # win rate
    closed = [t for t in data["trade_log"] if t.get("profit") is not None]
    if closed:
        wins    = sum(1 for t in closed if t["profit"] > 0)
        win_pct = wins / len(closed) * 100
        total_profit = sum(t["profit"] for t in closed)
        wclr = GREEN if win_pct >= 55 else (YELLOW if win_pct >= 45 else RED)
        section("STATS")
        print(f"  {'Closed Trades':<28}{clr(str(len(closed)), WHITE, BOLD)}")
        print(f"  {'Win Rate':<28}{clr(f'{win_pct:.1f}%', wclr, BOLD)}")
        print(f"  {'Total Realised P&L':<28}{clr(f'${total_profit:+,.2f}', GREEN if total_profit >= 0 else RED, BOLD)}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner()
    today = date.today().isoformat()

    # load state
    data    = load_trades()
    history = load_history()

    # get latest predictions from history
    predictions = {}
    if history["runs"]:
        latest_run = history["runs"][-1]
        if latest_run["date"] == today:
            for ticker, entry in latest_run["tickers"].items():
                if ticker in US_TICKERS and ticker != "^GSPC":
                    predictions[ticker] = entry
        else:
            explain(f"Latest predictions are from {latest_run['date']}, not today.")
            explain("Run predictor.py first to get today's predictions, then re-run trader.py.")

    tradeable = [t for t in predictions] if predictions else list(data["positions"].keys())

    section("STEP 1 — Fetching Current Prices")
    all_tickers = list(set(list(predictions.keys()) + list(data["positions"].keys())))
    prices = get_current_prices([t for t in all_tickers if t != "^GSPC"])
    done(f"Got prices for {len(prices)} stocks")
    for t, p in prices.items():
        print(clr(f"    {t:<14}  ${p:,.2f}", GREY))

    section("STEP 2 — Checking Stop Losses & Take Profits")
    if data["positions"]:
        data = check_exits(data, prices, today)
        if not any(True for _ in []):
            explain("No stop loss or take profit triggers today.")
    else:
        explain("No open positions to check.")

    if predictions:
        section("STEP 3 — Applying Today's Predictions")
        explain(f"Only opening trades with Overall Score ≥ {MIN_SCORE:.0%}")
        explain(f"Max {MAX_POS_PCT:.0%} of portfolio per stock · Stop {STOP_LOSS_PCT:.0%} · Target +{TAKE_PROFIT_PCT:.0%}")
        data = apply_predictions(data, predictions, prices, today)
    else:
        section("STEP 3 — No New Predictions")
        explain("Run predictor.py first to generate today's signals.")

    # snapshot
    total = portfolio_value(data, prices)
    data["snapshots"].append({
        "date": today,
        "portfolio_value": round(total, 2),
        "cash": round(data["cash"], 2),
        "open_positions": len(data["positions"]),
    })

    save_trades(data)
    print_portfolio(data, prices)

    print()
    print(clr("  ─────────────────────────────────────────────", GREY))
    print(clr("  ⚠  Simulated trading only. No real money involved.", GREY))
    print(clr("  ─────────────────────────────────────────────", GREY))
    print()
    print(clr("  Tip: run predictor.py first, then trader.py each day.", DIM))
    print()
