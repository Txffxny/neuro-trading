"""Trading rules and scoring. No neurons in here, so it's easy to check.

Rules (the same for every trader we compare):
  - buy at the day's close when the signal says so (only if not already holding)
  - sell at the first close that is +3% (take profit) or -2% (stop loss)
  - otherwise sell after 10 trading days
"""
import csv
import random

TAKE_PROFIT = 0.03
STOP_LOSS = 0.02
MAX_HOLD = 10


def load_prices(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r["Date"] for r in rows], [float(r["Close"]) for r in rows]


def daily_returns(closes):
    return [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]


def trade_outcome(closes, i):
    """Enter at close i. Returns (exit_index, return)."""
    last = min(len(closes) - 1, i + MAX_HOLD)
    for j in range(i + 1, last + 1):
        r = closes[j] / closes[i] - 1
        if r >= TAKE_PROFIT or r <= -STOP_LOSS:
            return j, r
    return last, closes[last] / closes[i] - 1


def run_signals(closes, signal, start, end):
    """signal: set of day indices where the trader wants to buy.
    Trades never overlap. Returns a list of (entry, exit, return)."""
    trades, i = [], start
    while i < end:
        if i in signal and i + 1 < len(closes):
            j, r = trade_outcome(closes, i)
            trades.append((i, j, r))
            i = j + 1
        else:
            i += 1
    return trades


def summarise(trades):
    equity, peak, drawdown = 1.0, 1.0, 0.0
    for _, _, r in trades:
        equity *= 1 + r
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    wins = sum(1 for t in trades if t[2] > 0)
    return {
        "trades": len(trades),
        "total_return": equity - 1,
        "win_rate": wins / len(trades) if trades else 0.0,
        "max_drawdown": drawdown,
    }


def equity_curve(closes, trades, start, end):
    """Daily equity, flat when not holding."""
    curve, eq = [], 1.0
    by_entry = {t[0]: t for t in trades}
    i, holding = start, None
    while i <= end:
        if holding is None and i in by_entry:
            holding = by_entry[i]
            base = eq
        if holding is not None:
            entry, exit_, _ = holding
            eq = base * closes[i] / closes[entry]
            if i == exit_:
                holding = None
        curve.append(eq)
        i += 1
    return curve


def buy_and_hold(closes, start, end):
    return closes[end] / closes[start] - 1


def random_baseline(closes, start, end, rate, sims=1000, seed=0):
    """Total returns of `sims` random traders who buy on a random `rate`
    fraction of days, using the same exit rules."""
    rng = random.Random(seed)
    results = []
    for _ in range(sims):
        signal = {i for i in range(start, end) if rng.random() < rate}
        results.append(summarise(run_signals(closes, signal, start, end))["total_return"])
    return sorted(results)


def percentile_of(value, sorted_values):
    """What fraction of random traders did worse than `value`."""
    return sum(1 for v in sorted_values if v < value) / len(sorted_values)
