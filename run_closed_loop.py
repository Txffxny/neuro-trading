"""Closed-loop mode (uses the CL1): the neurons decide when to buy, and get
feedback when each trade closes. Compare conditions like the Connect Four run.

    python run_closed_loop.py --data data/SPY.csv --condition feedback
    python run_closed_loop.py --data data/SPY.csv --condition random_feedback
    python run_closed_loop.py --data data/SPY.csv --condition none

Honest caveat for the pitch: rewards arrive days after the decision, which is
much harder for neurons than Connect Four's instant feedback.
"""
import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cl
import backtest as bt
import neural_trading as nt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/SPY.csv")
ap.add_argument("--condition", choices=["feedback", "random_feedback", "none"], default="feedback")
ap.add_argument("--days", type=int, default=250)
args = ap.parse_args()

dates, closes = bt.load_prices(args.data)
rets = bt.daily_returns(closes)
start_day = max(nt.LOOKBACK, len(closes) - args.days)
end_day = len(closes) - 1

os.makedirs("results_trading", exist_ok=True)
out = f"results_trading/closed_{args.condition}_{Path(args.data).stem}_{datetime.now():%H%M%S}.csv"
trades, decisions, holding = [], 0, None

print(f"Simulator: {cl.is_simulator()}  |  {args.condition}  |  {dates[start_day]} to {dates[end_day]}")
t0 = time.time()
with open(out, "w", newline="") as f, cl.open() as neurons:
    w = csv.writer(f)
    w.writerow(["entry_date", "exit_date", "return"])
    for i in range(start_day, end_day):
        if holding is not None:
            entry, exit_, r = holding
            if i == exit_:
                trades.append(holding)
                w.writerow([dates[entry], dates[exit_], f"{r:.4f}"]); f.flush()
                if args.condition == "feedback":
                    nt.give_feedback(neurons, r > 0)
                elif args.condition == "random_feedback":
                    nt.give_feedback(neurons, random.random() < 0.5)
                else:
                    nt.rest(neurons, 1.0)
                holding = None
            continue  # can't buy while holding (or on the exit day)
        counts = nt.present(neurons, rets[i - nt.LOOKBACK + 1:i + 1])
        decisions += 1
        if nt.decide(counts):
            j, r = bt.trade_outcome(closes, i)
            holding = (i, j, r)
        if decisions % 25 == 0:
            print(f"{dates[i]}: {len(trades)} trades so far  ({time.time() - t0:.0f}s elapsed)")

s = bt.summarise(trades)
rate = len(trades) / max(1, decisions)
randoms = bt.random_baseline(closes, start_day, end_day, rate)
print(f"\nNeurons: {s['trades']} trades, return {s['total_return']:.1%}, win rate {s['win_rate']:.0%}")
print(f"Random traders, same trading rate (median): {randoms[len(randoms) // 2]:.1%}")
print(f"Neurons beat {bt.percentile_of(s['total_return'], randoms):.0%} of random traders")
print(f"Buy and hold: {bt.buy_and_hold(closes, start_day, end_day):.1%}")
print(f"Trades saved to {out}")
