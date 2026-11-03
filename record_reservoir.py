"""Reservoir mode, step 1 (uses the CL1): show the neurons each day's recent
returns and record how they respond. No feedback, no learning in the neurons.

    python record_reservoir.py --data data/SPY.csv --days 500

About 0.5 s per day in real time, so 500 days takes roughly 4-5 minutes.
Saves results_trading/reservoir_<data>_<time>.npz for analyse_reservoir.py.
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cl
import backtest as bt
import neural_trading as nt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/SPY.csv")
ap.add_argument("--days", type=int, default=500, help="most recent N days to present")
ap.add_argument("--rest", type=float, default=0.2, help="seconds of rest between days")
args = ap.parse_args()

dates, closes = bt.load_prices(args.data)
rets = bt.daily_returns(closes)
first = max(nt.LOOKBACK, len(closes) - args.days)
days = list(range(first, len(closes)))

os.makedirs("results_trading", exist_ok=True)
out = f"results_trading/reservoir_{Path(args.data).stem}_{datetime.now():%H%M%S}.npz"
counts = np.zeros((len(days), 64), dtype=np.int32)

print(f"Simulator: {cl.is_simulator()}  |  {len(days)} days from {dates[first]} to {dates[-1]}")
start = time.time()
with cl.open() as neurons:
    for n, i in enumerate(days):
        counts[n] = nt.present(neurons, rets[i - nt.LOOKBACK + 1:i + 1])
        nt.rest(neurons, args.rest)
        if n % 50 == 0 or n == len(days) - 1:
            np.savez(out, days=np.array(days[:n + 1]), counts=counts[:n + 1], data=args.data)
            print(f"day {n + 1}/{len(days)}  ({time.time() - start:.0f}s elapsed)")

print(f"Saved {out}\nNext: python analyse_reservoir.py {out}")
