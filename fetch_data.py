"""Download daily prices (needs internet: run this BEFORE the event), or make
a synthetic test series.

    python fetch_data.py                 # SPY, last 3 years -> data/SPY.csv
    python fetch_data.py --ticker AAPL   # any Yahoo Finance ticker
    python fetch_data.py --synthetic     # fake prices with a hidden pattern -> data/synthetic.csv

The synthetic series has trends built in (prices drift up or down for a few
weeks at a time), so a model that reads recent returns SHOULD find an edge there.
It's the sanity check: if the pipeline can't beat random on synthetic data,
something is broken.
"""
import argparse
import csv
import os
import random
from datetime import date, timedelta

os.makedirs("data", exist_ok=True)


def save(path, dates, closes):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Close"])
        for d, c in zip(dates, closes):
            w.writerow([d, f"{c:.4f}"])
    print(f"Saved {len(closes)} days to {path}")


def synthetic(days=750, seed=7):
    """Trending regimes: prices drift up for a while, then down for a while,
    with daily noise on top. Recent returns therefore hint at the next few days."""
    rng = random.Random(seed)
    price, drift = 100.0, 0.004
    dates, closes = [], []
    d = date(2023, 1, 2)
    while len(closes) < days:
        if d.weekday() < 5:
            if rng.random() < 0.06:          # trends last ~3-4 weeks on average
                drift = rng.choice([0.004, -0.004])
            price *= 1 + drift + rng.gauss(0, 0.01)
            dates.append(d.isoformat())
            closes.append(price)
        d += timedelta(days=1)
    return dates, closes


ap = argparse.ArgumentParser()
ap.add_argument("--ticker", default="SPY")
ap.add_argument("--period", default="3y")
ap.add_argument("--synthetic", action="store_true")
args = ap.parse_args()

if args.synthetic:
    save("data/synthetic.csv", *synthetic())
else:
    import yfinance as yf
    df = yf.download(args.ticker, period=args.period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data came back for {args.ticker}. Check the ticker and your internet connection.")
    close = df["Close"]
    if hasattr(close, "columns"):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    close = close.dropna()
    save(f"data/{args.ticker}.csv", [d.strftime("%Y-%m-%d") for d in close.index], [float(c) for c in close])
