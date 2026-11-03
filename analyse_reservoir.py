"""Reservoir mode, step 2 (runs on your laptop, no CL1 needed).

    python analyse_reservoir.py results_trading/reservoir_SPY_190512.npz

Trains a simple model on the first 60% of days to predict "would a trade
started today be profitable?", then trades the last 40% with it. Compares:
  - Neurons:            model reading the neurons' spike counts
  - No neurons:         the same model reading the raw returns directly
  - Shuffled neurons:   spike counts matched to the wrong days (should be useless)
  - 1,000 random traders who trade just as often, with the same exit rules
  - Buy and hold
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import backtest as bt
import neural_trading as nt

TRAIN_FRACTION = 0.6
TRADE_RATE = 0.3   # trade on the 30% of days the model likes most


def ridge_fit(X, y, alpha=10.0):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X1.T @ X1 + alpha * np.eye(X1.shape[1]), X1.T @ y)


def ridge_score(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    raise SystemExit("Give the .npz file from record_reservoir.py, e.g.\n  python analyse_reservoir.py results_trading/reservoir_SPY_190512.npz")
saved = np.load(path, allow_pickle=True)
days, counts, data_path = saved["days"], saved["counts"], str(saved["data"])
dates, closes = bt.load_prices(data_path)
rets = bt.daily_returns(closes)

labels = np.array([1.0 if bt.trade_outcome(closes, i)[1] > 0 else 0.0 for i in days])
split = int(len(days) * TRAIN_FRACTION)
train = np.arange(split - bt.MAX_HOLD)   # gap so training labels never peek into the test period
test = np.arange(split, len(days))
start_day, end_day = int(days[test[0]]), int(days[test[-1]])

X_neurons = counts[:, nt.FEATURE_CHANNELS].astype(float)
X_raw = np.array([rets[i - nt.LOOKBACK + 1:i + 1] for i in days])
rng = np.random.default_rng(0)
X_shuffled = X_neurons[rng.permutation(len(X_neurons))]

results, curves = {}, {}
for name, X in [("Neurons", X_neurons), ("No neurons (raw returns)", X_raw), ("Shuffled neurons", X_shuffled)]:
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
    Z = (X - mu) / sd
    w = ridge_fit(Z[train], labels[train])
    threshold = np.quantile(ridge_score(Z[train], w), 1 - TRADE_RATE)
    picks = {int(days[k]) for k in test if ridge_score(Z[k:k + 1], w)[0] > threshold}
    trades = bt.run_signals(closes, picks, start_day, end_day)
    results[name] = bt.summarise(trades)
    curves[name] = bt.equity_curve(closes, trades, start_day, end_day)

random_returns = bt.random_baseline(closes, start_day, end_day, TRADE_RATE)
bh = bt.buy_and_hold(closes, start_day, end_day)

print(f"\nData: {data_path}   test period: {dates[start_day]} to {dates[end_day]}  ({len(test)} days)")
print(f"Spikes recorded per day (average): {counts.sum(1).mean():.1f}\n")
print(f"{'Trader':28s}{'Trades':>8s}{'Return':>10s}{'Win rate':>10s}{'Max drop':>10s}{'Beats random':>14s}")
for name, s in results.items():
    pct = f"{bt.percentile_of(s['total_return'], random_returns):.0%}" if s["trades"] else "no trades"
    print(f"{name:28s}{s['trades']:>8d}{s['total_return']:>10.1%}{s['win_rate']:>10.0%}{s['max_drawdown']:>10.1%}{pct:>14s}")
print(f"{'Random traders (median)':28s}{'':>8s}{random_returns[len(random_returns) // 2]:>10.1%}")
print(f"{'Buy and hold':28s}{'':>8s}{bh:>10.1%}")
print("\n'Beats random' = share of 1,000 random traders with the same exit rules that did worse.")
print("Above 95% is unlikely to be luck; anything from 5% to 95% is consistent with chance.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
colors = {"Neurons": "#D8342C", "No neurons (raw returns)": "#1F4FB8", "Shuffled neurons": "#7A869E"}
for name, c in curves.items():
    a1.plot([v - 1 for v in c], label=name, color=colors[name], linewidth=2)
bh_curve = [closes[i] / closes[start_day] - 1 for i in range(start_day, end_day + 1)]
a1.plot(bh_curve, label="Buy and hold", color="#E8B21C", linewidth=1.5, linestyle="--")
a1.set_title("Test period: growth of each trader")
a1.set_xlabel("Trading day")
a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a1.legend(fontsize=9)

a2.hist(random_returns, bins=40, color="#CBD3E1")
for name, s in results.items():
    a2.axvline(s["total_return"], color=colors[name], linewidth=2.5, label=name)
a2.set_title("Against 1,000 random traders")
a2.set_xlabel("Total return")
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
a2.legend(fontsize=9)
plt.tight_layout()
png = path.replace(".npz", ".png")
plt.savefig(png, dpi=150)
print(f"\nChart saved to {png}")
plt.show()
