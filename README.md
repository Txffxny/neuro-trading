# Neuron trading experiment (backup project)

Living neurons decide **when to buy** a stock. Fixed rules handle every exit,
so any difference from random traders comes from the neurons' entry timing.

Exit rules (same for every trader): sell at the first daily close that is +3%
or −2% from the entry price, otherwise after 10 trading days.

## Before the event (needs internet)
    pip install yfinance
    python fetch_data.py                # SPY, last 3 years -> data/SPY.csv
    python fetch_data.py --synthetic    # test data with hidden trends
Commit the `data` folder so it's on your laptop even without venue wifi.

## Mode 1: reservoir (recommended)
The neurons only transform market data; the learning happens on your laptop,
so the neurons' short memory doesn't matter.

    python record_reservoir.py --data data/SPY.csv --days 500     # uses the CL1, ~4-5 min
    python analyse_reservoir.py results_trading/reservoir_SPY_<time>.npz

The analysis trains on the first 60% of days and trades the last 40%, then
compares: Neurons, No neurons (the same model on raw returns), Shuffled
neurons (spike counts matched to the wrong days), 1,000 random traders, and
buy-and-hold.

## Mode 2: closed loop with feedback
    python run_closed_loop.py --data data/SPY.csv --condition feedback
    python run_closed_loop.py --data data/SPY.csv --condition random_feedback
    python run_closed_loop.py --data data/SPY.csv --condition none
About 250 days per run, a few minutes each.

## Reading the results
"Beats random" is the share of 1,000 random traders (same trading rate, same
exit rules) that did worse. Above 95% is unlikely to be luck. A single run
can land above 95% by chance about 1 time in 20, so repeat runs before
claiming anything.

The key comparison is Neurons vs No neurons: if the neurons don't beat the
same model reading raw returns, the biology isn't adding anything yet.

## Sanity check
On `data/synthetic.csv` the No neurons model should beat nearly all random
traders (the trends are real). On the simulator, Neurons should look random.

## Live trading terminal (for the demo)
    python serve_trading.py
Opens http://127.0.0.1:8770. Choose the data, condition, number of days and
pace, then press Start. You'll see the price chart with buys and exits, the
electrode array firing, the buy-versus-wait signal, a trade blotter, and P&L
against buy-and-hold. At the end it reports how the neurons did against 1,000
random traders. Each session is saved to results_trading/live_*.json.
Normal pace runs about 250 days in 3-4 minutes; Fast is quicker.

## Limitations to mention
- No trading costs or slippage: every trade is assumed free and filled at the close.
- One instrument (SPY by default) and a short test window, so few trades.
- Long only: the strategy never bets on prices falling.
- The electrode grid on screen is in channel-number order, not physical position.
