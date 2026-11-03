"""Live trading terminal: watch the neurons trade day by day in your browser.

    python serve_trading.py

Opens http://127.0.0.1:8770. Pick a dataset, condition and pace, press Start.
Press Ctrl + C in the terminal to stop the server.

Each simulated trading day the neurons are shown the last 5 daily returns and
their spikes are read out. When not holding, BUY electrodes firing more than
WAIT electrodes means buy. Exits follow the fixed rules in backtest.py.
"""
import glob
import json
import os
import random
import threading
import time
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cl
import backtest as bt
import neural_trading as nt

PORT = 8770
HERE = Path(__file__).parent
PACES = {"fast": 0.0, "normal": 0.35, "slow": 1.0}   # extra seconds per day, for presenting

lock = threading.Lock()
run = {"id": 0, "events": [], "running": False, "stop": False}


def emit(event):
    with lock:
        run["events"].append(event)


def channel_roles():
    roles = {}
    for c in nt.UP_CHANNELS:
        roles[c] = "up"
    for c in nt.DOWN_CHANNELS:
        roles[c] = "down"
    for c in nt.BUY_GROUP:
        roles[c] = "buy"
    for c in nt.WAIT_GROUP:
        roles[c] = "wait"
    return roles


def trading_session(data_path, condition, days, pace):
    dates, closes = bt.load_prices(data_path)
    rets = bt.daily_returns(closes)
    start = max(nt.LOOKBACK, len(closes) - days)
    end = len(closes) - 1
    emit({"type": "start", "symbol": Path(data_path).stem.upper(), "condition": condition,
          "start_date": dates[start], "end_date": dates[end], "days": end - start,
          "simulator": cl.is_simulator(), "roles": channel_roles(),
          "history": [{"date": dates[i], "close": closes[i]} for i in range(max(0, start - 20), start)]})

    trades, holding, decisions, equity, base = [], None, 0, 1.0, 1.0
    os.makedirs("results_trading", exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    log_path = f"results_trading/live_{condition}_{Path(data_path).stem}_{stamp}.json"

    with cl.open() as neurons:
        for i in range(start, end):
            if run["stop"]:
                emit({"type": "stopped"})
                break
            t0 = time.time()
            event = {"type": "day", "date": dates[i], "close": closes[i], "ret": rets[i]}

            # Close the position if today hits an exit rule
            if holding is not None and i == holding[1]:
                entry, exit_, r = holding
                trades.append(holding)
                equity = base * (1 + r)
                event["closed"] = {"entry_date": dates[entry], "exit_date": dates[exit_],
                                   "entry": closes[entry], "exit": closes[exit_], "ret": r}
                if condition == "feedback":
                    nt.give_feedback(neurons, r > 0)
                elif condition == "random_feedback":
                    nt.give_feedback(neurons, random.random() < 0.5)
                holding = None

            counts = nt.present(neurons, rets[i - nt.LOOKBACK + 1:i + 1])
            buy, wait = int(counts[nt.BUY_GROUP].sum()), int(counts[nt.WAIT_GROUP].sum())
            event.update(counts=counts.tolist(), buy=buy, wait=wait)

            if holding is None and "closed" not in event:
                decisions += 1
                if buy > wait:
                    j, r = bt.trade_outcome(closes, i)
                    holding = (i, j, r)
                    base = equity
                    event["decision"] = "buy"
                else:
                    event["decision"] = "wait"
            else:
                event["decision"] = "holding" if holding else "exit"

            if holding is not None:
                equity = base * closes[i] / closes[holding[0]]
            event["equity"] = equity - 1
            event["buy_and_hold"] = closes[i] / closes[start] - 1
            event["position"] = None if holding is None else {"entry_date": dates[holding[0]], "entry": closes[holding[0]]}
            emit(event)

            spare = PACES.get(pace, 0.35) - (time.time() - t0)
            if spare > 0:
                nt.rest(neurons, spare)

    s = bt.summarise(trades)
    rate = len(trades) / max(1, decisions)
    randoms = bt.random_baseline(closes, start, end, rate) if trades else []
    summary = {
        "type": "done",
        "trades": s["trades"], "total_return": s["total_return"], "win_rate": s["win_rate"],
        "max_drawdown": s["max_drawdown"], "buy_and_hold": bt.buy_and_hold(closes, start, end - 1),
        "random_median": randoms[len(randoms) // 2] if randoms else None,
        "beats_random": bt.percentile_of(s["total_return"], randoms) if randoms else None,
        "log": log_path,
    }
    emit(summary)
    with lock:
        events = list(run["events"])
    with open(log_path, "w") as f:
        json.dump(events, f)
    print(f"Session saved to {log_path}")


def runner(*args):
    try:
        trading_session(*args)
    except Exception as err:
        print("Error:", repr(err))
        emit({"type": "error", "message": f"{type(err).__name__}: {err}"})
    finally:
        run["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "terminal.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/datasets":
            self.send_json(sorted(glob.glob("data/*.csv")))
        elif self.path.startswith("/events"):
            since = int(self.path.split("since=")[-1]) if "since=" in self.path else 0
            with lock:
                self.send_json({"run": run["id"], "running": run["running"], "events": run["events"][since:]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/start":
            if run["running"]:
                self.send_json({"error": "A session is already running. Stop it first."}, 400)
                return
            data = payload.get("data", "data/SPY.csv")
            if not os.path.exists(data):
                self.send_json({"error": f"No data file at {data}. Run fetch_data.py first."}, 400)
                return
            with lock:
                run.update(id=run["id"] + 1, events=[], running=True, stop=False)
            args = (data, payload.get("condition", "feedback"), int(payload.get("days", 250)), payload.get("pace", "normal"))
            threading.Thread(target=runner, args=args, daemon=True).start()
            self.send_json({"ok": True, "run": run["id"]})
        elif self.path == "/stop":
            run["stop"] = True
            self.send_json({"ok": True})
        else:
            self.send_error(404)


def main():
    server = None
    for port in range(PORT, PORT + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port found.")
    url = f"http://127.0.0.1:{port}"
    print(f"Trading terminal at {url}  (Ctrl + C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        run["stop"] = True


if __name__ == "__main__":
    main()
