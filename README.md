# PM·BN Watcher

Live logger for **Polymarket** prediction markets and **Binance** spot,
with event-sourced storage for precise order-book-level backtesting.

## Architecture

```
Binance WS ──┐                    ┌── bn_trades.parquet
              ├── event_recording ─┤── bn_books.parquet
Polymarket WS┘                    ├── poly_books.parquet
                                   ├── poly_trades.parquet
                                   └── poly_markets.parquet
                                        │
                                        ▼
                                   book_replay.py
                                   (backtest engine)
```

Two storage paths run in parallel:

- **Event store** (`historical_data/events/`) — raw trades and book snapshots
  in native Arrow columnar format. Used by the backtest engine to replay
  the order book at any point in time with exact slippage simulation.

- **Derived snapshots** (`historical_data/live/`) — 101-column Parquet
  with pre-computed features (z-scores, rolling volumes, etc.).
  Used by the live dashboard.

## Quick start

```bash
# On your VM
git clone <repo> ~/polymarket-logger
cd ~/polymarket-logger
bash deploy/setup.sh
```

That's it. The setup script creates a venv, installs deps, sets up systemd,
and starts the service.

## Manual run (development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Project structure

```
├── main.py              # Entry point
├── config.py            # All constants and market definitions
├── ws_clients.py        # Polymarket + Binance WebSocket clients
├── logger_core.py       # Main loop, market tracking, derived features
├── event_store.py       # Event-sourced Parquet storage (5 tables)
├── event_recording.py   # Hooks WS clients → EventStore
├── book_replay.py       # Backtest engine with order book replay
├── api.py               # REST API client (Gamma, CLOB, Binance)
├── storage.py           # Legacy derived-snapshot storage
├── csv_schema.py        # 101-column schema for derived snapshots
├── utils.py             # Shared utilities
├── logger_setup.py      # Python logging configuration
├── dashboard.py         # Live WebSocket dashboard
├── requirements.txt
├── deploy/
│   ├── setup.sh         # First-time VM setup
│   └── deploy.sh        # Git pull + restart
└── tools/
    ├── health_check.py
    ├── backfill_outcomes.py
    ├── validate.py
    ├── inspect_live.py
    ├── run_tests.py
    ├── analyze_lead_lag.py
    ├── plot_lead_lag.py
    └── wal_watchdog.py
```

## Backtesting

The backtest engine replays recorded events to reconstruct the exact
order book state at any point in time:

```python
from book_replay import BookReplayEngine

engine = BookReplayEngine("historical_data/events")
engine.load("2026-03-20", token_id="0xabc123...")

for snap in engine.iter_snapshots(step_us=250_000):  # 250ms
    book = snap.book

    # Taker order — exact fill with slippage
    fill = snap.simulate_taker_buy(size_usdc=100.0)
    print(f"avg price: {fill.avg_price}, levels: {fill.levels_consumed}")

    # Limit order — conservative last-in-queue model
    order = snap.post_limit_buy(price=0.52, size_usdc=200.0)

    # Check fills on each tick
    for filled in snap.check_fills():
        print(f"order {filled.order_id} filled at {filled.fill_price}")
```

## Operations

```bash
# Live logs
sudo journalctl -u pm-watcher -f

# Restart
sudo systemctl restart pm-watcher

# Stop
sudo systemctl stop pm-watcher

# Health check
cd ~/polymarket-logger && venv/bin/python tools/health_check.py

# Check event store size
find historical_data/events -name "*.parquet" -exec ls -lh {} \;

# Deploy update
bash deploy/deploy.sh
```

## Markets tracked

| Coin | Binance symbol | Intervals |
|------|---------------|-----------|
| BTC  | BTCUSDT       | 5m, 15m   |
| ETH  | ETHUSDT       | 5m, 15m   |
| SOL  | SOLUSDT       | 5m, 15m   |
| XRP  | XRPUSDT       | 5m, 15m   |

## Data volumes

- **Event store**: ~800 MB/day compressed (zstd)
- **Derived snapshots**: ~200 MB/day compressed (snappy)
- **RAM usage**: ~50–80 MB steady state
