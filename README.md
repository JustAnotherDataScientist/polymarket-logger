# PM·BN Watcher

Live logger for **Polymarket** prediction markets and **Binance** spot data,
with event-sourced storage for precise order-book-level backtesting.

## Architecture

```
Binance WS ──┐                    ┌── bn_trades/     (every trade)
              ├── event_recording ─┤── bn_books/      (depth20 @100ms)
Polymarket WS┘                    ├── poly_books/    (every book update)
                                   ├── poly_trades/   (every confirmed trade)
                                   └── poly_markets/  (market metadata)
                                        │
                                        ▼
                                   book_replay.py
                                   (backtest engine)
```

Two storage paths run in parallel:

- **Event store** (`historical_data/events/`) — raw trades and book snapshots
  in native Arrow columnar format (zstd compressed). The backtest engine replays
  these to reconstruct the order book at any point in time with exact slippage.

- **Derived snapshots** (`historical_data/live/`) — 101-column Parquet
  with pre-computed features (z-scores, rolling volumes, book metrics,
  cross-asset signals). Written at 4Hz. Used by the dashboard and ML pipelines.

## Quick start

```bash
# On your VM
git clone <repo> ~/polymarket-logger
cd ~/polymarket-logger
bash deploy/setup.sh
```

The setup script creates a venv, installs deps, configures systemd + cron,
and starts the service.

## Manual run (development)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Project structure

```
├── main.py                 # Entry point
├── config.py               # Constants, market definitions, WS URLs
├── ws_clients.py           # Polymarket + Binance WebSocket clients
├── logger_core.py          # Main loop, market tracking, feature computation
├── event_store.py          # Event-sourced storage (5 Parquet tables)
├── event_recording.py      # Hooks WS clients → EventStore
├── storage.py              # Derived snapshot storage (101-col Parquet)
├── csv_schema.py           # Schema for derived snapshots
├── book_replay.py          # Backtest engine (book replay + execution sim)
├── api.py                  # REST clients (Gamma, CLOB, Binance)
├── utils.py                # Shared utilities (vol, z-score, probability)
├── logger_setup.py         # Python logging configuration
├── dashboard.py            # Live WebSocket dashboard
├── requirements.txt
│
├── tests/                  # Data quality + engine validation
│   ├── run_all.py          # Run all tests in sequence
│   ├── test_events.py      # Event store: schemas, timestamps, books, coverage
│   ├── test_backtest.py    # BookReplayEngine: fills, slippage, settlement
│   ├── test_crossval.py    # Events ↔ live snapshot agreement
│   └── test_live.py        # Live snapshots: logical consistency
│
├── tools/                  # Operations + analysis
│   ├── health_check.py     # Cron: service, disk, RAM, WAL monitoring
│   ├── backfill_outcomes.py# Cron: fetch outcomes from Polymarket API
│   ├── wal_watchdog.py     # Cron: restart if WAL stale
│   ├── explore_events.py   # Manual: data exploration + statistics
│   ├── inspect_live.py     # Manual: inspect live snapshot columns
│   ├── validate.py         # Manual: synthetic schema validation
│   ├── analyze_lead_lag.py # Manual: Binance → Polymarket lead-lag
│   └── plot_lead_lag.py    # Manual: lead-lag visualization
│
└── deploy/
    ├── setup.sh            # First-time VM setup (venv, systemd, cron)
    └── deploy.sh           # Git pull + restart
```

## Event store schema

### poly_books
Every Polymarket order book update (full book state after each event).

| Column | Type | Description |
|--------|------|-------------|
| ts_us | int64 | Receive timestamp (microseconds) |
| token_id | string | UP or DOWN token |
| event_type | uint8 | 0=price_change, 1=book_snapshot |
| bid_prices | list\<float32\> | Bid prices (descending) |
| bid_sizes | list\<float32\> | Bid sizes (USDC) |
| ask_prices | list\<float32\> | Ask prices (ascending) |
| ask_sizes | list\<float32\> | Ask sizes (USDC) |
| best_bid | float32 | Top of bid book |
| best_ask | float32 | Top of ask book |

### poly_trades
Every confirmed Polymarket trade.

| Column | Type | Description |
|--------|------|-------------|
| ts_us | int64 | Trade timestamp (from event) |
| recv_us | int64 | Local receive timestamp |
| token_id | string | Token traded |
| price | float32 | Trade price [0, 1] |
| size | float32 | Trade size (USDC) |
| tx_hash | string | Transaction hash |

### bn_trades
Every Binance trade.

| Column | Type | Description |
|--------|------|-------------|
| ts_us | int64 | Exchange timestamp |
| recv_us | int64 | Local receive timestamp |
| symbol | dict\<int8, string\> | e.g. BTCUSDT |
| price | float64 | Trade price |
| qty | float64 | Trade quantity |
| is_buy | bool | Buyer is taker |
| trade_id | int64 | Binance trade ID |

### bn_books
Binance depth snapshots (@depth20@100ms).

| Column | Type | Description |
|--------|------|-------------|
| ts_us | int64 | Receive timestamp |
| symbol | dict\<int8, string\> | e.g. BTCUSDT |
| bid_prices | list\<float64\> | 20 bid levels |
| bid_sizes | list\<float64\> | 20 bid sizes |
| ask_prices | list\<float64\> | 20 ask levels |
| ask_sizes | list\<float64\> | 20 ask sizes |

### poly_markets
Market metadata (logged on market switch + window close).

| Column | Type | Description |
|--------|------|-------------|
| ts_us | int64 | Log timestamp |
| slug | string | Market slug |
| coin | string | btc, eth, sol, xrp |
| interval_min | int16 | 5 or 15 |
| up_token_id | string | UP token address |
| down_token_id | string | DOWN token address |
| window_start_us | int64 | Window start (on switch row) |
| window_end_us | int64 | Window end (on switch row) |
| target_price | float64 | Target price |
| close_price | float64 | Close price (on close row) |
| outcome | int8 | 1=UP, 0=DOWN, -1=unknown |

## Backtesting

The backtest engine replays recorded events to reconstruct the exact
order book state at any point in time:

```python
from book_replay import BookReplayEngine

engine = BookReplayEngine("historical_data/events")
engine.load("2026-03-20", token_id="0xabc123...")

for snap in engine.iter_snapshots(step_us=250_000):  # 250ms ticks
    book = snap.book
    print(f"mid={book.mid:.4f}, spread={book.spread_bps:.0f}bps")

    # Taker order with exact slippage
    fill = snap.simulate_taker_buy(size_usdc=100.0)
    print(f"filled {fill.filled_size} @ avg {fill.avg_price:.4f}")
```

Window-aware backtesting with strategy functions:

```python
from book_replay import WindowBacktest, StrategyContext, Action

def my_strategy(ctx: StrategyContext):
    if ctx.position is None and ctx.book.mid < 0.45:
        return Action("buy_up", size_usdc=50)
    return None

bt = WindowBacktest("historical_data/events")
result = bt.run("2026-03-20", coin="btc", interval_min=5,
                strategy=my_strategy)
print(f"PnL: ${result.total_pnl:.2f} over {result.n_windows} windows")
```

## Testing

```bash
# Run all tests
cd ~/polymarket-logger
python tests/run_all.py historical_data

# Individual tests
python tests/test_events.py historical_data/events          # event store quality
python tests/test_backtest.py historical_data/events         # replay engine
python tests/test_crossval.py historical_data                # events vs snapshots
python tests/test_live.py --dir historical_data/live         # live snapshot logic

# Data exploration
python tools/explore_events.py historical_data/events
```

### test_events.py (30 checks)
Schema validation, timestamp sanity (16-digit μs), latency bounds,
book integrity (no crossed/unsorted books), per-minute coverage,
Binance trade ID continuity, outcome consistency, Polymarket price
ranges, best_bid/ask coherence with book top.

### test_backtest.py (25 checks)
ExecutionSimulator unit tests (single-level fill, multi-level slippage,
partial fills, empty book), BookReplayEngine load + iterate,
WindowBacktest with buy_and_hold across BTC/ETH 5m/15m, settlement
price verification, PnL math.

### test_crossval.py
Reconstructs UP-token book state from raw events at sampled timestamps,
compares against live snapshot `up_bid`/`up_ask`. Reports exact match
rate and max divergence.

## Markets tracked

| Coin | Binance | Intervals | Polymarket format |
|------|---------|-----------|-------------------|
| BTC | BTCUSDT | 5m, 15m | btc-updown-{interval}-{timestamp} |
| ETH | ETHUSDT | 5m, 15m | eth-updown-{interval}-{timestamp} |
| SOL | SOLUSDT | 5m, 15m | sol-updown-{interval}-{timestamp} |
| XRP | XRPUSDT | 5m, 15m | xrp-updown-{interval}-{timestamp} |

## Operations

```bash
# Live logs
sudo journalctl -u pm-watcher -f

# Restart / stop
sudo systemctl restart pm-watcher
sudo systemctl stop pm-watcher

# Health check
python tools/health_check.py

# Deploy update
bash deploy/deploy.sh

# Data size
du -sh historical_data/events/* historical_data/live/
```

## Data volumes

- **Event store**: ~800 MB/day compressed (zstd)
- **Derived snapshots**: ~200 MB/day compressed (snappy)
- **RAM usage**: ~50–80 MB steady state

## Cron jobs (installed by setup.sh)

| Schedule | Script | Purpose |
|----------|--------|---------|
| */15 * * * * | tools/health_check.py | Service, disk, RAM, WAL |
| 0 * * * * | tools/backfill_outcomes.py | Fetch resolved outcomes |
| 5 0 * * * | tools/backfill_outcomes.py | Daily full backfill |
| 0 2 * * * | log rotation | Truncate logs > 50MB |
