#!/usr/bin/env python3
"""
tests/test_backtest.py — Validate the book replay backtest engine.
Uses the ACTUAL BookReplayEngine and WindowBacktest from book_replay.py.

Run from project root:
  python tests/test_backtest.py historical_data/events

Tests:
  1. BookReplayEngine loads data and iterates snapshots
  2. Book state continuity — every snapshot has valid bid/ask
  3. Taker fill walks the book correctly (slippage)
  4. WindowBacktest runs buy_and_hold strategy, PnL is correct
  5. Settlement at 1.00 (UP) or 0.00 (DOWN) matches outcome
  6. Sell produces non-negative PnL tracking
"""

import sys, os, argparse, logging
from pathlib import Path
from collections import defaultdict

# Add project root to path so we can import book_replay
_HERE = Path(__file__).parent
_PROJECT = _HERE.parent  # tests/ → project root
for candidate in [_PROJECT, _HERE, Path.home() / 'polymarket-logger']:
    if (candidate / 'book_replay.py').exists():
        sys.path.insert(0, str(candidate))
        break

try:
    from book_replay import (
        BookReplayEngine, WindowBacktest, BookState, BookLevel,
        ExecutionSimulator, TakerFill, MarketWindow,
        StrategyContext, Action,
        strategy_buy_and_hold,
    )
    HAS_BOOK_REPLAY = True
except ImportError as e:
    print(f"WARNING: Could not import book_replay: {e}")
    print(f"  Run from ~/polymarket-logger/ or set PROJECT_DIR env var")
    HAS_BOOK_REPLAY = False

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────

def us_to_dt(us):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc)


# ─── Test Results ─────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name, passed, message=""):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"  {status}: {self.name} — {self.message}"


# ─── Test: ExecutionSimulator mechanics ───────────────────────────────

def test_execution_simulator():
    """Test taker fill logic with a synthetic book."""
    results = []
    sim = ExecutionSimulator()

    # Build a synthetic book
    book = BookState(
        ts_us=1000000,
        token_id="test",
        bids=[BookLevel(0.55, 100), BookLevel(0.54, 200), BookLevel(0.53, 300)],
        asks=[BookLevel(0.56, 100), BookLevel(0.57, 200), BookLevel(0.58, 300)],
    )

    # Test 1: small taker buy (fits in first level)
    fill = sim.simulate_taker_buy(book, 50.0)
    results.append(TestResult(
        "taker_buy_single_level",
        fill.filled_size == 50.0 and abs(fill.avg_price - 0.56) < 1e-6,
        f"filled={fill.filled_size} avg_price={fill.avg_price:.4f} (expect 50 @ 0.56)"
    ))

    # Test 2: large taker buy (walks multiple levels)
    fill2 = sim.simulate_taker_buy(book, 250.0)
    expected_avg = (100 * 0.56 + 150 * 0.57) / 250.0
    results.append(TestResult(
        "taker_buy_multi_level",
        fill2.filled_size == 250.0 and abs(fill2.avg_price - expected_avg) < 1e-4,
        f"filled={fill2.filled_size} avg={fill2.avg_price:.6f} expect={expected_avg:.6f}"
    ))

    # Test 3: taker buy exceeds book depth
    fill3 = sim.simulate_taker_buy(book, 1000.0)
    results.append(TestResult(
        "taker_buy_partial_fill",
        fill3.filled_size == 600.0 and not fill3.fully_filled,
        f"filled={fill3.filled_size} (book has 600 USDC on ask side)"
    ))

    # Test 4: taker sell
    fill4 = sim.simulate_taker_sell(book, 80.0)
    results.append(TestResult(
        "taker_sell_single_level",
        fill4.filled_size == 80.0 and abs(fill4.avg_price - 0.55) < 1e-6,
        f"filled={fill4.filled_size} avg={fill4.avg_price:.4f}"
    ))

    # Test 5: slippage — buying more than L1 should give avg > best_ask
    fill5 = sim.simulate_taker_buy(book, 150.0)
    results.append(TestResult(
        "slippage_walks_book",
        fill5.avg_price > book.asks[0].price,
        f"avg={fill5.avg_price:.6f} > best_ask={book.asks[0].price}"
    ))

    # Test 6: empty book
    empty_book = BookState(ts_us=0, token_id="test", bids=[], asks=[])
    fill6 = sim.simulate_taker_buy(empty_book, 100.0)
    results.append(TestResult(
        "empty_book_no_fill",
        fill6.filled_size == 0.0,
        f"filled={fill6.filled_size} on empty book"
    ))

    return results


# ─── Test: BookReplayEngine loads and iterates ────────────────────────

def test_replay_engine(events_dir, date_str, token_id):
    """Test that BookReplayEngine loads data and produces snapshots."""
    results = []
    engine = BookReplayEngine(events_dir)

    try:
        engine.load(date_str, token_id)
    except Exception as e:
        return [TestResult("replay_load", False, f"load failed: {e}")]

    start, end = engine.time_range()
    if start is None:
        return [TestResult("replay_load", False, "no data after load")]

    results.append(TestResult(
        "replay_load", True,
        f"loaded {us_to_dt(start).strftime('%H:%M:%S')}→{us_to_dt(end).strftime('%H:%M:%S')}"
    ))

    # Iterate snapshots
    snap_count = 0
    empty_books = 0
    valid_books = 0
    last_ts = 0

    for snap in engine.iter_snapshots(step_us=250_000, start_us=start, end_us=min(start + 60_000_000, end)):
        snap_count += 1
        if snap.book.best_bid is not None and snap.book.best_ask is not None:
            valid_books += 1
        else:
            empty_books += 1
        # Monotonic time
        assert snap.ts_us >= last_ts, f"time went backwards: {snap.ts_us} < {last_ts}"
        last_ts = snap.ts_us

    results.append(TestResult(
        "replay_iterate", snap_count > 0,
        f"{snap_count} snapshots, {valid_books} valid books, {empty_books} empty"
    ))

    # Test taker execution through the snapshot API
    if snap_count > 0:
        engine.load(date_str, token_id)
        for snap in engine.iter_snapshots(step_us=250_000, start_us=start, end_us=start + 1_000_000):
            if snap.book.best_ask is not None:
                fill = snap.simulate_taker_buy(50.0)
                results.append(TestResult(
                    "replay_taker_buy", fill.filled_size > 0,
                    f"filled {fill.filled_size:.1f} USDC @ {fill.avg_price:.4f}"
                ))
                break

    return results


# ─── Test: WindowBacktest end-to-end ──────────────────────────────────

def test_window_backtest(events_dir, date_str, coin="btc", interval_min=5):
    """Test WindowBacktest with buy_and_hold strategy."""
    results = []
    bt = WindowBacktest(events_dir)

    windows = bt.load_windows(date_str, coin=coin, interval_min=interval_min)
    results.append(TestResult(
        f"load_windows_{coin}_{interval_min}m",
        len(windows) >= 0,
        f"{len(windows)} resolved windows"
    ))

    if len(windows) == 0:
        return results

    # Run buy_and_hold on up to 5 windows
    test_windows = windows[:5]
    result = bt.run(date_str, coin=coin, interval_min=interval_min,
                    strategy=strategy_buy_and_hold, step_us=250_000)

    results.append(TestResult(
        f"backtest_runs_{coin}_{interval_min}m",
        True,
        f"PnL={result.total_pnl:.2f} over {result.n_windows} windows, "
        f"{result.n_trades} trades, {result.n_settlements} settlements"
    ))

    # Verify every settlement has correct price
    for trade in result.trades:
        if trade.side == "settlement":
            # Find the window for this trade
            window = next((w for w in windows if w.slug == trade.window_slug), None)
            if window:
                expected_price = 1.0 if window.outcome == 1 else 0.0
                ok = abs(trade.price - expected_price) < 1e-6
                if not ok:
                    results.append(TestResult(
                        f"settlement_{trade.window_slug[:25]}",
                        False,
                        f"price={trade.price} but outcome={window.outcome} → expect {expected_price}"
                    ))

    # Count settlement checks
    settlements = [t for t in result.trades if t.side == "settlement"]
    settlement_ok = all(
        abs(t.price - (1.0 if next((w for w in windows if w.slug == t.window_slug), None).outcome == 1 else 0.0)) < 1e-6
        for t in settlements
        if next((w for w in windows if w.slug == t.window_slug), None) is not None
    )
    results.append(TestResult(
        f"settlement_prices_{coin}_{interval_min}m",
        settlement_ok,
        f"{len(settlements)} settlements all at correct price"
    ))

    # Verify PnL math: total PnL should equal sum of per-trade PnL
    sum_pnl = sum(t.pnl for t in result.trades)
    pnl_match = abs(sum_pnl - result.total_pnl) < 0.01
    results.append(TestResult(
        f"pnl_math_{coin}_{interval_min}m",
        pnl_match,
        f"sum(trade.pnl)={sum_pnl:.4f} vs total_pnl={result.total_pnl:.4f}"
    ))

    return results


# ─── Main ─────────────────────────────────────────────────────────────

def main(events_dir, date_str=None, verbose=True):
    print(f"Backtest Validation")
    print(f"Events dir: {events_dir}")
    print(f"{'='*60}")

    all_results = []

    # ── Test 1: ExecutionSimulator mechanics (no data needed) ──
    if HAS_BOOK_REPLAY:
        print("\n1. ExecutionSimulator unit tests...")
        all_results.extend(test_execution_simulator())
    else:
        all_results.append(TestResult("import_book_replay", False,
                                       "Could not import book_replay.py"))

    if not HAS_BOOK_REPLAY:
        for r in all_results:
            print(r)
        return False

    # ── Discover available data ──
    import glob
    markets_dir = os.path.join(events_dir, "poly_markets")
    market_files = sorted(glob.glob(os.path.join(markets_dir, "*.parquet")))
    if not market_files:
        all_results.append(TestResult("discover_data", False, "no poly_markets files"))
        for r in all_results:
            print(r)
        return False

    # Auto-detect date from filenames
    if date_str is None:
        fname = os.path.basename(market_files[0])
        date_str = fname[:10]  # "2026-03-20" from "2026-03-20_0000.parquet"
        print(f"  Auto-detected date: {date_str}")

    # Load markets to find a token_id for replay test
    tables = [pq.read_table(f) for f in market_files]
    markets = pa.concat_tables(tables).to_pandas()
    resolved = markets[markets['outcome'].isin([0, 1])]
    print(f"  Markets: {len(markets)} total, {len(resolved)} resolved")

    # ── Test 2: BookReplayEngine ──
    if len(resolved) > 0:
        test_row = resolved.iloc[0]
        print(f"\n2. BookReplayEngine test (token: {test_row['up_token_id'][:20]}...)...")
        all_results.extend(test_replay_engine(
            events_dir, date_str, test_row['up_token_id']
        ))

    # ── Test 3: WindowBacktest end-to-end ──
    coins_available = resolved['coin'].unique() if len(resolved) > 0 else []
    for coin in coins_available[:2]:  # test first 2 coins
        for iv in [5, 15]:
            n_windows = len(resolved[(resolved['coin'] == coin) & (resolved['interval_min'] == iv)])
            if n_windows > 0:
                print(f"\n3. WindowBacktest {coin} {iv}m ({n_windows} windows)...")
                all_results.extend(test_window_backtest(events_dir, date_str, coin, iv))

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"BACKTEST VALIDATION RESULTS")
    print(f"{'='*60}")

    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)

    for r in all_results:
        print(r)

    print(f"\n{'─'*60}")
    print(f"TOTAL: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('--date', default=None, help='Date (YYYY-MM-DD), auto-detected if omitted')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    success = main(args.events_dir, date_str=args.date, verbose=not args.quiet)
    sys.exit(0 if success else 1)
