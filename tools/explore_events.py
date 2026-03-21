#!/usr/bin/env python3
"""
explore_events.py — Event store data exploration.
Run:  python3 explore_events.py ~/polymarket-logger/historical_data/events
"""

import sys, os, glob
from datetime import datetime, timezone
from collections import defaultdict, Counter
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


def us_to_dt(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc)


def load_table(base_dir, table_name):
    """Load from events/{table_name}/*.parquet."""
    table_dir = os.path.join(base_dir, table_name)
    files = sorted(glob.glob(os.path.join(table_dir, "*.parquet")))
    if not files:
        print(f"  [WARN] No files in {table_dir}")
        return None
    print(f"  Loading {table_name}: {len(files)} file(s)")
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def percentiles(arr, pcts=[1, 5, 25, 50, 75, 95, 99]):
    vals = np.array(arr, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return {}
    return {f"p{p}": float(np.percentile(vals, p)) for p in pcts}


def volume_summary(table, name, time_col='ts_us'):
    print(f"\n{'='*60}")
    print(f"TABLE: {name}")
    print(f"{'='*60}")
    n = table.num_rows
    print(f"  Rows:    {n:,}")
    print(f"  Columns: {table.column_names}")
    print(f"  Schema:")
    for field in table.schema:
        print(f"    {field.name}: {field.type}")
    ts = table.column(time_col).to_pylist()
    ts_valid = [t for t in ts if t and t > 1e15 and t < 2e16]
    if not ts_valid:
        print("  [WARN] No valid timestamps!")
        return
    ts_min, ts_max = min(ts_valid), max(ts_valid)
    span_sec = (ts_max - ts_min) / 1e6
    span_min = span_sec / 60
    print(f"  Time: {us_to_dt(ts_min).isoformat()} → {us_to_dt(ts_max).isoformat()}")
    print(f"  Span: {span_min:.1f} min ({span_sec:.0f}s)")
    print(f"  Rate: {n / max(span_min, 0.001):.1f} rows/min")
    ts_arr = np.array(ts_valid, dtype=np.int64)
    minute_buckets = (ts_arr - ts_min) // 60_000_000
    unique_mins, counts = np.unique(minute_buckets, return_counts=True)
    if len(counts) > 0:
        print(f"  Per-min: min={counts.min()} med={int(np.median(counts))} "
              f"max={counts.max()} std={counts.std():.1f}")
        expected = set(range(int(unique_mins.min()), int(unique_mins.max()) + 1))
        gaps = sorted(expected - set(unique_mins.tolist()))
        if gaps:
            print(f"  ⚠ {len(gaps)} minute(s) with 0 events")
        else:
            print(f"  ✓ No coverage gaps ({len(expected)} minutes)")


def latency_analysis(table, name):
    if 'recv_us' not in table.column_names:
        return
    ts = np.array(table.column('ts_us').to_pylist(), dtype=np.float64)
    recv = np.array(table.column('recv_us').to_pylist(), dtype=np.float64)
    lat_ms = (recv - ts) / 1000.0
    mask = (lat_ms > -1000) & (lat_ms < 60000)
    lat_ms = lat_ms[mask]
    print(f"\n  Latency ({name}): {len(lat_ms):,} samples")
    for k, v in percentiles(lat_ms).items():
        print(f"    {k}: {v:.2f} ms")
    neg = int(np.sum(lat_ms < 0))
    if neg: print(f"    ⚠ {neg} negative")


def book_depth_analysis(table, name):
    print(f"\n  Book depth ({name}):")
    bp = table.column('bid_prices').to_pylist()
    ap = table.column('ask_prices').to_pylist()
    bd = [len(b) if b else 0 for b in bp]
    ad = [len(a) if a else 0 for a in ap]
    print(f"    Bid levels: min={min(bd)} med={int(np.median(bd))} max={max(bd)}")
    print(f"    Ask levels: min={min(ad)} med={int(np.median(ad))} max={max(ad)}")
    empty_b = sum(1 for d in bd if d == 0)
    empty_a = sum(1 for d in ad if d == 0)
    if empty_b or empty_a:
        print(f"    ⚠ Empty: {empty_b} bid, {empty_a} ask")
    crossed = sum(1 for b, a in zip(bp, ap) if b and a and len(b) > 0 and len(a) > 0 and b[0] >= a[0])
    print(f"    Crossed: {crossed} / {len(bp)}" + (" ✓" if crossed == 0 else " ⚠"))


def trade_analysis(table, name, sym_col=None, price_col='price', size_col=None):
    print(f"\n  Trades ({name}):")
    prices = np.array(table.column(price_col).to_pylist(), dtype=np.float64)
    print(f"    Price: {prices.min():.6f} → {prices.max():.6f}")
    if size_col and size_col in table.column_names:
        sizes = np.array(table.column(size_col).to_pylist(), dtype=np.float64)
        print(f"    Size: {sizes.min():.4f} → {sizes.max():.4f}  med={np.median(sizes):.4f}")
    if sym_col and sym_col in table.column_names:
        syms = table.column(sym_col).to_pylist()
        print(f"    Symbols: {dict(Counter(syms))}")


def bn_trade_id_gaps(table):
    print(f"\n  Binance trade_id continuity:")
    syms = table.column('symbol').to_pylist()
    tids = table.column('trade_id').to_pylist()
    by_sym = defaultdict(list)
    for s, t in zip(syms, tids):
        by_sym[s].append(t)
    for sym, ids in sorted(by_sym.items()):
        ids_s = sorted(ids)
        gaps = sum(ids_s[i] - ids_s[i-1] - 1 for i in range(1, len(ids_s)) if ids_s[i] - ids_s[i-1] > 1)
        status = "✓" if gaps == 0 else "⚠"
        print(f"    {sym}: {len(ids):,} trades, {gaps} missing IDs {status}")


def markets_analysis(table):
    print(f"\n  Markets:")
    n = table.num_rows
    coins = table.column('coin').to_pylist()
    intervals = table.column('interval_min').to_pylist()
    outcomes = table.column('outcome').to_pylist()
    print(f"    {n} records")
    print(f"    Coins: {dict(Counter(coins))}")
    print(f"    Intervals: {dict(Counter(intervals))}")
    print(f"    Outcomes: {dict(Counter(outcomes))} (1=UP 0=DOWN -1=unknown)")
    if 'target_price' in table.column_names and 'close_price' in table.column_names:
        targets = table.column('target_price').to_pylist()
        closes = table.column('close_price').to_pylist()
        mismatches = 0
        for i in range(n):
            if outcomes[i] == -1: continue
            expected = 1 if closes[i] >= targets[i] else 0
            if expected != outcomes[i]: mismatches += 1
        print(f"    Outcome consistency: {mismatches} mismatches" + (" ✓" if mismatches == 0 else " ⚠"))


def main(base_dir):
    print(f"Event Store Exploration")
    print(f"Base dir: {base_dir}")
    print(f"{'='*60}")

    all_files = sorted(glob.glob(os.path.join(base_dir, "**/*.parquet"), recursive=True))
    print(f"\nFound {len(all_files)} parquet files:")
    for f in all_files[:20]:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.relpath(f, base_dir)} ({size_mb:.2f} MB)")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more")

    t = load_table(base_dir, 'poly_markets')
    if t:
        volume_summary(t, 'poly_markets')
        markets_analysis(t)

    t = load_table(base_dir, 'poly_books')
    if t:
        volume_summary(t, 'poly_books')
        book_depth_analysis(t, 'poly_books')
        if 'event_type' in t.column_names:
            print(f"    Event types: {dict(Counter(t.column('event_type').to_pylist()))}")
        if 'best_bid' in t.column_names:
            bb = np.array(t.column('best_bid').to_pylist(), dtype=np.float64)
            ba = np.array(t.column('best_ask').to_pylist(), dtype=np.float64)
            valid = (bb > 0) & (ba > 0)
            spread = ba[valid] - bb[valid]
            print(f"    Spread: {percentiles(spread, [25,50,75,95])}")

    t = load_table(base_dir, 'poly_trades')
    if t:
        volume_summary(t, 'poly_trades')
        latency_analysis(t, 'poly_trades')
        trade_analysis(t, 'poly_trades', sym_col='token_id', size_col='size')

    t = load_table(base_dir, 'bn_books')
    if t:
        volume_summary(t, 'bn_books')
        book_depth_analysis(t, 'bn_books')

    t = load_table(base_dir, 'bn_trades')
    if t:
        volume_summary(t, 'bn_trades')
        latency_analysis(t, 'bn_trades')
        trade_analysis(t, 'bn_trades', sym_col='symbol', size_col='qty')
        bn_trade_id_gaps(t)

    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")


if __name__ == '__main__':
    base_dir = sys.argv[1] if len(sys.argv) > 1 else 'historical_data/events'
    main(base_dir)
