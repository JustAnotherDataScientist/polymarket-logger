#!/usr/bin/env python3
"""
test_quality.py — Daily quality validation test suite.
Run:  pytest test_quality.py -v --tb=short
Or:   python3 test_quality.py ~/polymarket-logger/historical_data/events

Tests:
  1. Schema validation    — correct columns per table
  2. Timestamp sanity     — 16-digit μs, no absurd values
  3. Latency bounds       — recv_us - ts_us within tolerance
  4. Book integrity       — no crossed books, no inverted sort
  5. Coverage monitoring  — event rate, gap detection
  6. Binance trade ID     — no gaps per symbol
  7. Outcome consistency  — close vs target matches outcome
  8. Poly prices in [0,1]
  9. best_bid/ask matches book top-of-book
"""

import sys, os, glob, json, argparse
from datetime import datetime, timezone
from collections import defaultdict, Counter
import numpy as np

# ─── Thresholds ───────────────────────────────────────────────────────

THRESHOLDS = {
    # Poly trades: ts_us is on-chain time, recv_us is local. Latency can be
    # 10s+ for on-chain confirmation — this is normal, not a logger bug.
    'poly_trades_max_latency_ms': 30_000,
    # Binance: ts_us is exchange time (T field), recv_us is local.
    'bn_trades_max_latency_ms': 10_000,
    'max_negative_latency_ms': 100,     # clock skew tolerance
    'max_gap_minutes': 2,               # consecutive empty minutes
    'ts_min_us': 1_700_000_000_000_000, # ~2023-11
    'ts_max_us': 2_000_000_000_000_000, # ~2033
}

# Tables that should have continuous per-minute coverage
# poly_markets is event-driven (market switch + close) — NOT continuous
CONTINUOUS_TABLES = {'poly_books', 'poly_trades', 'bn_books', 'bn_trades'}

# ─── Data Loading ─────────────────────────────────────────────────────

import pyarrow.parquet as pq
import pyarrow as pa

def load_table(base_dir, table_name, date=None):
    """Load from events/{table_name}/*.parquet."""
    table_dir = os.path.join(base_dir, table_name)
    if date:
        pattern = os.path.join(table_dir, f"{date}*.parquet")
    else:
        pattern = os.path.join(table_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

# ─── Test Results ─────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name, passed, message="", details=None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"  {status}: {self.name} — {self.message}"

# ─── Test Functions ───────────────────────────────────────────────────

def test_schema(table, name, expected_cols):
    actual = set(table.column_names)
    missing = set(expected_cols) - actual
    extra = actual - set(expected_cols)
    passed = len(missing) == 0
    parts = []
    if missing: parts.append(f"missing: {missing}")
    if extra: parts.append(f"extra: {extra}")
    return TestResult(f"{name}/schema", passed,
                      "; ".join(parts) if parts else f"{len(actual)} columns OK")

def test_timestamps(table, name, ts_col='ts_us'):
    ts = np.array(table.column(ts_col).to_pylist(), dtype=np.int64)
    n = len(ts)
    issues = []
    zeros = int(np.sum(ts == 0))
    below = int(np.sum((ts > 0) & (ts < THRESHOLDS['ts_min_us'])))
    above = int(np.sum(ts > THRESHOLDS['ts_max_us']))
    if zeros: issues.append(f"{zeros} zeros")
    if below: issues.append(f"{below} below min")
    if above: issues.append(f"{above} above max")
    passed = len(issues) == 0
    msg = f"{n:,} timestamps OK" if passed else "; ".join(issues)
    diffs = np.diff(ts)
    ooo = int(np.sum(diffs < 0))
    if ooo > 0:
        msg += f"; {ooo} out-of-order ({ooo/n*100:.2f}%)"
    return TestResult(f"{name}/timestamps", passed, msg)

def test_latency(table, name, ts_col='ts_us', recv_col='recv_us'):
    if recv_col not in table.column_names:
        return TestResult(f"{name}/latency", True, "no recv_us — skipped")
    ts = np.array(table.column(ts_col).to_pylist(), dtype=np.float64)
    recv = np.array(table.column(recv_col).to_pylist(), dtype=np.float64)
    lat_ms = (recv - ts) / 1000.0
    median = float(np.median(lat_ms))
    p99 = float(np.percentile(lat_ms, 99))
    # Use per-table threshold
    max_lat = THRESHOLDS.get(f'{name}_max_latency_ms',
                              THRESHOLDS.get('bn_trades_max_latency_ms', 10_000))
    absurd_high = int(np.sum(lat_ms > max_lat))
    absurd_neg = int(np.sum(lat_ms < -THRESHOLDS['max_negative_latency_ms']))
    issues = []
    if absurd_high: issues.append(f"{absurd_high} > {max_lat}ms")
    if absurd_neg: issues.append(f"{absurd_neg} < -{THRESHOLDS['max_negative_latency_ms']}ms")
    passed = len(issues) == 0
    msg = f"median={median:.1f}ms p99={p99:.1f}ms"
    if issues: msg += "; " + "; ".join(issues)
    return TestResult(f"{name}/latency", passed, msg,
                      {"median_ms": median, "p99_ms": p99})

def test_book_integrity(table, name):
    bid_prices = table.column('bid_prices').to_pylist()
    ask_prices = table.column('ask_prices').to_pylist()
    n = len(bid_prices)
    crossed = unsorted_bids = unsorted_asks = empty_books = 0
    for i in range(n):
        bp, ap = bid_prices[i], ask_prices[i]
        if not bp or not ap or len(bp) == 0 or len(ap) == 0:
            empty_books += 1
            continue
        if bp[0] >= ap[0]: crossed += 1
        if len(bp) > 1 and any(bp[j] < bp[j+1] for j in range(len(bp)-1)):
            unsorted_bids += 1
        if len(ap) > 1 and any(ap[j] > ap[j+1] for j in range(len(ap)-1)):
            unsorted_asks += 1
    issues = []
    if crossed: issues.append(f"{crossed} crossed")
    if unsorted_bids: issues.append(f"{unsorted_bids} unsorted bids")
    if unsorted_asks: issues.append(f"{unsorted_asks} unsorted asks")
    passed = crossed == 0 and unsorted_bids == 0 and unsorted_asks == 0
    msg = f"{n:,} books OK" if passed else "; ".join(issues)
    if empty_books:
        msg += f" ({empty_books} empty books)"
    return TestResult(f"{name}/book_integrity", passed, msg)

def test_coverage(table, name, ts_col='ts_us'):
    if name not in CONTINUOUS_TABLES:
        return TestResult(f"{name}/coverage", True,
                          f"skipped (event-driven table, not continuous)")
    ts = np.array(table.column(ts_col).to_pylist(), dtype=np.int64)
    ts_valid = ts[(ts > THRESHOLDS['ts_min_us']) & (ts < THRESHOLDS['ts_max_us'])]
    if len(ts_valid) == 0:
        return TestResult(f"{name}/coverage", False, "no valid timestamps")
    ts_min, ts_max = ts_valid.min(), ts_valid.max()
    minute_buckets = (ts_valid - ts_min) // (60_000_000)
    unique_mins, counts = np.unique(minute_buckets, return_counts=True)
    expected_range = range(int(unique_mins.min()), int(unique_mins.max()) + 1)
    count_map = dict(zip(unique_mins.tolist(), counts.tolist()))
    gap_mins = [m for m in expected_range if count_map.get(m, 0) == 0]
    # Consecutive gap check
    max_consec = 0
    cur = 0
    for m in expected_range:
        if count_map.get(m, 0) == 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    passed = max_consec <= THRESHOLDS['max_gap_minutes'] and len(gap_mins) == 0
    msg = (f"{len(expected_range)} minutes, {int(np.mean(counts))} avg events/min" +
           (f"; {len(gap_mins)} empty mins (max consec: {max_consec})" if gap_mins else " ✓"))
    return TestResult(f"{name}/coverage", passed, msg)

def test_bn_trade_ids(table):
    if 'trade_id' not in table.column_names:
        return TestResult("bn_trades/trade_id_gaps", True, "no trade_id — skipped")
    syms = table.column('symbol').to_pylist()
    tids = table.column('trade_id').to_pylist()
    by_sym = defaultdict(list)
    for s, t in zip(syms, tids):
        by_sym[s].append(t)
    total_gaps = 0
    details = {}
    for sym, ids in sorted(by_sym.items()):
        ids_s = sorted(ids)
        gaps = sum(ids_s[i] - ids_s[i-1] - 1 for i in range(1, len(ids_s)) if ids_s[i] - ids_s[i-1] > 1)
        total_gaps += gaps
        details[sym] = {"count": len(ids), "gaps": gaps}
    passed = total_gaps == 0
    msg = f"{total_gaps} missing IDs across {len(by_sym)} symbols"
    return TestResult("bn_trades/trade_id_gaps", passed, msg, details)

def test_outcome_consistency(table):
    required = {'outcome', 'target_price', 'close_price'}
    if not required.issubset(set(table.column_names)):
        return TestResult("poly_markets/outcome", True, "missing columns — skipped")
    outcomes = table.column('outcome').to_pylist()
    targets = table.column('target_price').to_pylist()
    closes = table.column('close_price').to_pylist()
    slugs = table.column('slug').to_pylist() if 'slug' in table.column_names else ['?'] * len(outcomes)
    mismatches = []
    ambiguous = []
    checked = 0
    for i in range(len(outcomes)):
        oc = outcomes[i]
        if oc == -1: continue
        if closes[i] == 0.0: continue
        checked += 1
        # When close ≈ target (float equality), the outcome is ambiguous —
        # small float rounding can flip it either way. Don't flag as mismatch.
        if abs(closes[i] - targets[i]) < 1e-6:
            ambiguous.append(f"row {i}: slug={slugs[i][:30]} close≈target={closes[i]}")
            continue
        expected = 1 if closes[i] > targets[i] else 0
        if expected != oc:
            mismatches.append(f"row {i}: slug={slugs[i][:30]} close={closes[i]} target={targets[i]} expected={expected} got={oc}")
    passed = len(mismatches) == 0
    msg = f"{checked} checked, {len(mismatches)} mismatches"
    if ambiguous:
        msg += f", {len(ambiguous)} ambiguous (close≈target)"
    return TestResult("poly_markets/outcome", passed, msg, {"mismatches": mismatches[:5], "ambiguous": ambiguous[:5]})

def test_window_durations(table):
    """Check window durations by merging switch+close rows per slug.
    
    poly_markets has two row types per window:
      - "switch" row (outcome=-1): has window_start_us / window_end_us
      - "close" row (outcome=0 or 1): has outcome, but timestamps may be 0
    WindowBacktest.load_windows() merges these by slug. We do the same.
    """
    df = table.to_pandas()
    windows = {}
    for _, row in df.iterrows():
        slug = row['slug']
        if slug not in windows:
            windows[slug] = {
                'interval_min': int(row['interval_min']),
                'window_start_us': int(row['window_start_us']),
                'window_end_us': int(row['window_end_us']),
                'outcome': int(row['outcome']),
            }
        # Update with non-zero timestamps from any row
        if int(row['window_start_us']) != 0:
            windows[slug]['window_start_us'] = int(row['window_start_us'])
        if int(row['window_end_us']) != 0:
            windows[slug]['window_end_us'] = int(row['window_end_us'])
        # Update outcome from close row
        if int(row['outcome']) in (0, 1):
            windows[slug]['outcome'] = int(row['outcome'])

    issues = []
    checked = 0
    for slug, w in windows.items():
        if w['outcome'] == -1:
            continue
        checked += 1
        if w['window_start_us'] == 0 or w['window_end_us'] == 0:
            issues.append(f"{slug[:30]}: no timestamps on any row")
            continue
        dur_min = (w['window_end_us'] - w['window_start_us']) / 60e6
        expected = w['interval_min']
        if abs(dur_min - expected) > 0.5:
            issues.append(f"{slug[:30]}: {dur_min:.1f}min vs expected {expected}min")

    passed = len(issues) == 0
    msg = f"{checked} resolved windows, {len(issues)} duration issues"
    return TestResult("poly_markets/window_durations", passed, msg, {"issues": issues[:5]})

def test_poly_books_best_bid_ask(table):
    if 'best_bid' not in table.column_names:
        return TestResult("poly_books/best_bid_ask", True, "no best_bid — skipped")
    bb = table.column('best_bid').to_pylist()
    ba = table.column('best_ask').to_pylist()
    bid_prices = table.column('bid_prices').to_pylist()
    ask_prices = table.column('ask_prices').to_pylist()
    mismatches = 0
    checked = 0
    for i in range(len(bb)):
        bp, ap = bid_prices[i], ask_prices[i]
        if not bp or not ap or len(bp) == 0 or len(ap) == 0: continue
        checked += 1
        if bb[i] is not None and abs(bb[i] - bp[0]) > 1e-6: mismatches += 1
        if ba[i] is not None and abs(ba[i] - ap[0]) > 1e-6: mismatches += 1
    passed = mismatches == 0
    return TestResult("poly_books/best_bid_ask", passed, f"{checked} checked, {mismatches} mismatches")

def test_poly_trades_prices(table):
    prices = np.array(table.column('price').to_pylist(), dtype=np.float64)
    below = int(np.sum(prices < 0))
    above = int(np.sum(prices > 1))
    passed = below == 0 and above == 0
    return TestResult("poly_trades/price_range", passed,
                      f"{below} below 0, {above} above 1" if not passed else f"all {len(prices)} in [0,1]")

def test_poly_books_event_type(table):
    if 'event_type' not in table.column_names:
        return TestResult("poly_books/event_type", True, "no event_type — skipped")
    etypes = table.column('event_type').to_pylist()
    invalid = [e for e in etypes if e not in (0, 1)]
    passed = len(invalid) == 0
    dist = Counter(etypes)
    return TestResult("poly_books/event_type", passed,
                      f"distribution: {dict(dist)}" + (f", {len(invalid)} invalid" if invalid else ""))

# ─── Runner ───────────────────────────────────────────────────────────

EXPECTED_SCHEMAS = {
    'poly_books':   ['ts_us', 'token_id', 'event_type', 'bid_prices', 'bid_sizes',
                     'ask_prices', 'ask_sizes', 'best_bid', 'best_ask'],
    'poly_trades':  ['ts_us', 'recv_us', 'token_id', 'price', 'size', 'tx_hash'],
    'bn_trades':    ['ts_us', 'recv_us', 'symbol', 'price', 'qty', 'is_buy', 'trade_id'],
    'bn_books':     ['ts_us', 'symbol', 'bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes'],
    'poly_markets': ['ts_us', 'slug', 'coin', 'interval_min', 'up_token_id',
                     'down_token_id', 'window_start_us', 'window_end_us',
                     'target_price', 'close_price', 'outcome'],
}

def run_all_tests(base_dir, date=None):
    results = []
    tables = {}
    for tname in EXPECTED_SCHEMAS:
        t = load_table(base_dir, tname, date)
        if t is None:
            results.append(TestResult(f"{tname}/load", False, "no data found"))
        else:
            tables[tname] = t
            results.append(TestResult(f"{tname}/load", True, f"{t.num_rows:,} rows"))

    for tname, expected in EXPECTED_SCHEMAS.items():
        if tname in tables:
            results.append(test_schema(tables[tname], tname, expected))
    for tname in tables:
        results.append(test_timestamps(tables[tname], tname))
    for tname in ['poly_trades', 'bn_trades']:
        if tname in tables:
            results.append(test_latency(tables[tname], tname))
    for tname in ['poly_books', 'bn_books']:
        if tname in tables:
            results.append(test_book_integrity(tables[tname], tname))
    for tname in tables:
        results.append(test_coverage(tables[tname], tname))
    if 'bn_trades' in tables:
        results.append(test_bn_trade_ids(tables['bn_trades']))
    if 'poly_markets' in tables:
        results.append(test_outcome_consistency(tables['poly_markets']))
        results.append(test_window_durations(tables['poly_markets']))
    if 'poly_books' in tables:
        results.append(test_poly_books_best_bid_ask(tables['poly_books']))
        results.append(test_poly_books_event_type(tables['poly_books']))
    if 'poly_trades' in tables:
        results.append(test_poly_trades_prices(tables['poly_trades']))
    return results

def print_results(results):
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"\n{'='*60}")
    print(f"QUALITY VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(r)
    print(f"\n{'─'*60}")
    print(f"TOTAL: {passed} passed, {failed} failed, {len(results)} total")
    if failed:
        print(f"\nFailed:")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.message}")
                if r.details:
                    for k, v in r.details.items():
                        if isinstance(v, list) and len(v) > 3:
                            print(f"    {k}: {v[:3]}... ({len(v)} total)")
                        else:
                            print(f"    {k}: {v}")
    return failed == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', nargs='?', default='historical_data/events')
    parser.add_argument('--date', default=None)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    results = run_all_tests(args.base_dir, args.date)
    if args.json:
        out = [{"name": r.name, "passed": r.passed, "message": r.message, "details": r.details} for r in results]
        print(json.dumps(out, indent=2, default=str))
    else:
        ok = print_results(results)
        sys.exit(0 if ok else 1)
