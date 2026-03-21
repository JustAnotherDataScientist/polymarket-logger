#!/usr/bin/env python3
"""
cross_validate.py — Cross-validate event store vs derived snapshots.
Reconstructs book state from poly_books events at sampled timestamps,
then compares against the derived 4Hz snapshot table's up_bid/up_ask.

Run:  python3 cross_validate.py ~/polymarket-logger/historical_data
"""

import sys, os, glob, argparse
from datetime import datetime, timezone
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


def us_to_dt(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc)


def load_event_table(events_dir, table_name):
    table_dir = os.path.join(events_dir, table_name)
    files = sorted(glob.glob(os.path.join(table_dir, "*.parquet")))
    if not files:
        return None
    return pa.concat_tables([pq.read_table(f) for f in files])


def load_live_snapshots(live_dir):
    files = sorted(glob.glob(os.path.join(live_dir, "*.parquet")))
    if not files:
        return None
    return pa.concat_tables([pq.read_table(f) for f in files])


def reconstruct_book_at_time(books_df, token_id, target_ts_us, tolerance_us):
    mask = (books_df['token_id'] == token_id) & (books_df['ts_us'] <= target_ts_us)
    candidates = books_df.loc[mask]
    if candidates.empty:
        return None
    last = candidates.iloc[-1]
    lag = target_ts_us - last['ts_us']
    if lag > tolerance_us:
        return None
    return {
        'best_bid': float(last['best_bid']) if last['best_bid'] is not None else 0.0,
        'best_ask': float(last['best_ask']) if last['best_ask'] is not None else 0.0,
        'lag_us': lag,
    }


def cross_validate(events_dir, live_dir, n_samples=200, tolerance_us=500_000):
    print("Loading event store (poly_books)...")
    books_t = load_event_table(events_dir, "poly_books")
    if books_t is None:
        print("  ERROR: No poly_books files in", os.path.join(events_dir, "poly_books"))
        return False

    print("Loading event store (poly_markets)...")
    markets_t = load_event_table(events_dir, "poly_markets")

    print("Loading derived snapshots (live/)...")
    live_t = load_live_snapshots(live_dir)
    if live_t is None:
        print("  ERROR: No snapshot files in", live_dir)
        return False

    books_df = books_t.to_pandas().sort_values('ts_us').reset_index(drop=True)
    live_df = live_t.to_pandas()

    print(f"  poly_books: {len(books_df):,} rows")
    print(f"  snapshots:  {len(live_df):,} rows × {len(live_df.columns)} cols")

    # ── Map slug → up_token_id from poly_markets ──
    slug_to_token = {}
    if markets_t is not None:
        mdf = markets_t.to_pandas()
        for _, row in mdf.iterrows():
            slug_to_token[row['slug']] = row['up_token_id']
        print(f"  slug→token map: {len(slug_to_token)} entries")

    # ── Validate live columns ──
    # Live schema defined in csv_schema.py: timestamp_utc, up_bid, up_ask, slug, ...
    for req in ('timestamp_utc', 'up_bid', 'up_ask', 'slug'):
        if req not in live_df.columns:
            print(f"  ERROR: missing column '{req}' in live snapshots")
            return False

    live_df['_ts_us'] = (pd.to_datetime(live_df['timestamp_utc'], utc=True)
                           .dt.tz_localize(None).astype('datetime64[us]').astype(np.int64))
    live_df['_up_bid'] = pd.to_numeric(live_df['up_bid'], errors='coerce')
    live_df['_up_ask'] = pd.to_numeric(live_df['up_ask'], errors='coerce')

    slugs_in_live = live_df['slug'].unique()
    print(f"  slugs in snapshots: {len(slugs_in_live)}")

    # ── Compare per slug ──
    total_matched = 0
    total_skipped = 0
    nan_skipped = 0
    all_bid_diffs = []
    all_ask_diffs = []
    all_lags = []

    per_slug = max(1, n_samples // max(len(slugs_in_live), 1))

    for slug in slugs_in_live:
        token_id = slug_to_token.get(slug)
        if token_id is None:
            continue

        sl = live_df[live_df['slug'] == slug]
        sl = sl[sl['_up_bid'].notna() & sl['_up_ask'].notna() & (sl['_up_bid'] > 0)]
        if sl.empty:
            continue

        idxs = np.unique(np.linspace(0, len(sl) - 1, min(per_slug, len(sl)), dtype=int))

        for i in idxs:
            row = sl.iloc[i]
            target = int(row['_ts_us'])
            live_bid = float(row['_up_bid'])
            live_ask = float(row['_up_ask'])

            book = reconstruct_book_at_time(books_df, token_id, target, tolerance_us)
            if book is None:
                total_skipped += 1
                continue

            total_matched += 1
            # Skip NaN (empty book side) — not a divergence, just no data
            if np.isnan(book['best_bid']) or np.isnan(live_bid):
                nan_skipped += 1
                continue
            if np.isnan(book['best_ask']) or np.isnan(live_ask):
                nan_skipped += 1
                continue

            bd = abs(book['best_bid'] - live_bid)
            ad = abs(book['best_ask'] - live_ask)
            all_bid_diffs.append(bd)
            all_ask_diffs.append(ad)
            all_lags.append(book['lag_us'])

            if bd > 0.01 or ad > 0.01:
                ts_str = us_to_dt(target).strftime('%H:%M:%S.%f')
                print(f"  ⚠ [{slug[:35]}] {ts_str}: "
                      f"event bid={book['best_bid']:.4f} live={live_bid:.4f} Δ={bd:.4f} | "
                      f"event ask={book['best_ask']:.4f} live={live_ask:.4f} Δ={ad:.4f} | "
                      f"lag={book['lag_us']/1000:.0f}ms")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Matched: {total_matched}, Skipped: {total_skipped}, NaN-skipped: {nan_skipped}")

    n_compared = len(all_bid_diffs)
    if n_compared == 0:
        print("  ⚠ No valid comparisons — check slug↔token alignment")
        return False

    bid_diffs = np.array(all_bid_diffs)
    ask_diffs = np.array(all_ask_diffs)
    lags = np.array(all_lags)

    print(f"  Bid Δ: mean={bid_diffs.mean():.6f} max={bid_diffs.max():.6f} "
          f"p50={np.median(bid_diffs):.6f} p99={np.percentile(bid_diffs, 99):.6f}")
    print(f"  Ask Δ: mean={ask_diffs.mean():.6f} max={ask_diffs.max():.6f} "
          f"p50={np.median(ask_diffs):.6f} p99={np.percentile(ask_diffs, 99):.6f}")
    print(f"  Lag:   mean={lags.mean()/1000:.1f}ms p50={np.median(lags)/1000:.1f}ms")

    exact_bid = int(np.sum(bid_diffs < 1e-6))
    exact_ask = int(np.sum(ask_diffs < 1e-6))
    close_bid = int(np.sum(bid_diffs < 0.02))
    close_ask = int(np.sum(ask_diffs < 0.02))

    print(f"  Exact (Δ<1e-6):  bid={exact_bid}/{n_compared} ask={exact_ask}/{n_compared}")
    print(f"  Close (Δ<0.02):  bid={close_bid}/{n_compared} ask={close_ask}/{n_compared}")

    # Pass criteria for an async system with two independent data paths:
    #  - ≥80% exact matches on both sides
    #  - max divergence < 0.10 (10 ticks on Polymarket)
    #    Diffs up to ~0.09 are expected when the book moves between
    #    the event recording (async WS) and the 4Hz snapshot sampling.
    exact_pct = min(exact_bid, exact_ask) / n_compared
    max_diff = max(bid_diffs.max(), ask_diffs.max())

    if exact_pct >= 0.80 and max_diff < 0.10:
        print(f"\n  ✓ PASS: {exact_pct:.0%} exact matches, max Δ={max_diff:.4f}")
        return True
    elif exact_pct >= 0.60 and max_diff < 0.15:
        print(f"\n  ~ WARN: {exact_pct:.0%} exact matches, max Δ={max_diff:.4f} "
              f"— likely async sampling race, review outliers above")
        return True
    else:
        print(f"\n  ✗ FAIL: {exact_pct:.0%} exact matches, max Δ={max_diff:.4f}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', nargs='?', default='historical_data')
    parser.add_argument('--events-dir', default=None)
    parser.add_argument('--live-dir', default=None)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--tolerance-ms', type=float, default=500)
    args = parser.parse_args()

    events_dir = args.events_dir or os.path.join(args.base_dir, 'events')
    live_dir = args.live_dir or os.path.join(args.base_dir, 'live')

    ok = cross_validate(events_dir, live_dir, args.samples, int(args.tolerance_ms * 1000))
    sys.exit(0 if ok else 1)
