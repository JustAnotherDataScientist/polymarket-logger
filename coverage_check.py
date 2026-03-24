#!/usr/bin/env python3
"""coverage_check.py — Show data coverage per day per coin."""
import glob, sys
import pyarrow.parquet as pq, pyarrow as pa
from collections import defaultdict
from datetime import datetime, timezone

mdir = sys.argv[1] if len(sys.argv) > 1 else 'historical_data/events/poly_markets'
files = sorted(glob.glob(f'{mdir}/*.parquet'))
if not files:
    mdir = 'historical_data/events/poly_markets'
    files = sorted(glob.glob(f'{mdir}/*.parquet'))

mdf = pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()

windows = {}
for _, row in mdf.iterrows():
    slug = row['slug']
    if int(row['interval_min']) != 15:
        continue
    if slug not in windows:
        windows[slug] = {
            'coin': row['coin'],
            'start_us': int(row['window_start_us']),
            'outcome': -1,
        }
    if int(row.get('outcome', -1)) in (0, 1):
        windows[slug]['outcome'] = int(row['outcome'])

by_date_coin = defaultdict(int)
for w in windows.values():
    if w['outcome'] not in (0, 1):
        continue
    dt = datetime.fromtimestamp(w['start_us'] / 1e6, tz=timezone.utc)
    by_date_coin[(dt.strftime('%Y-%m-%d'), w['coin'])] += 1

dates = sorted(set(k[0] for k in by_date_coin))
coins = sorted(set(k[1] for k in by_date_coin))

print()
print("  COVERAGE — resolved 15m windows per day (max = 96 per coin)")
print("  " + "=" * 60)
header = "  date        " + "  ".join(f"{c:>6s}" for c in coins) + "  total"
print(header)
print("  " + "-" * 60)

for d in dates:
    parts = []
    for c in coins:
        n = by_date_coin.get((d, c), 0)
        parts.append(f"{n:>3d}/96")
    total = sum(by_date_coin.get((d, c), 0) for c in coins)
    pct = total / (96 * len(coins)) * 100
    print(f"  {d}  " + "  ".join(parts) + f"  {total:>4d}  ({pct:.0f}%)")

# Hourly heatmap
print()
print("  HOURLY HEATMAP — number of resolved windows per hour (all coins)")
print("  " + "=" * 60)

by_hour = defaultdict(int)
for w in windows.values():
    if w['outcome'] not in (0, 1):
        continue
    dt = datetime.fromtimestamp(w['start_us'] / 1e6, tz=timezone.utc)
    by_hour[(dt.strftime('%Y-%m-%d'), dt.hour)] += 1

# Max per hour = 4 windows/coin * N coins
max_per_hour = 4 * len(coins)

for d in dates:
    cells = []
    for h in range(24):
        n = by_hour.get((d, h), 0)
        if n == 0:
            cells.append(" ..")
        elif n >= max_per_hour:
            cells.append(f" {n:>2d}")
        else:
            cells.append(f" {n:>2d}")
    print(f"  {d} {''.join(cells)}")

print("  " + " " * 11 + "".join(f" {h:>2d}" for h in range(24)))
print(f"  (max per hour = {max_per_hour} = 4 windows x {len(coins)} coins)")
print()

# Summary
total_resolved = sum(by_date_coin.values())
total_possible = 96 * len(coins) * len(dates)
print(f"  Total: {total_resolved} / {total_possible} = {total_resolved/total_possible*100:.1f}% coverage")
print()
