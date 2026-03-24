#!/usr/bin/env python3
"""
analyze_strategy.py — Deep dive on a specific strategy, trade by trade.

Usage:
  python analyze_strategy.py historical_data/events
"""

import sys, os, glob, time, argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from backtest_allcoins import (
    load_all_markets, load_books_and_trades, load_bn_trades,
    cache_fills, compute_all_signals, evaluate, stats, COIN_TO_SYMBOL,
)


def main(events_dir):
    print(f"\n{'═'*74}")
    print(f"  BTC C3 STRATEGY — DETAILED ANALYSIS")
    print(f"  fi + dist>=+5bps, BTC only, 15m windows, taker+fees")
    print(f"{'═'*74}")

    t0 = time.time()
    print(f"\n  Loading data...")
    all_windows = load_all_markets(events_dir)

    # BTC only
    btc_windows = {s: w for s, w in all_windows.items() if w['coin'] == 'btc'}
    tokens = set()
    for w in btc_windows.values():
        tokens.add(w['up_token']); tokens.add(w['down_token'])

    books, trades = load_books_and_trades(events_dir, tokens)
    bn_by_symbol = load_bn_trades(events_dir)
    bn = bn_by_symbol.get('BTCUSDT')

    fills = cache_fills(btc_windows, books)
    signals = compute_all_signals(btc_windows, books, trades, bn_data=bn)

    # Run BTC C3
    fi15 = dict(obs_secs=30, flow_threshold=0.2, use_imbalance=True, interval_filter=15)
    results = evaluate(btc_windows, fills, signals, **fi15, bn_distance_min=5)
    results = sorted(results, key=lambda t: t['start_us'])

    s = stats(results)
    print(f"\n  Overall: N={s['n']}  PnL=${s['pnl']:+.1f}  WR={s['wr']}%  "
          f"Sharpe={s['sharpe']:+.3f}  DD=${s['dd']:.1f}")
    print(f"  avg_win=${s['avg_win']:+.2f}  avg_loss=${s['avg_loss']:+.2f}  "
          f"max_consec_loss={s['max_consec_loss']}  max_consec_win={s['max_consec_win']}")

    # ═══════════════════════════════════════════════════════════════
    # Trade-by-trade table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  TRADE-BY-TRADE LOG")
    print(f"{'═'*90}")
    print(f"\n  {'#':>3s}  {'date':>18s}  {'dir':>4s}  {'entry':>6s}  {'settle':>6s}  "
          f"{'PnL':>8s}  {'cumPnL':>8s}  {'WR':>5s}  {'dist':>7s}  {'flow':>5s}  {'imb':>5s}")
    print(f"  {'─'*88}")

    cum_pnl = 0
    wins = 0
    for i, t in enumerate(results):
        cum_pnl += t['pnl']
        if t['pnl'] > 0:
            wins += 1
        wr = wins / (i + 1) * 100
        dt = datetime.fromtimestamp(t['start_us'] / 1e6, tz=timezone.utc)
        pnl_str = f"${t['pnl']:>+7.2f}"
        cum_str = f"${cum_pnl:>+7.1f}"
        win_class = "W" if t['pnl'] > 0 else "L"
        dist = t.get('bn_distance_bps', 0)
        flow = t.get('flow', 0)
        imb = t.get('imbalance', 0)
        print(f"  {i+1:>3d}  {dt.strftime('%Y-%m-%d %H:%M'):>18s}  {t['direction']:>4s}  "
              f"{t['entry']:>6.4f}  {t['settle']:>6.1f}  "
              f"{pnl_str}  {cum_str}  {wr:>4.0f}%  {dist:>+6.1f}  {flow:>+.2f}  {imb:>+.2f}  {win_class}")

    # ═══════════════════════════════════════════════════════════════
    # Rolling metrics
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  ROLLING METRICS (cumulative as trades accumulate)")
    print(f"{'═'*74}")

    pnls = np.array([t['pnl'] for t in results])
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak

    print(f"\n  {'after N':>8s}  {'cum PnL':>9s}  {'WR':>5s}  {'avg PnL':>8s}  {'Sharpe':>7s}  {'DD':>8s}  {'peak':>8s}")
    print(f"  {'─'*60}")

    milestones = list(range(5, len(pnls) + 1, 5))
    if len(pnls) not in milestones:
        milestones.append(len(pnls))

    for n in milestones:
        sub = pnls[:n]
        w = int((sub > 0).sum())
        wr = w / n * 100
        avg = float(sub.mean())
        sh = float(sub.mean() / sub.std()) if sub.std() > 0 else 0
        print(f"  {n:>8d}  ${cum[n-1]:>+8.1f}  {wr:>4.0f}%  ${avg:>+7.2f}  {sh:>+6.3f}  ${dd[n-1]:>+7.1f}  ${peak[n-1]:>+7.1f}")

    # ═══════════════════════════════════════════════════════════════
    # Entry price distribution
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  ENTRY PRICE ANALYSIS")
    print(f"{'═'*74}")

    entries = np.array([t['entry'] for t in results])
    win_entries = entries[pnls > 0]
    loss_entries = entries[pnls <= 0]

    print(f"\n  All entries:   min={entries.min():.4f}  median={np.median(entries):.4f}  "
          f"max={entries.max():.4f}  mean={entries.mean():.4f}")
    if len(win_entries) > 0:
        print(f"  Winners:       min={win_entries.min():.4f}  median={np.median(win_entries):.4f}  "
              f"max={win_entries.max():.4f}  mean={win_entries.mean():.4f}")
    if len(loss_entries) > 0:
        print(f"  Losers:        min={loss_entries.min():.4f}  median={np.median(loss_entries):.4f}  "
              f"max={loss_entries.max():.4f}  mean={loss_entries.mean():.4f}")

    # PnL when entry price is cheap vs expensive
    med_entry = np.median(entries)
    cheap = pnls[entries < med_entry]
    expensive = pnls[entries >= med_entry]
    if len(cheap) > 0:
        print(f"\n  Entry < {med_entry:.3f} (cheap):    N={len(cheap)}  "
              f"WR={int((cheap>0).sum())/len(cheap)*100:.0f}%  avg=${cheap.mean():+.2f}")
    if len(expensive) > 0:
        print(f"  Entry >= {med_entry:.3f} (expensive): N={len(expensive)}  "
              f"WR={int((expensive>0).sum())/len(expensive)*100:.0f}%  avg=${expensive.mean():+.2f}")

    # ═══════════════════════════════════════════════════════════════
    # Distance analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  DISTANCE AT ENTRY")
    print(f"{'═'*74}")

    dists = np.array([t.get('bn_distance_bps', 0) for t in results])
    win_dists = dists[pnls > 0]
    loss_dists = dists[pnls <= 0]

    print(f"\n  All:     min={dists.min():+.1f}  median={np.median(dists):+.1f}  "
          f"max={dists.max():+.1f}  mean={dists.mean():+.1f}")
    if len(win_dists) > 0:
        print(f"  Winners: min={win_dists.min():+.1f}  median={np.median(win_dists):+.1f}  "
              f"max={win_dists.max():+.1f}  mean={win_dists.mean():+.1f}")
    if len(loss_dists) > 0:
        print(f"  Losers:  min={loss_dists.min():+.1f}  median={np.median(loss_dists):+.1f}  "
              f"max={loss_dists.max():+.1f}  mean={loss_dists.mean():+.1f}")

    # Buckets
    print(f"\n  PnL by distance bucket:")
    print(f"  {'bucket':>16s}  {'N':>4s}  {'WR':>5s}  {'avg PnL':>8s}  {'total':>9s}")
    print(f"  {'─'*48}")
    buckets = [(5, 7, "5-7 bps"), (7, 10, "7-10 bps"), (10, 15, "10-15 bps"),
               (15, 25, "15-25 bps"), (25, 50, "25-50 bps"), (50, 999, "50+ bps")]
    for lo, hi, label in buckets:
        mask = (dists >= lo) & (dists < hi)
        sub = pnls[mask]
        if len(sub) == 0: continue
        wr = int((sub > 0).sum()) / len(sub) * 100
        print(f"  {label:>16s}  {len(sub):>4d}  {wr:>4.0f}%  ${sub.mean():>+7.2f}  ${sub.sum():>+8.1f}")

    # ═══════════════════════════════════════════════════════════════
    # Time of day analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  TIME OF DAY (UTC)")
    print(f"{'═'*74}")

    print(f"\n  {'hour':>6s}  {'N':>4s}  {'WR':>5s}  {'avg PnL':>8s}  {'total':>9s}")
    print(f"  {'─'*38}")
    by_hour = defaultdict(list)
    for t in results:
        dt = datetime.fromtimestamp(t['start_us'] / 1e6, tz=timezone.utc)
        by_hour[dt.hour].append(t['pnl'])

    for h in sorted(by_hour.keys()):
        hp = np.array(by_hour[h])
        wr = int((hp > 0).sum()) / len(hp) * 100
        print(f"  {h:>4d}h  {len(hp):>4d}  {wr:>4.0f}%  ${hp.mean():>+7.2f}  ${hp.sum():>+8.1f}")

    # ═══════════════════════════════════════════════════════════════
    # Streak analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  STREAK ANALYSIS")
    print(f"{'═'*74}")

    streaks_w = []
    streaks_l = []
    cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            if cur_l > 0:
                streaks_l.append(cur_l)
            cur_l = 0
        else:
            cur_l += 1
            if cur_w > 0:
                streaks_w.append(cur_w)
            cur_w = 0
    if cur_w > 0: streaks_w.append(cur_w)
    if cur_l > 0: streaks_l.append(cur_l)

    print(f"\n  Win streaks:  {streaks_w}")
    print(f"  Loss streaks: {streaks_l}")
    if streaks_w:
        print(f"  Avg win streak:  {np.mean(streaks_w):.1f}  max: {max(streaks_w)}")
    if streaks_l:
        print(f"  Avg loss streak: {np.mean(streaks_l):.1f}  max: {max(streaks_l)}")

    # Probability of N consecutive losses (theoretical)
    wr_est = s['wr'] / 100
    print(f"\n  Probability of consecutive losses (at WR={s['wr']}%):")
    for n_loss in [2, 3, 4, 5]:
        # Probability of seeing N consecutive losses somewhere in M trades
        m = 100  # over 100 trades
        p_single = (1 - wr_est) ** n_loss
        p_somewhere = 1 - (1 - p_single) ** m
        print(f"    {n_loss} in a row over {m} trades: {p_somewhere:.1%}")

    # ═══════════════════════════════════════════════════════════════
    # Comparison with other thresholds (BTC only)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  BTC — THRESHOLD COMPARISON (same strategy, different distance)")
    print(f"{'═'*74}")

    print(f"\n  {'threshold':>14s}  {'N':>4s}  {'PnL':>9s}  {'avg':>7s}  {'WR':>5s}  {'Sharpe':>7s}  {'consec_L':>8s}")
    print(f"  {'─'*60}")

    for d_min in [0, 3, 5, 7, 10, 15, 20]:
        r = evaluate(btc_windows, fills, signals, **fi15, bn_distance_min=d_min)
        st = stats(r)
        if st['n'] == 0: continue
        print(f"  {'dist>=+'+str(d_min)+'bps':>14s}  {st['n']:>4d}  ${st['pnl']:>+8.1f}  "
              f"${st['avg']:>+6.2f}  {st['wr']:>4.1f}%  {st['sharpe']:>+6.3f}  "
              f"{st['max_consec_loss']:>8d}")

    print(f"\n{'═'*74}")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"{'═'*74}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    args = parser.parse_args()
    main(args.events_dir)
