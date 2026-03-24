#!/usr/bin/env python3
"""
backtest_dynamic.py — Continuous-scan backtest engine.

Instead of "look at first 30s, decide once", this engine builds a timeline
of snapshots every ~5s for each window, then each strategy scans the timeline
and enters at the first moment its trigger fires.

All PnL includes Polymarket crypto taker fees.

Usage:
  python backtest_dynamic.py historical_data/events
  python backtest_dynamic.py historical_data/events --coin btc sol
"""

import sys, os, glob, argparse, time, math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))

COIN_TO_SYMBOL = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}

def taker_fee_rate(price):
    p = max(0.01, min(0.99, price))
    return 0.072 * (p * (1 - p))

def prob_up_phi(distance_bps, vol_bps_s, seconds_remaining):
    if distance_bps is None or vol_bps_s is None:
        return None
    if seconds_remaining is None or seconds_remaining <= 0:
        return 1.0 if distance_bps > 0 else (0.0 if distance_bps < 0 else 0.5)
    t = max(seconds_remaining, 30.0)
    v = max(vol_bps_s, 0.5)
    z = distance_bps / (v * math.sqrt(t))
    return 0.5 * math.erfc(-z / math.sqrt(2))


# ═══════════════════════════════════════════════════════════════════════
# Data loading (same as v4, compact)
# ═══════════════════════════════════════════════════════════════════════

def load_markets(events_dir):
    mdir = os.path.join(events_dir, 'poly_markets')
    files = sorted(glob.glob(os.path.join(mdir, '*.parquet')))
    if not files: return {}
    mdf = pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()
    windows = {}
    for _, row in mdf.iterrows():
        slug = row['slug']
        if slug not in windows:
            windows[slug] = {
                'slug': slug, 'coin': row['coin'], 'interval': int(row['interval_min']),
                'up_token': row['up_token_id'], 'down_token': row['down_token_id'],
                'start_us': int(row['window_start_us']), 'end_us': int(row['window_end_us']),
                'target_price': float(row.get('target_price', 0)), 'outcome': -1,
            }
        if int(row.get('outcome', -1)) in (0, 1):
            windows[slug]['outcome'] = int(row['outcome'])
    return {s: w for s, w in windows.items() if w['outcome'] in (0, 1)}

def load_books(events_dir, token_ids):
    t0 = time.time()
    token_set = set(token_ids)
    bdir = os.path.join(events_dir, 'poly_books')
    chunks = []
    for f in sorted(glob.glob(os.path.join(bdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(token_set)]
        if len(df) > 0: chunks.append(df)
    books = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(books) > 0:
        books = books[(books['best_bid'] > 0) & (books['best_ask'] > 0)].copy()
        books['mid'] = (books['best_bid'] + books['best_ask']) / 2
        books['spread'] = books['best_ask'] - books['best_bid']
    print(f"    Books: {len(books):,} | {time.time()-t0:.1f}s")
    return books

def load_trades(events_dir, token_ids):
    t0 = time.time()
    token_set = set(token_ids)
    tdir = os.path.join(events_dir, 'poly_trades')
    chunks = []
    for f in sorted(glob.glob(os.path.join(tdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(token_set)]
        if len(df) > 0: chunks.append(df)
    trades = pd.concat(chunks, ignore_index=True).sort_values('ts_us') if chunks else pd.DataFrame()
    print(f"    Trades: {len(trades):,} | {time.time()-t0:.1f}s")
    return trades

def load_bn_trades(events_dir):
    t0 = time.time()
    bdir = os.path.join(events_dir, 'bn_trades')
    files = sorted(glob.glob(os.path.join(bdir, '*.parquet')))
    if not files: return {}
    chunks = []
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
            if len(df) > 0: chunks.append(df)
        except Exception: pass
    if not chunks: return {}
    bn = pd.concat(chunks, ignore_index=True).sort_values('ts_us')
    print(f"    Binance: {len(bn):,} trades | {time.time()-t0:.1f}s")
    result = {}
    for sym, grp in bn.groupby('symbol'):
        grp = grp.sort_values('ts_us')
        result[sym] = {'ts_us': grp['ts_us'].values, 'price': grp['price'].values.astype(np.float64),
                        'qty': grp['qty'].values.astype(np.float64), 'is_buy': grp['is_buy'].values}
    return result


# ═══════════════════════════════════════════════════════════════════════
# Timeline builder — one snapshot every STEP_S seconds per window
# ═══════════════════════════════════════════════════════════════════════

STEP_US = 5_000_000  # 5 seconds

def _bn_price_at(bn, t_us):
    idx = np.searchsorted(bn['ts_us'], t_us, side='right') - 1
    return float(bn['price'][idx]) if idx >= 0 else None

def _bn_vol_at(bn, t_us, lookback_s=60):
    ts, px = bn['ts_us'], bn['price']
    i_lo = np.searchsorted(ts, t_us - lookback_s*1_000_000, side='left')
    i_hi = np.searchsorted(ts, t_us, side='right')
    if i_hi - i_lo < 10: return None
    w = px[i_lo:i_hi]
    step = max(1, len(w) // lookback_s)
    s = w[::step].astype(np.float64)
    if len(s) < 3: return None
    return float(np.std(np.diff(np.log(s))) * 10000)

def _bn_ret_at(bn, t_us, lookback_us):
    ts, px = bn['ts_us'], bn['price']
    idx_now = np.searchsorted(ts, t_us, side='right') - 1
    idx_past = np.searchsorted(ts, t_us - lookback_us, side='right') - 1
    if idx_now < 0 or idx_past < 0: return 0
    pn, pp = float(px[idx_now]), float(px[idx_past])
    return (pn/pp - 1)*10000 if pp > 0 else 0

def _bn_flow_at(bn, t_us, lookback_us):
    ts, qty, ib = bn['ts_us'], bn['qty'], bn['is_buy']
    i_lo = np.searchsorted(ts, t_us - lookback_us, side='left')
    i_hi = np.searchsorted(ts, t_us, side='right')
    if i_hi <= i_lo: return 0
    wq, wb = qty[i_lo:i_hi], ib[i_lo:i_hi]
    bv, sv = float(wq[wb].sum()), float(wq[~wb].sum())
    return (bv-sv)/(bv+sv) if (bv+sv) > 0 else 0

def _poly_flow_at(trade_idx, t_us, lookback_us):
    """Compute Poly trade flow (buy-sell imbalance) over lookback."""
    if trade_idx is None: return 0, 0
    tts = trade_idx['ts_us']
    j_lo = np.searchsorted(tts, t_us - lookback_us, side='left')
    j_hi = np.searchsorted(tts, t_us, side='right')
    n = j_hi - j_lo
    if n == 0: return 0, 0
    sz = trade_idx['size'][j_lo:j_hi]
    ib = trade_idx['is_buy'][j_lo:j_hi]
    bv, sv = float(sz[ib].sum()), float(sz[~ib].sum())
    total = bv + sv
    flow = (bv - sv) / total if total > 0 else 0
    return flow, n


def build_timelines(resolved, books, trades, bn_by_symbol, interval_filter=15):
    """
    For each window, build a numpy-friendly timeline of snapshots.
    Returns dict: slug → timeline dict with arrays.
    """
    # Index books by UP token
    book_idx = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        book_idx[tid] = {
            'ts_us': grp['ts_us'].values,
            'mid': grp['mid'].values.astype(np.float64),
            'best_bid': grp['best_bid'].values.astype(np.float64),
            'best_ask': grp['best_ask'].values.astype(np.float64),
            'spread': grp['spread'].values.astype(np.float64),
            'ask_prices': grp['ask_prices'].values if 'ask_prices' in grp.columns else None,
            'ask_sizes': grp['ask_sizes'].values if 'ask_sizes' in grp.columns else None,
        }

    # Index trades by token + classify buy/sell
    trade_idx = {}
    if len(trades) > 0:
        for tid, grp in trades.groupby('token_id'):
            grp = grp.sort_values('ts_us')
            td = {'ts_us': grp['ts_us'].values,
                  'price': grp['price'].values.astype(np.float64),
                  'size': grp['size'].values.astype(np.float64)}
            # Classify using book mid
            bd = book_idx.get(tid)
            if bd is not None and len(bd['ts_us']) > 0:
                idx = np.clip(np.searchsorted(bd['ts_us'], td['ts_us'], side='right')-1, 0, len(bd['mid'])-1)
                td['is_buy'] = td['price'] >= bd['mid'][idx]
            else:
                td['is_buy'] = td['price'] > 0
            trade_idx[tid] = td

    timelines = {}
    n_skip = 0

    for slug, w in resolved.items():
        if w['interval'] != interval_filter: continue
        coin = w['coin']
        symbol = COIN_TO_SYMBOL.get(coin, '')
        bn = bn_by_symbol.get(symbol)
        if bn is None: n_skip += 1; continue

        up_token = w['up_token']
        dn_token = w['down_token']
        bd = book_idx.get(up_token)
        bd_dn = book_idx.get(dn_token)
        if bd is None or len(bd['ts_us']) < 5: n_skip += 1; continue

        start_us, end_us = w['start_us'], w['end_us']
        target = w['target_price']
        total_s = (end_us - start_us) / 1e6
        if target <= 0 or total_s <= 0: n_skip += 1; continue

        bk_ts = bd['ts_us']
        # Pre-filter book range for this window
        bi_lo = np.searchsorted(bk_ts, start_us, side='left')
        bi_hi = np.searchsorted(bk_ts, end_us, side='right')
        if bi_hi - bi_lo < 3: n_skip += 1; continue

        # DOWN book range (may be None if no data)
        dn_bk_ts = bd_dn['ts_us'] if bd_dn is not None else None
        dn_bi_lo = np.searchsorted(dn_bk_ts, start_us, side='left') if dn_bk_ts is not None else 0
        dn_bi_hi = np.searchsorted(dn_bk_ts, end_us, side='right') if dn_bk_ts is not None else 0

        # Build snapshots at regular intervals
        snap_times = np.arange(start_us + STEP_US, end_us, STEP_US)
        n_snaps = len(snap_times)
        if n_snaps < 3: n_skip += 1; continue

        # Pre-allocate arrays
        elapsed_s = np.empty(n_snaps, dtype=np.float64)
        remaining_s = np.empty(n_snaps, dtype=np.float64)
        pct_elapsed = np.empty(n_snaps, dtype=np.float64)
        bn_price_arr = np.empty(n_snaps, dtype=np.float64)
        bn_vol_arr = np.empty(n_snaps, dtype=np.float64)
        bn_ret_10s = np.empty(n_snaps, dtype=np.float64)
        bn_ret_30s = np.empty(n_snaps, dtype=np.float64)
        bn_ret_60s = np.empty(n_snaps, dtype=np.float64)
        bn_flow_30s = np.empty(n_snaps, dtype=np.float64)
        distance_bps = np.empty(n_snaps, dtype=np.float64)
        fair_value = np.empty(n_snaps, dtype=np.float64)
        poly_mid = np.empty(n_snaps, dtype=np.float64)
        poly_ask = np.empty(n_snaps, dtype=np.float64)
        poly_bid = np.empty(n_snaps, dtype=np.float64)
        poly_spread = np.empty(n_snaps, dtype=np.float64)
        edge_arr = np.empty(n_snaps, dtype=np.float64)
        edge_vs_ask = np.empty(n_snaps, dtype=np.float64)
        z_score_arr = np.empty(n_snaps, dtype=np.float64)
        poly_flow_arr = np.empty(n_snaps, dtype=np.float64)
        poly_imb_arr = np.empty(n_snaps, dtype=np.float64)

        valid = np.ones(n_snaps, dtype=bool)
        up_ti = trade_idx.get(up_token)
        dn_ti = trade_idx.get(dn_token)

        # DOWN token ask price array
        dn_poly_ask = np.full(n_snaps, np.nan, dtype=np.float64)
        dn_poly_flow_arr = np.zeros(n_snaps, dtype=np.float64)

        for i, t in enumerate(snap_times):
            el = (t - start_us) / 1e6
            rem = (end_us - t) / 1e6
            elapsed_s[i] = el
            remaining_s[i] = rem
            pct_elapsed[i] = el / total_s * 100

            # Binance
            bp = _bn_price_at(bn, t)
            if bp is None:
                valid[i] = False; continue
            bn_price_arr[i] = bp
            bv = _bn_vol_at(bn, t, lookback_s=60)
            bn_vol_arr[i] = bv if bv is not None else 0
            bn_ret_10s[i] = _bn_ret_at(bn, t, 10_000_000)
            bn_ret_30s[i] = _bn_ret_at(bn, t, 30_000_000)
            bn_ret_60s[i] = _bn_ret_at(bn, t, 60_000_000)
            bn_flow_30s[i] = _bn_flow_at(bn, t, 30_000_000)

            dist = (bp - target) / target * 10000
            distance_bps[i] = dist

            vol_eff = bv if bv is not None and bv > 0.5 else 0.5
            t_eff = max(rem, 30.0)
            z = dist / (vol_eff * math.sqrt(t_eff))
            z_score_arr[i] = z
            fv = 0.5 * math.erfc(-z / math.sqrt(2))
            fair_value[i] = fv

            # UP Poly book
            bk_idx_i = np.searchsorted(bk_ts, t, side='right') - 1
            bk_idx_i = max(bi_lo, min(bk_idx_i, bi_hi - 1))
            pm = float(bd['mid'][bk_idx_i])
            pa_ = float(bd['best_ask'][bk_idx_i])
            pb = float(bd['best_bid'][bk_idx_i])
            ps = float(bd['spread'][bk_idx_i])
            poly_mid[i] = pm
            poly_ask[i] = pa_
            poly_bid[i] = pb
            poly_spread[i] = ps
            edge_arr[i] = fv - pm
            edge_vs_ask[i] = fv - pa_

            # DOWN Poly book
            if dn_bk_ts is not None and dn_bi_hi > dn_bi_lo:
                dn_idx = np.searchsorted(dn_bk_ts, t, side='right') - 1
                dn_idx = max(dn_bi_lo, min(dn_idx, dn_bi_hi - 1))
                dn_poly_ask[i] = float(bd_dn['best_ask'][dn_idx])

            # Poly flow (30s lookback) — UP token
            pf, pn = _poly_flow_at(up_ti, t, 30_000_000)
            poly_flow_arr[i] = pf
            poly_imb_arr[i] = pf  # simplified

            # DOWN token flow
            dnf, _ = _poly_flow_at(dn_ti, t, 30_000_000)
            dn_poly_flow_arr[i] = dnf

        timelines[slug] = {
            'coin': coin, 'outcome': w['outcome'], 'target': target,
            'start_us': start_us, 'end_us': end_us, 'total_s': total_s,
            'valid': valid,
            'snap_us': snap_times,
            'elapsed_s': elapsed_s, 'remaining_s': remaining_s, 'pct_elapsed': pct_elapsed,
            'bn_price': bn_price_arr, 'bn_vol': bn_vol_arr,
            'bn_ret_10s': bn_ret_10s, 'bn_ret_30s': bn_ret_30s, 'bn_ret_60s': bn_ret_60s,
            'bn_flow_30s': bn_flow_30s,
            'distance_bps': distance_bps, 'z_score': z_score_arr,
            'fair_value': fair_value,
            'poly_mid': poly_mid, 'poly_ask': poly_ask, 'poly_bid': poly_bid,
            'poly_spread': poly_spread,
            'edge': edge_arr, 'edge_vs_ask': edge_vs_ask,
            'poly_flow': poly_flow_arr,
            'dn_poly_ask': dn_poly_ask,
            'dn_poly_flow': dn_poly_flow_arr,
            # Book data for fills (UP and DOWN)
            '_up_book': bd, '_up_bk_ts': bk_ts,
            '_dn_book': bd_dn, '_dn_bk_ts': dn_bk_ts,
        }

    print(f"    Built {len(timelines)} timelines, skipped {n_skip}")
    return timelines


# ═══════════════════════════════════════════════════════════════════════
# Fill at timestamp — get actual ask price from book + apply fee
# ═══════════════════════════════════════════════════════════════════════

def _vwap_asks(ask_prices, ask_sizes, size_usdc):
    if not isinstance(ask_prices, (list, np.ndarray)) or len(ask_prices) == 0:
        return None
    rem, wsum, total = size_usdc, 0.0, 0.0
    for p, s in zip(ask_prices, ask_sizes):
        if rem <= 0: break
        s = float(s)
        if s <= 0: continue
        fill = min(rem, s)
        wsum += fill * float(p)
        total += fill
        rem -= fill
    return wsum / total if total > 0 else None

def get_fill_at(tl, snap_idx, direction="up", size_usdc=50.0):
    """Get taker fill price (with fee) at a specific snapshot index."""
    if direction == "up":
        bd = tl['_up_book']
        bk_ts = tl['_up_bk_ts']
    else:
        bd = tl.get('_dn_book')
        bk_ts = tl.get('_dn_bk_ts')
        if bd is None or bk_ts is None:
            return None

    t = tl['snap_us'][snap_idx]

    if bd.get('ask_prices') is None:
        # Fallback to best_ask from precomputed array
        if direction == "up":
            ask = tl['poly_ask'][snap_idx]
        else:
            ask = tl['dn_poly_ask'][snap_idx]
        if np.isnan(ask) or ask <= 0 or ask >= 1: return None
        return ask * (1 + taker_fee_rate(ask))

    bk_idx = np.searchsorted(bk_ts, t, side='right') - 1
    if bk_idx < 0: return None
    raw = _vwap_asks(bd['ask_prices'][bk_idx], bd['ask_sizes'][bk_idx], size_usdc)
    if raw is None: return None
    return raw * (1 + taker_fee_rate(raw))


# ═══════════════════════════════════════════════════════════════════════
# Strategy engine — each strategy is a function: timeline → entry index or None
# ═══════════════════════════════════════════════════════════════════════

def _make_trigger(name, trigger_fn, direction="up", min_elapsed_pct=5, max_elapsed_pct=80):
    """Wrap a trigger function with metadata."""
    return {
        'name': name,
        'trigger': trigger_fn,
        'direction': direction,
        'min_pct': min_elapsed_pct,
        'max_pct': max_elapsed_pct,
    }


def define_dynamic_strategies():
    """
    Each strategy returns the index of first snapshot where the trigger fires,
    or None if it never fires during the window.
    """
    strats = []

    # ── S1: Z-score breakout — enter when z > threshold ──────────
    for z_thr in [1.5, 2.0, 2.5, 3.0]:
        def make_s(zt):
            def trigger(tl):
                v = tl['valid']
                z = tl['z_score']
                pct = tl['pct_elapsed']
                for i in range(len(z)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if z[i] > zt: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S1  z>{z_thr:.1f} → BUY UP", make_s(z_thr), direction="up"))

    # ── S2: Z-score breakdown — enter when z < -threshold ────────
    for z_thr in [1.5, 2.0, 2.5]:
        def make_s(zt):
            def trigger(tl):
                v, z, pct = tl['valid'], tl['z_score'], tl['pct_elapsed']
                for i in range(len(z)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if z[i] < -zt: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S2  z<-{z_thr:.1f} → BUY DOWN", make_s(z_thr), direction="down"))

    # ── S3: Edge threshold — enter when fair_value - ask > X ─────
    for edge_thr in [0.03, 0.05, 0.08, 0.10]:
        def make_s(et):
            def trigger(tl):
                v, e, pct = tl['valid'], tl['edge_vs_ask'], tl['pct_elapsed']
                for i in range(len(e)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if e[i] > et: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S3  edge_vs_ask>{edge_thr:.2f} → BUY UP", make_s(edge_thr), direction="up"))

    # ── S4: BN shock + Poly lag — BN moves >X bps in 10s, Poly hasn't followed
    for bn_thr in [5, 8, 12]:
        def make_s(bt):
            def trigger(tl):
                v, ret, edge, pct = tl['valid'], tl['bn_ret_10s'], tl['edge_vs_ask'], tl['pct_elapsed']
                for i in range(len(ret)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if ret[i] > bt and edge[i] > 0.02: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S4  BN+{bn_thr}bps/10s + edge>2c → UP", make_s(bn_thr), direction="up"))

    # ── S5: BN shock DOWN + lag
    for bn_thr in [5, 8, 12]:
        def make_s(bt):
            def trigger(tl):
                v, ret, edge, pct = tl['valid'], tl['bn_ret_10s'], tl['edge'], tl['pct_elapsed']
                for i in range(len(ret)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if ret[i] < -bt and edge[i] < -0.02: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S5  BN-{bn_thr}bps/10s + edge<-2c → DN", make_s(bn_thr), direction="down"))

    # ── S6: Mean-reversion dip-buy — price below target + BN dropping + flow bullish
    for dist_thr in [-3, -5, -8]:
        def make_s(dt):
            def trigger(tl):
                v, dist, ret, flow, pct = tl['valid'], tl['distance_bps'], tl['bn_ret_30s'], tl['poly_flow'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 10 or pct[i] > 70: continue
                    if dist[i] < dt and ret[i] < 0 and flow[i] > 0.1: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S6  dip-buy dist<{dist_thr} + ret<0 + flow>0", make_s(dist_thr), direction="up"))

    # ── S7: Distance sustained — price above target for 3+ consecutive snapshots
    for dist_thr in [3, 5, 7]:
        def make_s(dt):
            def trigger(tl):
                v, dist, pct = tl['valid'], tl['distance_bps'], tl['pct_elapsed']
                consec = 0
                for i in range(len(dist)):
                    if not v[i]: consec = 0; continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if dist[i] > dt:
                        consec += 1
                        if consec >= 3: return i
                    else:
                        consec = 0
                return None
            return trigger
        strats.append(_make_trigger(f"S7  dist>{dist_thr} sustained 15s → UP", make_s(dist_thr), direction="up"))

    # ── S8: Edge + low spread — tradable edge with tight market
    for edge_thr in [0.03, 0.05]:
        def make_s(et):
            def trigger(tl):
                v, e, sp, pct = tl['valid'], tl['edge_vs_ask'], tl['poly_spread'], tl['pct_elapsed']
                for i in range(len(e)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if e[i] > et and sp[i] < 0.02: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S8  edge>{edge_thr:.2f} + spread<2c → UP", make_s(edge_thr), direction="up"))

    # ── S9: Late window certainty — z-score high late in window
    for z_thr in [2.0, 3.0]:
        def make_s(zt):
            def trigger(tl):
                v, z, pct = tl['valid'], tl['z_score'], tl['pct_elapsed']
                for i in range(len(z)):
                    if not v[i]: continue
                    if pct[i] < 50 or pct[i] > 90: continue
                    if z[i] > zt: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S9  late z>{z_thr:.1f} (50-90%) → UP", make_s(z_thr), direction="up"))

    # ── S10: Momentum + distance convergence
    for dist_thr in [3, 5]:
        def make_s(dt):
            def trigger(tl):
                v, dist, ret, flow, pct = tl['valid'], tl['distance_bps'], tl['bn_ret_30s'], tl['bn_flow_30s'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 75: continue
                    if dist[i] > dt and ret[i] > 0 and flow[i] > 0: return i
                return None
            return trigger
        strats.append(_make_trigger(f"S10  dist>{dist_thr}+ret>0+flow>0 → UP", make_s(dist_thr), direction="up"))

    # ── S11: BN flow reversal — flow flips positive while below target
    def make_s11():
        def trigger(tl):
            v, dist, flow, pct = tl['valid'], tl['distance_bps'], tl['bn_flow_30s'], tl['pct_elapsed']
            prev_flow = 0
            for i in range(len(dist)):
                if not v[i]: continue
                if pct[i] < 10 or pct[i] > 70: continue
                if dist[i] < 0 and flow[i] > 0.1 and prev_flow < -0.05: return i
                prev_flow = flow[i]
            return None
        return trigger
    strats.append(_make_trigger("S11  flow reversal (neg→pos) while dist<0 → UP", make_s11(), direction="up"))

    # ── S12: Benchmark — always enter at 5% elapsed
    def make_bench_up():
        def trigger(tl):
            v, pct = tl['valid'], tl['pct_elapsed']
            for i in range(len(pct)):
                if v[i] and pct[i] >= 5: return i
            return None
        return trigger
    strats.append(_make_trigger("S0  benchmark: always UP at 5%", make_bench_up(), direction="up"))

    # ── DC1: Dynamic C3 — dist>=5 + poly_flow>0.2 at any moment ─────
    for dist_thr in [3, 5, 7, 10]:
        def make_dc(dt):
            def trigger(tl):
                v, dist, flow, pct = tl['valid'], tl['distance_bps'], tl['poly_flow'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if dist[i] > dt and flow[i] > 0.2: return i
                return None
            return trigger
        strats.append(_make_trigger(f"DC1  dist>{dist_thr} + flow>0.2 → UP", make_dc(dist_thr), direction="up"))

    # ── DC2: Dynamic C3 + BN flow aligned ────────────────────────────
    for dist_thr in [3, 5]:
        def make_dc2(dt):
            def trigger(tl):
                v, dist, pflow, bflow, pct = tl['valid'], tl['distance_bps'], tl['poly_flow'], tl['bn_flow_30s'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if dist[i] > dt and pflow[i] > 0.2 and bflow[i] > 0: return i
                return None
            return trigger
        strats.append(_make_trigger(f"DC2  dist>{dist_thr} + flow>0.2 + BN aligned → UP", make_dc2(dist_thr), direction="up"))

    # ── DC3: Dynamic dip-buy — dist drops below threshold + poly flow bullish
    for dist_thr in [-3, -5]:
        def make_dc3(dt):
            def trigger(tl):
                v, dist, pflow, ret, pct = tl['valid'], tl['distance_bps'], tl['poly_flow'], tl['bn_ret_30s'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 10 or pct[i] > 70: continue
                    if dist[i] < dt and pflow[i] > 0.2 and ret[i] < 0: return i
                return None
            return trigger
        strats.append(_make_trigger(f"DC3  dip dist<{dist_thr} + flow>0.2 + ret<0 → UP", make_dc3(dist_thr), direction="up"))

    # ── DC4: Dynamic DOWN — dist drops below threshold + poly DOWN flow bullish
    for dist_thr in [-3, -5, -8]:
        def make_dc4(dt):
            def trigger(tl):
                v, dist, dnflow, pct = tl['valid'], tl['distance_bps'], tl['dn_poly_flow'], tl['pct_elapsed']
                for i in range(len(dist)):
                    if not v[i]: continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if dist[i] < dt and dnflow[i] > 0.2: return i
                return None
            return trigger
        strats.append(_make_trigger(f"DC4  dist<{dist_thr} + DN flow>0.2 → BUY DOWN", make_dc4(dist_thr), direction="down"))

    # ── DC5: Dynamic distance sustained + flow ───────────────────────
    for dist_thr in [3, 5]:
        def make_dc5(dt):
            def trigger(tl):
                v, dist, pflow, pct = tl['valid'], tl['distance_bps'], tl['poly_flow'], tl['pct_elapsed']
                consec = 0
                for i in range(len(dist)):
                    if not v[i]: consec = 0; continue
                    if pct[i] < 5 or pct[i] > 80: continue
                    if dist[i] > dt and pflow[i] > 0.1:
                        consec += 1
                        if consec >= 3: return i
                    else:
                        consec = 0
                return None
            return trigger
        strats.append(_make_trigger(f"DC5  dist>{dist_thr}+flow sustained 15s → UP", make_dc5(dist_thr), direction="up"))

    return strats


# ═══════════════════════════════════════════════════════════════════════
# Run strategies on timelines
# ═══════════════════════════════════════════════════════════════════════

def run_strategies(timelines, strategies, size_usdc=50.0, coins_filter=None):
    """
    For each strategy × each window, find the trigger point and compute PnL.
    Returns dict: strategy_name → list of trades.
    """
    all_results = {s['name']: [] for s in strategies}

    for slug, tl in timelines.items():
        if coins_filter and tl['coin'] not in coins_filter:
            continue

        outcome = tl['outcome']

        for strat in strategies:
            trigger_fn = strat['trigger']
            direction = strat['direction']

            idx = trigger_fn(tl)
            if idx is None:
                continue

            entry_price = get_fill_at(tl, idx, direction=direction, size_usdc=size_usdc)
            if entry_price is None or entry_price <= 0 or entry_price >= 1:
                continue

            if direction == "up":
                settle = 1.0 if outcome == 1 else 0.0
            else:
                settle = 1.0 if outcome == 0 else 0.0

            pnl = (settle / entry_price - 1.0) * size_usdc

            all_results[strat['name']].append({
                'slug': slug, 'coin': tl['coin'], 'outcome': outcome,
                'direction': direction, 'pnl': pnl,
                'entry': entry_price, 'settle': settle,
                'start_us': tl['start_us'],
                'entry_pct': float(tl['pct_elapsed'][idx]),
                'entry_dist': float(tl['distance_bps'][idx]),
                'entry_z': float(tl['z_score'][idx]),
                'entry_edge': float(tl['edge'][idx]),
                'entry_fv': float(tl['fair_value'][idx]),
            })

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Stats & reporting (reuse from v4)
# ═══════════════════════════════════════════════════════════════════════

def stats(results):
    if not results:
        return {'n':0,'pnl':0,'avg':0,'wr':0,'sharpe':0,'dd':0}
    pnl = np.array([r['pnl'] for r in results])
    n = len(pnl)
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {'n': n, 'pnl': round(float(pnl.sum()),1), 'avg': round(float(pnl.mean()),2),
            'wr': round(int((pnl>0).sum())/n*100,1),
            'sharpe': round(float(pnl.mean()/pnl.std()),3) if pnl.std()>0 else 0,
            'dd': round(dd,1)}

def _daily_map(results):
    by_date = defaultdict(list)
    for t in results:
        dt = datetime.fromtimestamp(t['start_us']/1e6, tz=timezone.utc)
        by_date[dt.strftime("%Y-%m-%d")].append(t)
    return by_date

def print_card(label, results, all_dates, coins):
    s = stats(results)
    if s['n'] == 0: return
    print(f"\n  ┌─ {label}")
    print(f"  │  N={s['n']}  PnL=${s['pnl']:+.1f}  avg=${s['avg']:+.2f}  "
          f"WR={s['wr']:.1f}%  Sharpe={s['sharpe']:+.3f}  DD=${s['dd']:.1f}")

    # Entry stats
    entry_pcts = [r['entry_pct'] for r in results]
    entry_dists = [r['entry_dist'] for r in results]
    entry_zs = [r['entry_z'] for r in results]
    print(f"  │  Avg entry: {np.mean(entry_pcts):.0f}% elapsed | dist={np.mean(entry_dists):+.1f}bps | z={np.mean(entry_zs):+.2f}")

    by_date = _daily_map(results)
    print(f"  │")
    print(f"  │  {'date':>12s}  {'N':>4s}  {'PnL':>9s}  {'avg':>8s}  {'WR':>5s}")
    print(f"  │  {'─'*44}")
    for date in sorted(all_dates):
        day = by_date.get(date, [])
        if not day: print(f"  │  {date:>12s}  {'—':>4s}"); continue
        dp = np.array([t['pnl'] for t in day])
        dw = int((dp>0).sum())
        print(f"  │  {date:>12s}  {len(dp):>4d}  ${dp.sum():>+8.1f}  ${dp.mean():>+7.2f}  {dw/len(dp)*100:>4.0f}%")

    print(f"  │")
    print(f"  │  {'coin':>6s}  {'N':>4s}  {'PnL':>9s}  {'avg':>8s}  {'WR':>5s}  {'Sharpe':>7s}")
    print(f"  │  {'─'*50}")
    for coin in coins:
        ct = [r for r in results if r['coin'] == coin]
        if not ct: continue
        cs = stats(ct)
        print(f"  │  {coin.upper():>6s}  {cs['n']:>4d}  ${cs['pnl']:>+8.1f}  ${cs['avg']:>+7.2f}  {cs['wr']:>4.1f}%  {cs['sharpe']:>+6.3f}")
    print(f"  └{'─'*60}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(events_dir, coins_filter=None):
    print(f"\n{'═'*74}")
    print(f"  DYNAMIC BACKTEST — continuous scan, trigger-based entries")
    print(f"  All PnL includes taker fee: 0.072 × p × (1−p)")
    print(f"{'═'*74}")

    t_total = time.time()
    print(f"\n  Loading data...")
    resolved = load_markets(events_dir)
    coins = coins_filter if coins_filter else sorted(set(w['coin'] for w in resolved.values()))
    resolved = {s: w for s, w in resolved.items() if w['coin'] in coins}

    tokens = set()
    for w in resolved.values():
        tokens.add(w['up_token']); tokens.add(w['down_token'])

    books = load_books(events_dir, tokens)
    trades = load_trades(events_dir, tokens)
    bn_by_symbol = load_bn_trades(events_dir)

    # Build timelines
    print(f"\n  Building timelines (5s resolution)...")
    t_tl = time.time()
    timelines = build_timelines(resolved, books, trades, bn_by_symbol, interval_filter=15)
    print(f"    Done in {time.time()-t_tl:.1f}s")

    target_coins = [c for c in ['btc','eth','sol'] if c in coins]
    all_dates = sorted(set(datetime.fromtimestamp(tl['start_us']/1e6, tz=timezone.utc).strftime("%Y-%m-%d")
                           for tl in timelines.values()))

    print(f"\n  Coins: {', '.join(c.upper() for c in target_coins)}")
    print(f"  Dates: {', '.join(all_dates)} ({len(all_dates)} days)")
    print(f"  Windows with timelines: {len(timelines)}")

    # Run strategies
    print(f"\n  Running strategies...")
    strategies = define_dynamic_strategies()
    t_run = time.time()
    all_results = run_strategies(timelines, strategies, coins_filter=target_coins)
    print(f"    {len(strategies)} strategies evaluated in {time.time()-t_run:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Grid
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  STRATEGY GRID")
    print(f"{'═'*90}")
    print(f"\n  {'':>48s}  {'N':>4s}  {'PnL':>9s}  {'avg':>7s}  {'WR':>5s}  {'Sharpe':>7s}  {'DD':>9s}")
    print(f"  {'─'*95}")

    for strat in strategies:
        r = all_results[strat['name']]
        s = stats(r)
        if s['n'] == 0:
            print(f"  {strat['name']:>48s}  {'—':>4s}")
        else:
            print(f"  {strat['name']:>48s}  {s['n']:>4d}  ${s['pnl']:>+8.1f}  ${s['avg']:>+6.2f}  {s['wr']:>4.1f}%  {s['sharpe']:>+6.3f}  ${s['dd']:>+8.1f}")

    # Rankings
    ranked = [(name, stats(r)) for name, r in all_results.items() if stats(r)['n'] >= 10]
    by_sharpe = sorted(ranked, key=lambda x: x[1]['sharpe'], reverse=True)
    by_pnl = sorted(ranked, key=lambda x: x[1]['pnl'], reverse=True)

    print(f"\n{'═'*74}")
    print(f"  TOP 5 BY SHARPE (N≥10)")
    print(f"{'═'*74}")
    for name, s in by_sharpe[:5]:
        print(f"    {s['sharpe']:>+6.3f}  {name}  N={s['n']} PnL=${s['pnl']:+.1f} WR={s['wr']}%")

    print(f"\n{'═'*74}")
    print(f"  TOP 5 BY PnL (N≥10)")
    print(f"{'═'*74}")
    for name, s in by_pnl[:5]:
        print(f"    ${s['pnl']:>+8.1f}  {name}  N={s['n']} Sharpe={s['sharpe']:+.3f} WR={s['wr']}%")

    # ═══════════════════════════════════════════════════════════════
    # Cards for top strategies
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  DETAILED CARDS")
    print(f"{'═'*74}")

    shown = set()
    cards = []
    for name, _ in by_sharpe[:5]:
        if name not in shown: cards.append(name); shown.add(name)
    for name, s in ranked:
        if s['sharpe'] > 0.05 and name not in shown: cards.append(name); shown.add(name)
    # Benchmark
    bench = "S0  benchmark: always UP at 5%"
    if bench not in shown: cards.append(bench); shown.add(bench)

    for name in cards:
        print_card(name, all_results.get(name, []), all_dates, target_coins)

    print(f"\n{'═'*74}")
    print(f"  DONE in {time.time()-t_total:.0f}s")
    print(f"{'═'*74}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('--coin', nargs='+', help='Filter coins')
    args = parser.parse_args()
    main(args.events_dir, args.coin)