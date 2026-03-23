#!/usr/bin/env python3
"""
backtest_allcoins.py — Run flow strategy eval across ALL coins.

Loads all data once, then evaluates per coin + cross-coin.
Focused on the variants that matter (trimmed from v2 bloat).

Usage:
  python backtest_allcoins.py historical_data/events
  python backtest_allcoins.py historical_data/events --coin btc eth
"""

import sys, os, glob, argparse, time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))

COIN_TO_SYMBOL = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}


# ═══════════════════════════════════════════════════════════════════════
# Data loading — all coins at once
# ═══════════════════════════════════════════════════════════════════════

def load_all_markets(events_dir):
    """Load all resolved windows, grouped by coin."""
    mdir = os.path.join(events_dir, 'poly_markets')
    files = sorted(glob.glob(os.path.join(mdir, '*.parquet')))
    if not files:
        return {}, pd.DataFrame()
    mdf = pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()

    windows = {}
    for _, row in mdf.iterrows():
        slug = row['slug']
        if slug not in windows:
            windows[slug] = {
                'slug': slug, 'coin': row['coin'],
                'interval': int(row['interval_min']),
                'up_token': row['up_token_id'], 'down_token': row['down_token_id'],
                'start_us': int(row['window_start_us']), 'end_us': int(row['window_end_us']),
                'target_price': float(row.get('target_price', 0)),
                'outcome': -1,
            }
        if int(row.get('outcome', -1)) in (0, 1):
            windows[slug]['outcome'] = int(row['outcome'])

    resolved = {s: w for s, w in windows.items() if w['outcome'] in (0, 1)}
    return resolved, mdf


def load_books_and_trades(events_dir, token_ids):
    """Load books and trades filtered to a set of token IDs."""
    t0 = time.time()
    token_set = set(token_ids)

    # Books
    bdir = os.path.join(events_dir, 'poly_books')
    chunks = []
    for f in sorted(glob.glob(os.path.join(bdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(token_set)]
        if len(df) > 0:
            chunks.append(df)
    books = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(books) > 0:
        books = books[(books['best_bid'] > 0) & (books['best_ask'] > 0)].copy()
        books['mid'] = (books['best_bid'] + books['best_ask']) / 2
        books['spread_bps'] = (books['best_ask'] - books['best_bid']) * 10000
        def total_size(col):
            return col.apply(lambda x: sum(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0)
        books['bid_depth'] = total_size(books['bid_sizes'])
        books['ask_depth'] = total_size(books['ask_sizes'])
        books['imbalance'] = (books['bid_depth'] - books['ask_depth']) / (books['bid_depth'] + books['ask_depth']).clip(lower=1)

    # Trades
    tdir = os.path.join(events_dir, 'poly_trades')
    tchunks = []
    for f in sorted(glob.glob(os.path.join(tdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(token_set)]
        if len(df) > 0:
            tchunks.append(df)
    trades = pd.concat(tchunks, ignore_index=True).sort_values('ts_us') if tchunks else pd.DataFrame()

    print(f"    Books: {len(books):,} | Trades: {len(trades):,} | {time.time()-t0:.1f}s")
    return books, trades


def load_bn_trades(events_dir):
    """Load Binance trades, return indexed by symbol."""
    t0 = time.time()
    bdir = os.path.join(events_dir, 'bn_trades')
    files = sorted(glob.glob(os.path.join(bdir, '*.parquet')))
    if not files:
        print(f"    No bn_trades data found")
        return {}

    chunks = []
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
            if len(df) > 0:
                chunks.append(df)
        except Exception:
            pass

    if not chunks:
        return {}

    bn = pd.concat(chunks, ignore_index=True).sort_values('ts_us')
    print(f"    Binance trades: {len(bn):,} rows | {time.time()-t0:.1f}s")

    # Index by symbol: sorted arrays
    bn_by_symbol = {}
    for sym, grp in bn.groupby('symbol'):
        grp = grp.sort_values('ts_us')
        bn_by_symbol[sym] = {
            'ts_us': grp['ts_us'].values,
            'price': grp['price'].values.astype(np.float64),
            'qty': grp['qty'].values.astype(np.float64),
            'is_buy': grp['is_buy'].values,
        }
    return bn_by_symbol


def _bn_features_at(bn_data, t_us):
    """
    Compute Binance features at a given timestamp.
    Returns dict with momentum, volatility, flow features.
    """
    if bn_data is None:
        return {}

    ts = bn_data['ts_us']
    px = bn_data['price']
    qty = bn_data['qty']
    is_buy = bn_data['is_buy']

    # Current price: last trade at or before t_us
    idx_now = np.searchsorted(ts, t_us, side='right') - 1
    if idx_now < 0:
        return {}
    price_now = float(px[idx_now])

    features = {'bn_price': price_now}

    # Momentum: return over last N seconds
    for label, lookback_us in [('30s', 30_000_000), ('60s', 60_000_000),
                                ('2m', 120_000_000), ('5m', 300_000_000)]:
        t_past = t_us - lookback_us
        idx_past = np.searchsorted(ts, t_past, side='right') - 1
        if idx_past >= 0:
            price_past = float(px[idx_past])
            if price_past > 0:
                features[f'bn_ret_{label}'] = (price_now / price_past - 1) * 10000  # bps
            else:
                features[f'bn_ret_{label}'] = 0
        else:
            features[f'bn_ret_{label}'] = 0

    # Volatility: std of log returns over last N seconds, using 1s buckets
    for label, lookback_us in [('1m', 60_000_000), ('5m', 300_000_000)]:
        t_past = t_us - lookback_us
        i_lo = np.searchsorted(ts, t_past, side='left')
        i_hi = np.searchsorted(ts, t_us, side='right')
        if i_hi - i_lo > 10:
            prices_window = px[i_lo:i_hi]
            # Sample every ~100 trades for efficiency
            step = max(1, len(prices_window) // 200)
            sampled = prices_window[::step].astype(np.float64)
            if len(sampled) > 2:
                log_returns = np.diff(np.log(sampled))
                features[f'bn_vol_{label}'] = float(np.std(log_returns) * 10000)  # bps
            else:
                features[f'bn_vol_{label}'] = 0
        else:
            features[f'bn_vol_{label}'] = 0

    # Flow: buy/sell imbalance over last 30s
    for label, lookback_us in [('30s', 30_000_000), ('60s', 60_000_000)]:
        t_past = t_us - lookback_us
        i_lo = np.searchsorted(ts, t_past, side='left')
        i_hi = np.searchsorted(ts, t_us, side='right')
        if i_hi > i_lo:
            window_qty = qty[i_lo:i_hi]
            window_buy = is_buy[i_lo:i_hi]
            buy_vol = float(window_qty[window_buy].sum())
            sell_vol = float(window_qty[~window_buy].sum())
            total = buy_vol + sell_vol
            features[f'bn_flow_{label}'] = (buy_vol - sell_vol) / total if total > 0 else 0
        else:
            features[f'bn_flow_{label}'] = 0

    return features


# ═══════════════════════════════════════════════════════════════════════
# Fill caching + signal computation (same as backtest_btc_v2)
# ═══════════════════════════════════════════════════════════════════════

def _vwap_from_asks(ask_prices, ask_sizes, size_usdc):
    if not isinstance(ask_prices, (list, np.ndarray)) or len(ask_prices) == 0:
        return None
    remaining = size_usdc
    weighted_sum = 0.0
    total_filled = 0.0
    for p, s in zip(ask_prices, ask_sizes):
        if remaining <= 0:
            break
        s = float(s)
        if s <= 0:
            continue
        fill = min(remaining, s)
        weighted_sum += fill * float(p)
        total_filled += fill
        remaining -= fill
    return weighted_sum / total_filled if total_filled > 0 else None


def cache_fills(resolved, books, size_usdc=50.0):
    token_index = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        token_index[tid] = (
            grp['ts_us'].values,
            grp['ask_prices'].values,
            grp['ask_sizes'].values,
        )

    entry_offsets = [0, 15, 30, 45, 60]
    fills = {}

    for slug, w in resolved.items():
        row = {'outcome': w['outcome'], 'interval': w['interval']}
        for side, token in [('up', w['up_token']), ('down', w['down_token'])]:
            idx_data = token_index.get(token)
            if idx_data is None:
                continue
            ts_arr, ask_p_arr, ask_s_arr = idx_data
            for offset in entry_offsets:
                entry_us = w['start_us'] + offset * 1_000_000
                if entry_us >= w['end_us']:
                    continue
                pos = np.searchsorted(ts_arr, entry_us, side='left')
                deadline_us = min(entry_us + 15_000_000, w['end_us'])
                price = None
                while pos < len(ts_arr) and ts_arr[pos] <= deadline_us:
                    price = _vwap_from_asks(ask_p_arr[pos], ask_s_arr[pos], size_usdc)
                    if price is not None:
                        break
                    pos += 1
                row[f'{side}_fill_{offset}s'] = price
        fills[slug] = row
    return fills


def compute_all_signals(resolved, books, trades, bn_data=None):
    """Compute flow + imbalance on UP and DOWN tokens + Binance features.
    
    bn_data: optional dict from bn_by_symbol for this coin's Binance symbol.
    """
    books_by_token = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        books_by_token[tid] = {
            'ts_us': grp['ts_us'].values,
            'mid': grp['mid'].values,
            'spread_bps': grp['spread_bps'].values,
            'imbalance': grp['imbalance'].values,
        }

    trades_by_token = {}
    if len(trades) > 0:
        for tid, grp in trades.groupby('token_id'):
            grp = grp.sort_values('ts_us')
            trades_by_token[tid] = {
                'ts_us': grp['ts_us'].values,
                'price': grp['price'].values.astype(np.float64),
                'size': grp['size'].values.astype(np.float64),
            }

    trade_is_buy = {}
    for tid, tdata in trades_by_token.items():
        bdata = books_by_token.get(tid)
        if bdata is None or len(bdata['ts_us']) == 0:
            trade_is_buy[tid] = tdata['price'] > 0
            continue
        idx = np.searchsorted(bdata['ts_us'], tdata['ts_us'], side='right') - 1
        idx = np.clip(idx, 0, len(bdata['mid']) - 1)
        trade_is_buy[tid] = tdata['price'] >= bdata['mid'][idx]

    def _token_flow_imb(token_id, t_start, obs_us):
        """Compute flow + imbalance for any token."""
        bdata = books_by_token.get(token_id)
        if bdata is None or len(bdata['ts_us']) == 0:
            return 0, 0, 0, 0, 0  # flow, imb, mid, vol, n_trades
        bk_ts = bdata['ts_us']
        i_lo = np.searchsorted(bk_ts, t_start, side='left')
        i_hi = np.searchsorted(bk_ts, t_start + 10_000_000, side='right')
        if i_lo >= i_hi:
            return 0, 0, 0, 0, 0
        mid = float(bdata['mid'][i_lo])
        imb = float(np.mean(bdata['imbalance'][i_lo:i_hi]))

        tdata = trades_by_token.get(token_id)
        if tdata is None or len(tdata['ts_us']) == 0:
            return 0, imb, mid, 0, 0
        t_ts = tdata['ts_us']
        j_lo = np.searchsorted(t_ts, t_start, side='left')
        j_hi = np.searchsorted(t_ts, t_start + obs_us, side='right')
        n_trades = j_hi - j_lo
        if n_trades == 0:
            return 0, imb, mid, 0, 0
        sizes = tdata['size'][j_lo:j_hi]
        is_buy = trade_is_buy.get(token_id)
        if is_buy is None:
            return 0, imb, mid, 0, n_trades
        is_buy_slice = is_buy[j_lo:j_hi]
        buy_vol = float(sizes[is_buy_slice].sum())
        sell_vol = float(sizes[~is_buy_slice].sum())
        total = buy_vol + sell_vol
        flow = (buy_vol - sell_vol) / total if total > 0 else 0
        return flow, imb, mid, total, n_trades

    all_signals = {}
    for obs_secs in [15, 30, 45, 60]:
        sigs = {}
        obs_us = obs_secs * 1_000_000
        for slug, w in resolved.items():
            up_token = w['up_token']
            dn_token = w['down_token']
            t_start = w['start_us']
            t_entry = t_start + obs_us

            # UP token signals
            up_flow, up_imb, up_mid, up_vol, up_ntrades = _token_flow_imb(up_token, t_start, obs_us)
            if up_mid == 0:
                continue  # no book data at all

            bdata = books_by_token.get(up_token)
            spread_start = 0
            if bdata is not None:
                bk_ts = bdata['ts_us']
                i_lo = np.searchsorted(bk_ts, t_start, side='left')
                i_hi = np.searchsorted(bk_ts, t_start + 10_000_000, side='right')
                if i_hi > i_lo:
                    spread_start = float(np.median(bdata['spread_bps'][i_lo:i_hi]))

            # DOWN token signals
            dn_flow, dn_imb, dn_mid, dn_vol, dn_ntrades = _token_flow_imb(dn_token, t_start, obs_us)

            sig = {
                'flow': up_flow, 'imbalance': up_imb,
                'mid_start': up_mid, 'spread_start': spread_start,
                'total_vol': up_vol, 'n_trades': up_ntrades,
                # DOWN token
                'dn_flow': dn_flow, 'dn_imbalance': dn_imb,
                'dn_mid': dn_mid, 'dn_vol': dn_vol, 'dn_n_trades': dn_ntrades,
            }

            # Binance features at entry time
            if bn_data is not None:
                bn_feat = _bn_features_at(bn_data, t_entry)
                sig.update(bn_feat)

                target = w.get('target_price', 0)
                if target and bn_feat.get('bn_price'):
                    sig['bn_distance_bps'] = (bn_feat['bn_price'] - target) / target * 10000
                else:
                    sig['bn_distance_bps'] = 0
            else:
                for k in ['bn_price', 'bn_ret_30s', 'bn_ret_60s', 'bn_ret_2m', 'bn_ret_5m',
                           'bn_vol_1m', 'bn_vol_5m', 'bn_flow_30s', 'bn_flow_60s',
                           'bn_distance_bps']:
                    sig[k] = 0

            sigs[slug] = sig
        all_signals[obs_secs] = sigs
    return all_signals


# ═══════════════════════════════════════════════════════════════════════
# Evaluate + report (same as backtest_btc_v2)
# ═══════════════════════════════════════════════════════════════════════

def evaluate(resolved, fills, signals, obs_secs=30, flow_threshold=0.0,
             use_imbalance=False, interval_filter=None, min_volume=0,
             min_n_trades=0, mid_range=None, size_usdc=50.0,
             # Direction override
             force_direction=None,       # "up", "down", or None (use flow/imb)
             # Binance filters
             bn_momentum_range=None,     # (lo, hi) for bn_ret_30s
             bn_vol_max=None,            # max bn_vol_1m
             bn_vol_min=None,            # min bn_vol_1m
             bn_flow_align=False,        # require bn_flow same sign as direction
             bn_distance_max=None,       # max abs(bn_distance_bps)
             bn_distance_min=None,       # min bn_distance_bps (signed)
             bn_distance_max_signed=None,# max bn_distance_bps (signed)
             # DOWN token filters
             dn_flow_min=None,           # min DOWN token flow (positive = people buying DOWN)
             dn_imb_max=None,            # max DOWN token imbalance (negative = ask pressure)
             ):
    sigs = signals.get(obs_secs, {})
    results = []
    for slug, w in resolved.items():
        if slug not in sigs or slug not in fills:
            continue
        if interval_filter and w['interval'] != interval_filter:
            continue
        sig = sigs[slug]
        f = fills[slug]
        flow = sig['flow']
        imb = sig['imbalance']

        # Flow threshold (skip if no signal) — bypassed when force_direction
        if force_direction is None:
            if abs(flow) < flow_threshold:
                continue

        if sig['total_vol'] < min_volume:
            continue
        if sig['n_trades'] < min_n_trades:
            continue
        if mid_range and not (mid_range[0] <= sig['mid_start'] <= mid_range[1]):
            continue

        # Binance filters
        if bn_momentum_range is not None:
            ret = sig.get('bn_ret_30s', 0)
            if not (bn_momentum_range[0] <= ret <= bn_momentum_range[1]):
                continue
        if bn_vol_min is not None and sig.get('bn_vol_1m', 0) < bn_vol_min:
            continue
        if bn_vol_max is not None and sig.get('bn_vol_1m', 0) > bn_vol_max:
            continue
        if bn_distance_max is not None and abs(sig.get('bn_distance_bps', 0)) > bn_distance_max:
            continue
        if bn_distance_min is not None and sig.get('bn_distance_bps', 0) < bn_distance_min:
            continue
        if bn_distance_max_signed is not None and sig.get('bn_distance_bps', 0) > bn_distance_max_signed:
            continue

        # DOWN token filters
        if dn_flow_min is not None and sig.get('dn_flow', 0) < dn_flow_min:
            continue
        if dn_imb_max is not None and sig.get('dn_imbalance', 0) > dn_imb_max:
            continue

        # Determine direction
        if force_direction is not None:
            direction = force_direction
        elif use_imbalance:
            if flow > 0 and imb > 0:
                direction = "up"
            elif flow < 0 and imb < 0:
                direction = "down"
            else:
                continue
        else:
            direction = "up" if flow > 0 else "down"

        # Binance flow alignment
        if bn_flow_align:
            bn_fl = sig.get('bn_flow_30s', 0)
            if direction == "up" and bn_fl <= 0:
                continue
            if direction == "down" and bn_fl >= 0:
                continue

        fill_key = f'{direction}_fill_{obs_secs}s'
        entry = f.get(fill_key)
        if entry is None:
            continue
        if direction == "up":
            settle = 1.0 if w['outcome'] == 1 else 0.0
        else:
            settle = 1.0 if w['outcome'] == 0 else 0.0
        pnl = (settle / entry - 1.0) * size_usdc if entry > 0 else 0.0
        results.append({
            'slug': slug, 'coin': w['coin'], 'interval': w['interval'],
            'outcome': w['outcome'], 'direction': direction,
            'flow': flow, 'imbalance': imb,
            'entry': entry, 'settle': settle, 'pnl': pnl,
            'mid_start': sig['mid_start'], 'spread': sig['spread_start'],
            'total_vol': sig['total_vol'], 'start_us': w['start_us'],
            'bn_ret_30s': sig.get('bn_ret_30s', 0),
            'bn_vol_1m': sig.get('bn_vol_1m', 0),
            'bn_flow_30s': sig.get('bn_flow_30s', 0),
            'bn_distance_bps': sig.get('bn_distance_bps', 0),
            'dn_flow': sig.get('dn_flow', 0),
            'dn_imbalance': sig.get('dn_imbalance', 0),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════

def stats(results):
    """Compute summary stats from a list of trade results."""
    if not results:
        return {'n': 0, 'pnl': 0, 'avg': 0, 'wr': 0, 'sharpe': 0, 'dd': 0, 'up': 0, 'dn': 0}
    pnl = np.array([r['pnl'] for r in results])
    n = len(pnl)
    wins = int((pnl > 0).sum())
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {
        'n': n, 'pnl': round(float(pnl.sum()), 1),
        'avg': round(float(pnl.mean()), 2),
        'wr': round(wins / n * 100, 1),
        'sharpe': round(float(pnl.mean() / pnl.std()), 3) if pnl.std() > 0 else 0,
        'dd': round(dd, 1),
        'up': sum(1 for r in results if r['direction'] == 'up'),
        'dn': sum(1 for r in results if r['direction'] == 'down'),
    }


def row(label, s, width=42):
    """Format one stats line."""
    if s['n'] == 0:
        return f"  {label:>{width}s}:  —"
    return (f"  {label:>{width}s}:  N={s['n']:>3d}  PnL=${s['pnl']:>+8.1f}  "
            f"avg=${s['avg']:>+6.2f}  WR={s['wr']:>4.1f}%  "
            f"Sharpe={s['sharpe']:>+6.3f}  DD=${s['dd']:>+8.1f}")


def section(title):
    print(f"\n  {'─'*64}")
    print(f"  {title}")
    print(f"  {'─'*64}")


def daily(results):
    """Print daily breakdown."""
    if not results:
        return
    by_date = defaultdict(list)
    for t in results:
        dt = datetime.fromtimestamp(t['start_us'] / 1e6, tz=timezone.utc)
        by_date[dt.strftime("%Y-%m-%d")].append(t)
    for date in sorted(by_date):
        day = by_date[date]
        pnl = [t['pnl'] for t in day]
        wins = sum(1 for p in pnl if p > 0)
        wr = wins / len(pnl) * 100 if pnl else 0
        print(f"    {date}: {len(pnl):>3d} trades  PnL=${sum(pnl):>+8.1f}  "
              f"avg=${np.mean(pnl):>+6.2f}  WR={wr:.0f}%")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(events_dir, coins_filter=None):
    print(f"\n{'═'*70}")
    print(f"  POLYMARKET FLOW BACKTEST")
    print(f"{'═'*70}")

    t_total = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────
    print(f"\n  Loading data...")
    all_windows, mdf = load_all_markets(events_dir)
    available_coins = sorted(set(w['coin'] for w in all_windows.values()))
    coins = coins_filter if coins_filter else available_coins

    all_tokens = set()
    coin_windows = {}
    for coin in coins:
        cw = {s: w for s, w in all_windows.items() if w['coin'] == coin}
        coin_windows[coin] = cw
        for w in cw.values():
            all_tokens.add(w['up_token'])
            all_tokens.add(w['down_token'])

    books, trades = load_books_and_trades(events_dir, all_tokens)
    bn_by_symbol = load_bn_trades(events_dir)

    # ── 2. Compute signals per coin ───────────────────────────────────
    coin_data = {}  # coin → (resolved, fills, signals)
    for coin in coins:
        resolved = coin_windows[coin]
        if not resolved:
            continue
        coin_tokens = set()
        for w in resolved.values():
            coin_tokens.add(w['up_token'])
            coin_tokens.add(w['down_token'])
        cb = books[books['token_id'].isin(coin_tokens)] if len(books) > 0 else books
        ct = trades[trades['token_id'].isin(coin_tokens)] if len(trades) > 0 else trades
        bn = bn_by_symbol.get(COIN_TO_SYMBOL.get(coin, ''))
        fills = cache_fills(resolved, cb)
        signals = compute_all_signals(resolved, cb, ct, bn_data=bn)
        coin_data[coin] = (resolved, fills, signals)

    # Merge BTC+ETH+SOL
    target_coins = [c for c in ['btc', 'eth', 'sol'] if c in coin_data]
    m_resolved, m_fills, m_signals = {}, {}, {obs: {} for obs in [15, 30, 45, 60]}
    for coin in target_coins:
        r, f, s = coin_data[coin]
        m_resolved.update(r)
        m_fills.update(f)
        for obs in [15, 30, 45, 60]:
            if obs in s:
                m_signals[obs].update(s[obs])

    # Count dates
    dates = set()
    for w in m_resolved.values():
        dt = datetime.fromtimestamp(w['start_us'] / 1e6, tz=timezone.utc)
        dates.add(dt.strftime("%Y-%m-%d"))

    print(f"\n  Coins: {', '.join(c.upper() for c in coins)}")
    print(f"  Dates: {', '.join(sorted(dates))}")
    for coin in coins:
        if coin in coin_data:
            r, _, _ = coin_data[coin]
            n5 = sum(1 for w in r.values() if w['interval'] == 5)
            n15 = sum(1 for w in r.values() if w['interval'] == 15)
            nup = sum(1 for w in r.values() if w['outcome'] == 1)
            print(f"  {coin.upper():>4s}: {len(r):>4d} windows (5m={n5}, 15m={n15}) | UP={nup} DOWN={len(r)-nup}")

    # ── 3. Strategy comparison ────────────────────────────────────────
    base = dict(obs_secs=30, flow_threshold=0.2)
    fi = dict(**base, use_imbalance=True)
    fi15 = dict(**fi, interval_filter=15)

    section("STRATEGY COMPARISON (BTC+ETH+SOL)")

    strategies = [
        ("always buy UP (benchmark)", dict(obs_secs=30, flow_threshold=0.0)),
        ("flow thr=0.2 (all intervals)", base),
        ("flow+imb thr=0.2 (all intervals)", fi),
        ("15m flow thr=0.2", dict(**base, interval_filter=15)),
        ("15m flow+imb thr=0.2", fi15),
        ("5m flow thr=0.2", dict(**base, interval_filter=5)),
    ]
    for label, kw in strategies:
        r = evaluate(m_resolved, m_fills, m_signals, **kw)
        print(row(label, stats(r)))

    # ── 4. Per coin (15m flow+imb) ────────────────────────────────────
    section("PER COIN — 15m flow+imb thr=0.2")

    for coin in coins:
        if coin not in coin_data:
            continue
        r, f, s = coin_data[coin]
        results = evaluate(r, f, s, **fi15)
        print(row(coin.upper(), stats(results)))

    # Combined
    r = evaluate(m_resolved, m_fills, m_signals, **fi15)
    print(row("BTC+ETH+SOL", stats(r)))

    # ── 5. Distance to target (the key finding) ──────────────────────
    section("DISTANCE TO TARGET — 15m flow+imb (BTC+ETH+SOL)")

    print("  Absolute distance:")
    for d in [9999, 20, 15, 10, 7, 5, 3]:
        label = "no filter" if d == 9999 else f"|dist| <= {d}bps"
        r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_max=d)
        print(row(label, stats(r), 30))

    print("\n  Directional:")
    r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=0)
    print(row("price ABOVE target", stats(r), 30))
    r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_max_signed=0)
    print(row("price BELOW target", stats(r), 30))

    print("\n  Above target by X bps:")
    for d in [0, 3, 5, 7, 10]:
        r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=d)
        print(row(f"above +{d}bps", stats(r), 30))

    print("\n  Below target by X bps:")
    for d in [0, -3, -5, -7, -10]:
        r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_max_signed=d)
        print(row(f"below {d}bps", stats(r), 30))

    # Per coin distance
    print("\n  Per coin — above +3bps:")
    for coin in target_coins:
        r, f, s = coin_data[coin]
        results = evaluate(r, f, s, **fi15, bn_distance_min=3)
        print(row(f"{coin.upper()} above +3bps", stats(results), 30))

    print("\n  Per coin — above +5bps:")
    for coin in target_coins:
        r, f, s = coin_data[coin]
        results = evaluate(r, f, s, **fi15, bn_distance_min=5)
        print(row(f"{coin.upper()} above +5bps", stats(results), 30))

    # ── 6. Other Binance features ─────────────────────────────────────
    section("OTHER BINANCE FEATURES — 15m flow+imb (BTC+ETH+SOL)")

    print("  BN momentum (ret 30s):")
    for label, lo, hi in [("all", -9999, 9999), ("positive", 0, 9999), ("negative", -9999, 0)]:
        r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_momentum_range=(lo, hi))
        print(row(label, stats(r), 30))

    print("\n  BN flow alignment:")
    r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_flow_align=False)
    print(row("no filter", stats(r), 30))
    r = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_flow_align=True)
    print(row("BN flow aligned", stats(r), 30))

    # ── 7. DOWN token strategies ──────────────────────────────────────
    b15 = dict(obs_secs=30, interval_filter=15)

    section("DOWN TOKEN STRATEGIES — 15m (BTC+ETH+SOL)")

    print("  Benchmarks:")
    r = evaluate(m_resolved, m_fills, m_signals, **b15, force_direction="up")
    print(row("always buy UP", stats(r), 36))
    r = evaluate(m_resolved, m_fills, m_signals, **b15, force_direction="down")
    print(row("always buy DOWN", stats(r), 36))

    print("\n  BN distance → force DOWN:")
    for d in [0, -3, -5, -7, -10]:
        r = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down", bn_distance_max_signed=d)
        print(row(f"DOWN when dist<={d}bps", stats(r), 36))

    print("\n  BN momentum → force DOWN:")
    for lo, hi in [(-9999, 0), (-9999, -3), (-9999, -5), (-9999, -10)]:
        r = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down", bn_momentum_range=(lo, hi))
        print(row(f"DOWN when ret∈[{lo},{hi}]", stats(r), 36))

    print("\n  BN distance + momentum → force DOWN:")
    for d, ret_hi in [(-3, 0), (-5, 0), (-3, -3), (-5, -5)]:
        r = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down",
                     bn_distance_max_signed=d, bn_momentum_range=(-9999, ret_hi))
        print(row(f"DOWN d<={d} & ret<{ret_hi}", stats(r), 36))

    print("\n  DOWN token flow (people buying DOWN on Poly):")
    for dn_fl in [0, 0.1, 0.2, 0.3]:
        r = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down", dn_flow_min=dn_fl)
        print(row(f"DOWN when dn_flow>={dn_fl}", stats(r), 36))

    print("\n  DOWN token flow + BN distance:")
    for dn_fl, d in [(0.1, 0), (0.2, 0), (0.1, -3), (0.2, -3), (0.1, -5)]:
        r = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down", dn_flow_min=dn_fl,
                     bn_distance_max_signed=d)
        print(row(f"DOWN dn_fl>={dn_fl} & d<={d}", stats(r), 36))

    # Custom: UP flow negative + distance negative
    print("\n  UP token bearish signals → buy DOWN:")
    r = evaluate(m_resolved, m_fills, m_signals,
                 obs_secs=30, flow_threshold=0.2, use_imbalance=True, interval_filter=15)
    dn_only = [t for t in r if t['direction'] == 'down']
    print(row(f"flow+imb gives DOWN", stats(dn_only), 36))

    r = evaluate(m_resolved, m_fills, m_signals,
                 obs_secs=30, flow_threshold=0.2, use_imbalance=True, interval_filter=15,
                 bn_distance_max_signed=0)
    dn_only = [t for t in r if t['direction'] == 'down']
    print(row(f"flow+imb DOWN + dist<=0", stats(dn_only), 36))

    print("\n  Combined UP+DOWN (best of both):")
    # UP strategy: flow+imb + distance >= 0
    up_trades = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=0)
    # DOWN strategy: force DOWN + distance <= -3
    dn_trades = evaluate(m_resolved, m_fills, m_signals, **b15,
                         force_direction="down", bn_distance_max_signed=-3)
    combined = sorted(up_trades + dn_trades, key=lambda t: t['start_us'])
    print(row("UP(fi+d>=0) + DN(d<=-3)", stats(combined), 36))

    up_trades2 = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=3)
    dn_trades2 = evaluate(m_resolved, m_fills, m_signals, **b15,
                          force_direction="down", bn_distance_max_signed=-3,
                          dn_flow_min=0.1)
    combined2 = sorted(up_trades2 + dn_trades2, key=lambda t: t['start_us'])
    print(row("UP(fi+d>=3) + DN(d<=-3,fl)", stats(combined2), 36))

    # ── 8. Daily breakdown ────────────────────────────────────────────
    section("DAILY — 15m flow+imb thr=0.2 (BTC+ETH+SOL)")

    r = evaluate(m_resolved, m_fills, m_signals, **fi15)
    daily(r)

    # ── 8. Daily breakdown — ALL key strategies ────────────────────────
    section("DAILY BREAKDOWN — ALL KEY STRATEGIES (BTC+ETH+SOL)")

    key_strats = [
        ("UP: flow+imb (base)",         fi15, {}),
        ("UP: fi + d>=0",               fi15, dict(bn_distance_min=0)),
        ("UP: fi + d>=+3bps",           fi15, dict(bn_distance_min=3)),
        ("UP: fi + d>=+5bps",           fi15, dict(bn_distance_min=5)),
        ("DOWN: d<=-3bps",              b15,  dict(force_direction="down", bn_distance_max_signed=-3)),
        ("DOWN: d<=-5bps",              b15,  dict(force_direction="down", bn_distance_max_signed=-5)),
    ]

    for label, base_kw, extra_kw in key_strats:
        r = evaluate(m_resolved, m_fills, m_signals, **base_kw, **extra_kw)
        s = stats(r)
        print(f"\n  {label}  (N={s['n']} WR={s['wr']}% PnL=${s['pnl']:+.0f})")
        daily(r)

    # Mixed strategies daily
    print(f"\n  MIX: UP(fi+d>=0) + DN(d<=-5)")
    up_t = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=0)
    dn_t = evaluate(m_resolved, m_fills, m_signals, **b15,
                    force_direction="down", bn_distance_max_signed=-5)
    mixed = sorted(up_t + dn_t, key=lambda t: t['start_us'])
    s = stats(mixed)
    print(f"  (N={s['n']} WR={s['wr']}% PnL=${s['pnl']:+.0f} Sharpe={s['sharpe']:+.3f})")
    daily(mixed)

    print(f"\n  MIX: UP(fi+d>=3) + DN(d<=-5)")
    up_t2 = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=3)
    dn_t2 = evaluate(m_resolved, m_fills, m_signals, **b15,
                     force_direction="down", bn_distance_max_signed=-5)
    mixed2 = sorted(up_t2 + dn_t2, key=lambda t: t['start_us'])
    s2 = stats(mixed2)
    print(f"  (N={s2['n']} WR={s2['wr']}% PnL=${s2['pnl']:+.0f} Sharpe={s2['sharpe']:+.3f})")
    daily(mixed2)

    # ── 9. Per coin for each strategy ─────────────────────────────────
    section("PER COIN × PER STRATEGY — 15m (BTC+ETH+SOL)")

    coin_strats = [
        ("fi base",    fi15, {}),
        ("fi+d>=0",    fi15, dict(bn_distance_min=0)),
        ("fi+d>=3",    fi15, dict(bn_distance_min=3)),
        ("DN d<=-5",   b15,  dict(force_direction="down", bn_distance_max_signed=-5)),
    ]

    for coin in target_coins:
        r, f, s = coin_data[coin]
        print(f"\n  {coin.upper()}:")
        for label, base_kw, extra_kw in coin_strats:
            results = evaluate(r, f, s, **base_kw, **extra_kw)
            print(row(label, stats(results), 20))
        # Daily for base
        results = evaluate(r, f, s, **fi15)
        if results:
            daily(results)

    # ── 10. XRP (always shown for reference) ──────────────────────────
    if 'xrp' in coin_data:
        section("XRP (reference — do not trade)")
        r, f, s = coin_data['xrp']
        for label, kw in [("15m flow+imb", fi15), ("15m flow only", dict(**base, interval_filter=15))]:
            results = evaluate(r, f, s, **kw)
            print(row(label, stats(results)))

    # ── 11. BEST STRATEGY SUMMARY ─────────────────────────────────────
    section("★ BEST STRATEGY — MIX: UP(fi+d>=0) + DN(d<=-5)")

    up_trades = evaluate(m_resolved, m_fills, m_signals, **fi15, bn_distance_min=0)
    dn_trades = evaluate(m_resolved, m_fills, m_signals, **b15,
                         force_direction="down", bn_distance_max_signed=-5)
    best = sorted(up_trades + dn_trades, key=lambda t: t['start_us'])
    s_best = stats(best)

    n_up = sum(1 for t in best if t['direction'] == 'up')
    n_dn = sum(1 for t in best if t['direction'] == 'down')
    up_wins = sum(1 for t in best if t['direction'] == 'up' and t['pnl'] > 0)
    dn_wins = sum(1 for t in best if t['direction'] == 'down' and t['pnl'] > 0)
    up_pnl = sum(t['pnl'] for t in best if t['direction'] == 'up')
    dn_pnl = sum(t['pnl'] for t in best if t['direction'] == 'down')

    print(f"\n  Total: {s_best['n']} trades | PnL ${s_best['pnl']:+.1f} | "
          f"WR {s_best['wr']}% | Sharpe {s_best['sharpe']:+.3f} | DD ${s_best['dd']:.1f}")
    print(f"  UP arm:   {n_up} trades ({up_wins}W/{n_up-up_wins}L) "
          f"WR={up_wins/n_up*100:.0f}% PnL=${up_pnl:+.1f}" if n_up else "")
    print(f"  DOWN arm: {n_dn} trades ({dn_wins}W/{n_dn-dn_wins}L) "
          f"WR={dn_wins/n_dn*100:.0f}% PnL=${dn_pnl:+.1f}" if n_dn else "")

    print(f"\n  Rules:")
    print(f"    BUY UP:   15m window + flow>0.2 + imbalance>0 + BN price >= target")
    print(f"    BUY DOWN: 15m window + BN price <= target - 5bps")
    print(f"    SKIP:     everything else")
    print(f"    Entry:    taker buy at t+30s | Size: $50 | Hold to settlement")

    # Daily
    print(f"\n  Daily:")
    by_date = defaultdict(list)
    for t in best:
        dt = datetime.fromtimestamp(t['start_us'] / 1e6, tz=timezone.utc)
        by_date[dt.strftime("%Y-%m-%d")].append(t)

    for date in sorted(by_date):
        day = by_date[date]
        pnl_arr = [t['pnl'] for t in day]
        n = len(pnl_arr)
        wins = sum(1 for p in pnl_arr if p > 0)
        d_up = [t for t in day if t['direction'] == 'up']
        d_dn = [t for t in day if t['direction'] == 'down']
        up_w = sum(1 for t in d_up if t['pnl'] > 0)
        dn_w = sum(1 for t in d_dn if t['pnl'] > 0)
        print(f"    {date}:  {n:>3d} trades  PnL=${sum(pnl_arr):>+8.1f}  WR={wins/n*100:.0f}%"
              f"  | UP {len(d_up)}t ({up_w}W) ${sum(t['pnl'] for t in d_up):>+7.1f}"
              f"  | DN {len(d_dn)}t ({dn_w}W) ${sum(t['pnl'] for t in d_dn):>+7.1f}")

    # Per coin
    print(f"\n  Per coin:")
    for coin in target_coins:
        ct = [t for t in best if t['coin'] == coin]
        if not ct:
            continue
        s_c = stats(ct)
        c_up = [t for t in ct if t['direction'] == 'up']
        c_dn = [t for t in ct if t['direction'] == 'down']
        print(f"    {coin.upper():>4s}:  {s_c['n']:>3d}t  PnL=${s_c['pnl']:>+7.1f}  WR={s_c['wr']}%"
              f"  | UP {len(c_up)}t ${sum(t['pnl'] for t in c_up):>+7.1f}"
              f"  | DN {len(c_dn)}t ${sum(t['pnl'] for t in c_dn):>+7.1f}")

    # At $5 sizing
    scale = 5 / 50
    n_days = len(by_date)
    print(f"\n  At $5/trade ({n_days} days):")
    print(f"    PnL: ${s_best['pnl']*scale:+.1f} total | "
          f"${s_best['pnl']*scale/n_days:+.1f}/day")
    print(f"    DD:  ${s_best['dd']*scale:.1f}")
    print(f"    Trades/day: {s_best['n']/n_days:.0f}")

    # ── Done ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  DONE in {time.time()-t_total:.0f}s")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('--coin', nargs='+', help='Filter to specific coins')
    args = parser.parse_args()
    main(args.events_dir, args.coin)