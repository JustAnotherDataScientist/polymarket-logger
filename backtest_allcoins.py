#!/usr/bin/env python3
"""
backtest_allcoins.py — v4: clean dashboard, real fees, expanded strategies.

All PnL includes Polymarket crypto taker fees: 0.072 × p × (1−p).
Maker dropped (adverse selection proven non-viable).

Usage:
  python backtest_allcoins.py historical_data/events
  python backtest_allcoins.py historical_data/events --coin btc eth
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

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_all_markets(events_dir):
    mdir = os.path.join(events_dir, 'poly_markets')
    files = sorted(glob.glob(os.path.join(mdir, '*.parquet')))
    if not files:
        return {}
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
    return {s: w for s, w in windows.items() if w['outcome'] in (0, 1)}

def load_books_and_trades(events_dir, token_ids):
    t0 = time.time()
    token_set = set(token_ids)
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
    t0 = time.time()
    bdir = os.path.join(events_dir, 'bn_trades')
    files = sorted(glob.glob(os.path.join(bdir, '*.parquet')))
    if not files:
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
    result = {}
    for sym, grp in bn.groupby('symbol'):
        grp = grp.sort_values('ts_us')
        result[sym] = {
            'ts_us': grp['ts_us'].values,
            'price': grp['price'].values.astype(np.float64),
            'qty': grp['qty'].values.astype(np.float64),
            'is_buy': grp['is_buy'].values,
        }
    return result

# ═══════════════════════════════════════════════════════════════════════
# Binance features
# ═══════════════════════════════════════════════════════════════════════

def _bn_features_at(bn_data, t_us):
    if bn_data is None:
        return {}
    ts, px, qty, is_buy = bn_data['ts_us'], bn_data['price'], bn_data['qty'], bn_data['is_buy']
    idx_now = np.searchsorted(ts, t_us, side='right') - 1
    if idx_now < 0:
        return {}
    price_now = float(px[idx_now])
    features = {'bn_price': price_now}
    for label, lb in [('10s',10_000_000),('30s',30_000_000),('60s',60_000_000),('2m',120_000_000),('5m',300_000_000)]:
        idx_p = np.searchsorted(ts, t_us - lb, side='right') - 1
        if idx_p >= 0:
            pp = float(px[idx_p])
            features[f'bn_ret_{label}'] = (price_now / pp - 1) * 10000 if pp > 0 else 0
        else:
            features[f'bn_ret_{label}'] = 0
    for label, lb in [('1m',60_000_000),('5m',300_000_000)]:
        i_lo, i_hi = np.searchsorted(ts, t_us - lb, side='left'), np.searchsorted(ts, t_us, side='right')
        if i_hi - i_lo > 10:
            pw = px[i_lo:i_hi]
            step = max(1, len(pw)//200)
            sampled = pw[::step].astype(np.float64)
            features[f'bn_vol_{label}'] = float(np.std(np.diff(np.log(sampled)))*10000) if len(sampled)>2 else 0
        else:
            features[f'bn_vol_{label}'] = 0
    for label, lb in [('10s',10_000_000),('30s',30_000_000),('60s',60_000_000)]:
        i_lo, i_hi = np.searchsorted(ts, t_us - lb, side='left'), np.searchsorted(ts, t_us, side='right')
        if i_hi > i_lo:
            wq, wb = qty[i_lo:i_hi], is_buy[i_lo:i_hi]
            bv, sv = float(wq[wb].sum()), float(wq[~wb].sum())
            features[f'bn_flow_{label}'] = (bv-sv)/(bv+sv) if (bv+sv)>0 else 0
        else:
            features[f'bn_flow_{label}'] = 0
    return features

# ═══════════════════════════════════════════════════════════════════════
# Fills (taker with fees)
# ═══════════════════════════════════════════════════════════════════════

def _vwap_from_asks(ask_prices, ask_sizes, size_usdc):
    if not isinstance(ask_prices, (list, np.ndarray)) or len(ask_prices) == 0:
        return None
    remaining, wsum, total = size_usdc, 0.0, 0.0
    for p, s in zip(ask_prices, ask_sizes):
        if remaining <= 0: break
        s = float(s)
        if s <= 0: continue
        fill = min(remaining, s)
        wsum += fill * float(p)
        total += fill
        remaining -= fill
    return wsum / total if total > 0 else None

def cache_fills(resolved, books, size_usdc=50.0):
    token_index = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        token_index[tid] = (grp['ts_us'].values, grp['ask_prices'].values, grp['ask_sizes'].values)
    entry_offsets = [0, 15, 30, 45, 60]
    fills = {}
    for slug, w in resolved.items():
        row = {'outcome': w['outcome'], 'interval': w['interval']}
        for side, token in [('up', w['up_token']), ('down', w['down_token'])]:
            idx = token_index.get(token)
            if idx is None: continue
            ts_arr, ask_p, ask_s = idx
            for offset in entry_offsets:
                entry_us = w['start_us'] + offset * 1_000_000
                if entry_us >= w['end_us']: continue
                pos = np.searchsorted(ts_arr, entry_us, side='left')
                deadline = min(entry_us + 15_000_000, w['end_us'])
                price = None
                while pos < len(ts_arr) and ts_arr[pos] <= deadline:
                    price = _vwap_from_asks(ask_p[pos], ask_s[pos], size_usdc)
                    if price is not None: break
                    pos += 1
                if price is not None:
                    row[f'{side}_fill_{offset}s'] = price * (1 + taker_fee_rate(price))
                else:
                    row[f'{side}_fill_{offset}s'] = None
        fills[slug] = row
    return fills

# ═══════════════════════════════════════════════════════════════════════
# Signals
# ═══════════════════════════════════════════════════════════════════════

def compute_all_signals(resolved, books, trades, bn_data=None):
    books_by_token = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        books_by_token[tid] = {'ts_us': grp['ts_us'].values, 'mid': grp['mid'].values,
                                'spread_bps': grp['spread_bps'].values, 'imbalance': grp['imbalance'].values}
    trades_by_token = {}
    if len(trades) > 0:
        for tid, grp in trades.groupby('token_id'):
            grp = grp.sort_values('ts_us')
            trades_by_token[tid] = {'ts_us': grp['ts_us'].values, 'price': grp['price'].values.astype(np.float64),
                                     'size': grp['size'].values.astype(np.float64)}
    trade_is_buy = {}
    for tid, td in trades_by_token.items():
        bd = books_by_token.get(tid)
        if bd is None or len(bd['ts_us'])==0:
            trade_is_buy[tid] = td['price'] > 0; continue
        idx = np.clip(np.searchsorted(bd['ts_us'], td['ts_us'], side='right')-1, 0, len(bd['mid'])-1)
        trade_is_buy[tid] = td['price'] >= bd['mid'][idx]

    def _flow_imb(token_id, t_start, obs_us):
        bd = books_by_token.get(token_id)
        if bd is None or len(bd['ts_us'])==0: return 0,0,0,0,0
        bts = bd['ts_us']
        i_lo, i_hi = np.searchsorted(bts, t_start, side='left'), np.searchsorted(bts, t_start+10_000_000, side='right')
        if i_lo >= i_hi: return 0,0,0,0,0
        mid = float(bd['mid'][i_lo])
        imb = float(np.mean(bd['imbalance'][i_lo:i_hi]))
        td = trades_by_token.get(token_id)
        if td is None or len(td['ts_us'])==0: return 0, imb, mid, 0, 0
        j_lo, j_hi = np.searchsorted(td['ts_us'], t_start, side='left'), np.searchsorted(td['ts_us'], t_start+obs_us, side='right')
        nt = j_hi - j_lo
        if nt == 0: return 0, imb, mid, 0, 0
        sz = td['size'][j_lo:j_hi]
        ib = trade_is_buy.get(token_id)
        if ib is None: return 0, imb, mid, 0, nt
        ibs = ib[j_lo:j_hi]
        bv, sv = float(sz[ibs].sum()), float(sz[~ibs].sum())
        total = bv + sv
        return ((bv-sv)/total if total>0 else 0), imb, mid, total, nt

    all_signals = {}
    for obs_secs in [15, 30, 45, 60]:
        sigs = {}
        obs_us = obs_secs * 1_000_000
        for slug, w in resolved.items():
            t_start = w['start_us']
            t_entry = t_start + obs_us
            up_flow, up_imb, up_mid, up_vol, up_nt = _flow_imb(w['up_token'], t_start, obs_us)
            if up_mid == 0: continue
            bd = books_by_token.get(w['up_token'])
            spread = 0
            if bd is not None:
                bts = bd['ts_us']
                il, ih = np.searchsorted(bts, t_start, side='left'), np.searchsorted(bts, t_start+10_000_000, side='right')
                if ih > il: spread = float(np.median(bd['spread_bps'][il:ih]))
            dn_flow, dn_imb, dn_mid, dn_vol, dn_nt = _flow_imb(w['down_token'], t_start, obs_us)
            sig = {'flow': up_flow, 'imbalance': up_imb, 'mid_start': up_mid, 'spread_start': spread,
                   'total_vol': up_vol, 'n_trades': up_nt,
                   'dn_flow': dn_flow, 'dn_imbalance': dn_imb, 'dn_mid': dn_mid}
            if bn_data is not None:
                bf = _bn_features_at(bn_data, t_entry)
                sig.update(bf)
                tgt = w.get('target_price', 0)
                sig['bn_distance_bps'] = (bf['bn_price']-tgt)/tgt*10000 if tgt and bf.get('bn_price') else 0
            else:
                for k in ['bn_price','bn_ret_10s','bn_ret_30s','bn_ret_60s','bn_ret_2m','bn_ret_5m',
                           'bn_vol_1m','bn_vol_5m','bn_flow_10s','bn_flow_30s','bn_flow_60s','bn_distance_bps']:
                    sig[k] = 0
            sigs[slug] = sig
        all_signals[obs_secs] = sigs
    return all_signals

# ═══════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════

def evaluate(resolved, fills, signals, obs_secs=30, flow_threshold=0.0,
             use_imbalance=False, interval_filter=None, size_usdc=50.0,
             force_direction=None,
             bn_distance_min=None, bn_distance_max_signed=None, bn_distance_max=None,
             bn_momentum_range=None, bn_flow_align=False,
             bn_vol_min=None, bn_vol_max=None, dn_flow_min=None):
    sigs = signals.get(obs_secs, {})
    results = []
    for slug, w in resolved.items():
        if slug not in sigs or slug not in fills: continue
        if interval_filter and w['interval'] != interval_filter: continue
        sig, f = sigs[slug], fills[slug]
        flow, imb = sig['flow'], sig['imbalance']
        if force_direction is None and abs(flow) < flow_threshold: continue
        if bn_momentum_range is not None:
            r = sig.get('bn_ret_30s', 0)
            if not (bn_momentum_range[0] <= r <= bn_momentum_range[1]): continue
        if bn_vol_min is not None and sig.get('bn_vol_1m',0) < bn_vol_min: continue
        if bn_vol_max is not None and sig.get('bn_vol_1m',0) > bn_vol_max: continue
        if bn_distance_max is not None and abs(sig.get('bn_distance_bps',0)) > bn_distance_max: continue
        if bn_distance_min is not None and sig.get('bn_distance_bps',0) < bn_distance_min: continue
        if bn_distance_max_signed is not None and sig.get('bn_distance_bps',0) > bn_distance_max_signed: continue
        if dn_flow_min is not None and sig.get('dn_flow',0) < dn_flow_min: continue
        if force_direction is not None:
            direction = force_direction
        elif use_imbalance:
            if flow > 0 and imb > 0: direction = "up"
            elif flow < 0 and imb < 0: direction = "down"
            else: continue
        else:
            direction = "up" if flow > 0 else "down"
        if bn_flow_align:
            bf = sig.get('bn_flow_30s', 0)
            if direction=="up" and bf<=0: continue
            if direction=="down" and bf>=0: continue
        entry = f.get(f'{direction}_fill_{obs_secs}s')
        if entry is None: continue
        settle = 1.0 if (direction=="up" and w['outcome']==1) or (direction=="down" and w['outcome']==0) else 0.0
        pnl = (settle/entry - 1.0) * size_usdc if entry > 0 else 0.0
        results.append({'slug': slug, 'coin': w['coin'], 'interval': w['interval'],
                        'outcome': w['outcome'], 'direction': direction,
                        'flow': flow, 'imbalance': imb, 'entry': entry, 'settle': settle,
                        'pnl': pnl, 'start_us': w['start_us'],
                        'bn_distance_bps': sig.get('bn_distance_bps',0),
                        'bn_ret_30s': sig.get('bn_ret_30s',0),
                        'bn_vol_1m': sig.get('bn_vol_1m',0),
                        'bn_flow_30s': sig.get('bn_flow_30s',0)})
    return results

# ═══════════════════════════════════════════════════════════════════════
# Stats & formatting
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

def print_strat_card(label, results, all_dates, show_coins=None):
    s = stats(results)
    if s['n'] == 0:
        print(f"\n  {label}:  — no trades"); return
    print(f"\n  ┌─ {label}")
    print(f"  │  N={s['n']}  PnL=${s['pnl']:+.1f}  avg=${s['avg']:+.2f}  "
          f"WR={s['wr']:.1f}%  Sharpe={s['sharpe']:+.3f}  DD=${s['dd']:.1f}")
    by_date = _daily_map(results)
    print(f"  │")
    print(f"  │  {'date':>12s}  {'N':>4s}  {'PnL':>9s}  {'avg':>8s}  {'WR':>5s}")
    print(f"  │  {'─'*44}")
    for date in sorted(all_dates):
        day = by_date.get(date, [])
        if not day:
            print(f"  │  {date:>12s}  {'—':>4s}"); continue
        dp = np.array([t['pnl'] for t in day])
        dw = int((dp>0).sum())
        print(f"  │  {date:>12s}  {len(dp):>4d}  ${dp.sum():>+8.1f}  ${dp.mean():>+7.2f}  {dw/len(dp)*100:>4.0f}%")
    if show_coins:
        print(f"  │")
        print(f"  │  {'coin':>6s}  {'N':>4s}  {'PnL':>9s}  {'avg':>8s}  {'WR':>5s}  {'Sharpe':>7s}")
        print(f"  │  {'─'*50}")
        for coin in show_coins:
            ct = [r for r in results if r['coin'] == coin]
            if not ct: continue
            cs = stats(ct)
            print(f"  │  {coin.upper():>6s}  {cs['n']:>4d}  ${cs['pnl']:>+8.1f}  ${cs['avg']:>+7.2f}  {cs['wr']:>4.1f}%  {cs['sharpe']:>+6.3f}")
    print(f"  └{'─'*60}")

# ═══════════════════════════════════════════════════════════════════════
# Strategy definitions
# ═══════════════════════════════════════════════════════════════════════

def define_strategies():
    b15 = dict(interval_filter=15)
    fi = dict(flow_threshold=0.2, use_imbalance=True)
    strats = []

    # A. Benchmarks
    strats.append(("A1  always buy UP",                dict(**b15, force_direction="up")))
    strats.append(("A2  always buy DOWN",              dict(**b15, force_direction="down")))

    # B. Flow only
    strats.append(("B1  flow>0.2",                     dict(**b15, flow_threshold=0.2)))
    strats.append(("B2  flow+imbalance",               dict(**b15, **fi)))

    # C. Distance: BUY UP when above target
    strats.append(("C1  fi + dist>=0",                 dict(**b15, **fi, bn_distance_min=0)))
    strats.append(("C2  fi + dist>=+3bps",             dict(**b15, **fi, bn_distance_min=3)))
    strats.append(("C3  fi + dist>=+5bps",             dict(**b15, **fi, bn_distance_min=5)))
    strats.append(("C4  fi + dist>=+7bps",             dict(**b15, **fi, bn_distance_min=7)))
    strats.append(("C5  fi + dist>=+10bps",            dict(**b15, **fi, bn_distance_min=10)))

    # D. Mean reversion: BUY UP when below target but flow bullish
    strats.append(("D1  fi + dist<=0 (mean-rev)",      dict(**b15, **fi, bn_distance_max_signed=0)))
    strats.append(("D2  fi + dist<=-3bps",             dict(**b15, **fi, bn_distance_max_signed=-3)))
    strats.append(("D3  fi + dist<=-5bps",             dict(**b15, **fi, bn_distance_max_signed=-5)))

    # E. Pure distance (no flow filter)
    strats.append(("E1  UP dist>=+5 (no flow)",        dict(**b15, force_direction="up", bn_distance_min=5)))
    strats.append(("E2  UP dist>=+10 (no flow)",       dict(**b15, force_direction="up", bn_distance_min=10)))
    strats.append(("E3  UP dist<=-5 (no flow, m-rev)", dict(**b15, force_direction="up", bn_distance_max_signed=-5)))

    # F. Momentum filter
    strats.append(("F1  fi+d>=3 + ret30s>0",           dict(**b15, **fi, bn_distance_min=3, bn_momentum_range=(0, 9999))))
    strats.append(("F2  fi+d>=3 + ret30s<0 (reversal)",dict(**b15, **fi, bn_distance_min=3, bn_momentum_range=(-9999, 0))))
    strats.append(("F3  fi+d<=0 + ret30s<0 (dip-buy)", dict(**b15, **fi, bn_distance_max_signed=0, bn_momentum_range=(-9999, 0))))
    strats.append(("F4  fi+d<=0 + ret30s>0 (bounce)",  dict(**b15, **fi, bn_distance_max_signed=0, bn_momentum_range=(0, 9999))))

    # G. BN flow alignment
    strats.append(("G1  fi+d>=3 + BN flow aligned",    dict(**b15, **fi, bn_distance_min=3, bn_flow_align=True)))
    strats.append(("G2  fi+d<=0 + BN flow aligned",    dict(**b15, **fi, bn_distance_max_signed=0, bn_flow_align=True)))

    # H. Observation window timing
    strats.append(("H1  fi+d>=5 obs=15s",              dict(**b15, **fi, bn_distance_min=5, obs_secs=15)))
    strats.append(("H2  fi+d>=5 obs=30s",              dict(**b15, **fi, bn_distance_min=5, obs_secs=30)))
    strats.append(("H3  fi+d>=5 obs=45s",              dict(**b15, **fi, bn_distance_min=5, obs_secs=45)))
    strats.append(("H4  fi+d>=5 obs=60s",              dict(**b15, **fi, bn_distance_min=5, obs_secs=60)))

    # I. Tight distance (50/50 zone)
    strats.append(("I1  fi + |dist|<=5bps",            dict(**b15, **fi, bn_distance_max=5)))
    strats.append(("I2  fi + |dist|<=3bps",            dict(**b15, **fi, bn_distance_max=3)))

    # J. Volatility filter
    strats.append(("J1  fi+d>=3 + low vol (<3)",       dict(**b15, **fi, bn_distance_min=3, bn_vol_max=3)))
    strats.append(("J2  fi+d>=3 + high vol (>3)",      dict(**b15, **fi, bn_distance_min=3, bn_vol_min=3)))
    strats.append(("J3  fi+d<=0 + high vol",           dict(**b15, **fi, bn_distance_max_signed=0, bn_vol_min=3)))

    # K. Combined: distance + momentum + vol
    strats.append(("K1  fi+d>=5 + ret30s>0 + lo-vol",  dict(**b15, **fi, bn_distance_min=5, bn_momentum_range=(0,9999), bn_vol_max=5)))
    strats.append(("K2  fi+d>=5 + BN flow aligned",    dict(**b15, **fi, bn_distance_min=5, bn_flow_align=True)))
    strats.append(("K3  fi+d<=0 + ret<0 + hi-vol",     dict(**b15, **fi, bn_distance_max_signed=0, bn_momentum_range=(-9999,0), bn_vol_min=2)))

    return strats

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(events_dir, coins_filter=None):
    print(f"\n{'═'*74}")
    print(f"  POLYMARKET BACKTEST v4")
    print(f"  All PnL includes taker fee: 0.072 × p × (1−p)")
    print(f"{'═'*74}")

    t_total = time.time()
    print(f"\n  Loading data...")
    all_windows = load_all_markets(events_dir)
    coins = coins_filter if coins_filter else sorted(set(w['coin'] for w in all_windows.values()))
    all_tokens, coin_windows = set(), {}
    for coin in coins:
        cw = {s: w for s, w in all_windows.items() if w['coin'] == coin}
        coin_windows[coin] = cw
        for w in cw.values():
            all_tokens.add(w['up_token']); all_tokens.add(w['down_token'])

    books, trades = load_books_and_trades(events_dir, all_tokens)
    bn_by_symbol = load_bn_trades(events_dir)

    print(f"\n  Computing signals + fills...")
    t_comp = time.time()
    coin_data = {}
    for coin in coins:
        resolved = coin_windows[coin]
        if not resolved: continue
        ct_set = set()
        for w in resolved.values(): ct_set.add(w['up_token']); ct_set.add(w['down_token'])
        cb = books[books['token_id'].isin(ct_set)] if len(books)>0 else books
        ct = trades[trades['token_id'].isin(ct_set)] if len(trades)>0 else trades
        bn = bn_by_symbol.get(COIN_TO_SYMBOL.get(coin, ''))
        coin_data[coin] = (resolved, cache_fills(resolved, cb), compute_all_signals(resolved, cb, ct, bn_data=bn))
    print(f"    Done in {time.time()-t_comp:.1f}s")

    target_coins = [c for c in ['btc','eth','sol'] if c in coin_data]
    m_res, m_fills, m_sigs = {}, {}, {obs:{} for obs in [15,30,45,60]}
    for coin in target_coins:
        r, f, s = coin_data[coin]
        m_res.update(r); m_fills.update(f)
        for obs in [15,30,45,60]:
            if obs in s: m_sigs[obs].update(s[obs])

    all_dates = sorted(set(datetime.fromtimestamp(w['start_us']/1e6, tz=timezone.utc).strftime("%Y-%m-%d") for w in m_res.values()))
    n_15m = sum(1 for w in m_res.values() if w['interval']==15)

    print(f"\n  Coins: {', '.join(c.upper() for c in target_coins)}")
    print(f"  Dates: {', '.join(all_dates)} ({len(all_dates)} days)")
    print(f"  15m resolved windows: {n_15m}")
    for coin in target_coins:
        r15 = {s:w for s,w in coin_windows.get(coin,{}).items() if w['interval']==15 and w['outcome'] in (0,1)}
        up = sum(1 for w in r15.values() if w['outcome']==1)
        print(f"    {coin.upper()}: {len(r15)} | UP={up} DOWN={len(r15)-up}")

    # ═══════════════════════════════════════════════════════════════
    # Strategy grid
    # ═══════════════════════════════════════════════════════════════
    strats = define_strategies()

    print(f"\n{'═'*90}")
    print(f"  STRATEGY GRID — BTC+ETH+SOL, 15m, taker+fees")
    print(f"{'═'*90}")
    print(f"\n  {'':>38s}  {'N':>4s}  {'PnL':>9s}  {'avg':>7s}  {'WR':>5s}  {'Sharpe':>7s}  {'DD':>9s}")
    print(f"  {'─'*85}")

    strat_results = {}
    for label, kw in strats:
        r = evaluate(m_res, m_fills, m_sigs, **kw)
        s = stats(r)
        strat_results[label] = r
        if s['n'] == 0:
            print(f"  {label:>38s}  {'—':>4s}")
        else:
            print(f"  {label:>38s}  {s['n']:>4d}  ${s['pnl']:>+8.1f}  ${s['avg']:>+6.2f}  {s['wr']:>4.1f}%  {s['sharpe']:>+6.3f}  ${s['dd']:>+8.1f}")

    # Rankings
    ranked = [(l, stats(r)) for l, r in strat_results.items() if stats(r)['n'] >= 15]
    by_sharpe = sorted(ranked, key=lambda x: x[1]['sharpe'], reverse=True)
    by_pnl = sorted(ranked, key=lambda x: x[1]['pnl'], reverse=True)

    print(f"\n{'═'*74}")
    print(f"  TOP 5 BY SHARPE (N≥15)")
    print(f"{'═'*74}")
    for l, s in by_sharpe[:5]:
        print(f"    {s['sharpe']:>+6.3f}  {l}  N={s['n']} PnL=${s['pnl']:+.1f} WR={s['wr']}%")

    print(f"\n{'═'*74}")
    print(f"  TOP 5 BY PnL (N≥15)")
    print(f"{'═'*74}")
    for l, s in by_pnl[:5]:
        print(f"    ${s['pnl']:>+8.1f}  {l}  N={s['n']} Sharpe={s['sharpe']:+.3f} WR={s['wr']}%")

    # ═══════════════════════════════════════════════════════════════
    # Detailed cards
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*74}")
    print(f"  DETAILED CARDS — top strategies + benchmarks")
    print(f"{'═'*74}")

    shown = set()
    cards = []
    for l, _ in by_sharpe[:5]:
        if l not in shown: cards.append(l); shown.add(l)
    for l, s in ranked:
        if s['sharpe'] > 0.05 and l not in shown: cards.append(l); shown.add(l)
    for l in ["A1  always buy UP", "B2  flow+imbalance"]:
        if l not in shown: cards.append(l); shown.add(l)

    for label in cards:
        print_strat_card(label, strat_results.get(label,[]), all_dates, show_coins=target_coins)

    # ═══════════════════════════════════════════════════════════════
    # Per coin
    # ═══════════════════════════════════════════════════════════════
    top_labels = [l for l,s in by_sharpe[:5] if s['sharpe']>0]
    ref_labels = ["A1  always buy UP", "B2  flow+imbalance"]

    print(f"\n{'═'*74}")
    print(f"  PER COIN — top strategies")
    print(f"{'═'*74}")

    for coin in target_coins:
        r, f, s = coin_data[coin]
        print(f"\n  {coin.upper()}:")
        print(f"  {'':>38s}  {'N':>4s}  {'PnL':>9s}  {'avg':>7s}  {'WR':>5s}  {'Sharpe':>7s}")
        print(f"  {'─'*72}")
        for label, kw in strats:
            if label not in top_labels and label not in ref_labels: continue
            cr = evaluate(r, f, s, **kw)
            cs = stats(cr)
            if cs['n'] == 0: continue
            print(f"  {label:>38s}  {cs['n']:>4d}  ${cs['pnl']:>+8.1f}  ${cs['avg']:>+6.2f}  {cs['wr']:>4.1f}%  {cs['sharpe']:>+6.3f}")

    print(f"\n{'═'*74}")
    print(f"  DONE in {time.time()-t_total:.0f}s")
    print(f"{'═'*74}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('--coin', nargs='+', help='Filter coins')
    args = parser.parse_args()
    main(args.events_dir, args.coin)