#!/usr/bin/env python3
"""
plot_windows.py — Generate interactive HTML viewer for 15min UP token price action.

Reads parquet data, extracts mid prices per window, embeds as JSON in a
self-contained HTML file with Chart.js.

Usage:
  python plot_windows.py historical_data/events
  → opens window_viewer.html in your browser
"""

import sys, os, glob, json, argparse, time, webbrowser
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def load_markets(events_dir):
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
                'interval_min': int(row['interval_min']),
                'up_token_id': row['up_token_id'],
                'down_token_id': row['down_token_id'],
                'window_start_us': int(row['window_start_us']),
                'window_end_us': int(row['window_end_us']),
                'target_price': float(row['target_price']),
                'outcome': -1,
            }
        if int(row.get('outcome', -1)) in (0, 1):
            windows[slug]['outcome'] = int(row['outcome'])
    return windows


def extract_window_series(events_dir, windows):
    """For each 15min resolved window, extract mid price time series."""
    # Filter to 15min resolved
    w15 = {s: w for s, w in windows.items()
            if w['interval_min'] == 15 and w['outcome'] in (0, 1)}

    print(f"  {len(w15)} resolved 15min windows")

    # Collect all UP tokens needed
    tokens_needed = set()
    for w in w15.values():
        tokens_needed.add(w['up_token_id'])

    # Load books
    print(f"  Loading books for {len(tokens_needed)} tokens...")
    bdir = os.path.join(events_dir, 'poly_books')
    chunks = []
    for f in sorted(glob.glob(os.path.join(bdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(tokens_needed)]
        if len(df) > 0:
            chunks.append(df)

    if not chunks:
        print("  No book data found!")
        return []

    books = pd.concat(chunks, ignore_index=True)
    books = books[(books['best_bid'] > 0) & (books['best_ask'] > 0)].copy()
    books['mid'] = (books['best_bid'] + books['best_ask']) / 2
    books['spread_bps'] = (books['best_ask'] - books['best_bid']) * 10000
    def total_size(col):
        return col.apply(lambda x: sum(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0)
    books['bid_depth'] = total_size(books['bid_sizes'])
    books['ask_depth'] = total_size(books['ask_sizes'])
    books['imbalance'] = (books['bid_depth'] - books['ask_depth']) / (books['bid_depth'] + books['ask_depth']).clip(lower=1)
    print(f"  {len(books):,} book snapshots loaded")

    # Load Binance trades
    COIN_TO_SYMBOL = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}
    bn_by_symbol = {}
    bn_dir = os.path.join(events_dir, 'bn_trades')
    bn_files = sorted(glob.glob(os.path.join(bn_dir, '*.parquet')))
    if bn_files:
        bn_chunks = []
        for f in bn_files:
            try:
                df = pq.read_table(f).to_pandas()
                if len(df) > 0:
                    bn_chunks.append(df)
            except Exception:
                pass
        if bn_chunks:
            bn_all = pd.concat(bn_chunks, ignore_index=True).sort_values('ts_us')
            for sym, grp in bn_all.groupby('symbol'):
                grp = grp.sort_values('ts_us')
                bn_by_symbol[sym] = {
                    'ts_us': grp['ts_us'].values,
                    'price': grp['price'].values.astype(np.float64),
                }
            print(f"  {len(bn_all):,} Binance trades loaded")

    # Index by token
    books_by_token = {}
    for tid, grp in books.groupby('token_id'):
        grp = grp.sort_values('ts_us')
        books_by_token[tid] = {
            'ts_us': grp['ts_us'].values,
            'mid': grp['mid'].values,
            'best_bid': grp['best_bid'].values,
            'best_ask': grp['best_ask'].values,
            'spread_bps': grp['spread_bps'].values,
            'imbalance': grp['imbalance'].values,
        }

    # Load trades for volume info
    print(f"  Loading trades...")
    tdir = os.path.join(events_dir, 'poly_trades')
    tchunks = []
    for f in sorted(glob.glob(os.path.join(tdir, '*.parquet'))):
        df = pq.read_table(f).to_pandas()
        df = df[df['token_id'].isin(tokens_needed)]
        if len(df) > 0:
            tchunks.append(df)
    trades = pd.concat(tchunks, ignore_index=True).sort_values('ts_us') if tchunks else pd.DataFrame()
    trades_by_token = {}
    if len(trades) > 0:
        for tid, grp in trades.groupby('token_id'):
            grp = grp.sort_values('ts_us')
            trades_by_token[tid] = {
                'ts_us': grp['ts_us'].values,
                'price': grp['price'].values.astype(float),
                'size': grp['size'].values.astype(float),
            }

    # Extract series per window
    results = []
    for slug, w in sorted(w15.items(), key=lambda x: x[1]['window_start_us']):
        token = w['up_token_id']
        bdata = books_by_token.get(token)
        if bdata is None:
            continue

        t0 = w['window_start_us']
        t1 = w['window_end_us']

        # Book data in this window
        mask = (bdata['ts_us'] >= t0) & (bdata['ts_us'] <= t1)
        if mask.sum() < 5:
            continue

        ts = bdata['ts_us'][mask]
        mids = bdata['mid'][mask]
        bids = bdata['best_bid'][mask]
        asks = bdata['best_ask'][mask]
        spreads = bdata['spread_bps'][mask]

        # Downsample to ~1 point per second (avoid huge JSON)
        step = max(1, len(ts) // 900)
        ts = ts[::step]
        mids = mids[::step]
        bids = bids[::step]
        asks = asks[::step]
        spreads = spreads[::step]

        # Relative time in seconds from window start
        rel_s = ((ts - t0) / 1e6).tolist()
        mids = [round(float(m), 4) for m in mids]
        bids = [round(float(b), 4) for b in bids]
        asks = [round(float(a), 4) for a in asks]

        # Trade data in this window
        trade_points = []
        tdata = trades_by_token.get(token)
        if tdata is not None:
            tmask = (tdata['ts_us'] >= t0) & (tdata['ts_us'] <= t1)
            if tmask.sum() > 0:
                t_ts = tdata['ts_us'][tmask]
                t_px = tdata['price'][tmask]
                t_sz = tdata['size'][tmask]
                # Downsample trades too
                t_step = max(1, len(t_ts) // 500)
                for i in range(0, len(t_ts), t_step):
                    trade_points.append({
                        't': round(float((t_ts[i] - t0) / 1e6), 2),
                        'p': round(float(t_px[i]), 4),
                        's': round(float(t_sz[i]), 2),
                    })

        # Compute flow in first 30s (same method as backtest: mid at trade time)
        flow = 0
        obs_vol = 0
        if tdata is not None:
            obs_end = t0 + 30_000_000
            obs_mask = (tdata['ts_us'] >= t0) & (tdata['ts_us'] <= obs_end)
            if obs_mask.sum() > 0:
                obs_ts = tdata['ts_us'][obs_mask]
                obs_prices = tdata['price'][obs_mask]
                obs_sizes = tdata['size'][obs_mask]
                # Classify each trade vs the book mid AT THAT TRADE'S TIME
                bk_ts_full = bdata['ts_us']
                bk_mid_full = bdata['mid']
                mid_idx = np.searchsorted(bk_ts_full, obs_ts, side='right') - 1
                mid_idx = np.clip(mid_idx, 0, len(bk_mid_full) - 1)
                mid_at_trade = bk_mid_full[mid_idx]
                is_buy = obs_prices >= mid_at_trade
                buy_vol = float(obs_sizes[is_buy].sum())
                sell_vol = float(obs_sizes[~is_buy].sum())
                total = buy_vol + sell_vol
                obs_vol = round(total, 1)
                flow = round((buy_vol - sell_vol) / total, 3) if total > 0 else 0

        # Compute imbalance in first 10s of book data
        imbalance = 0
        bk_ts_full = bdata['ts_us']
        i_lo = np.searchsorted(bk_ts_full, t0, side='left')
        i_hi = np.searchsorted(bk_ts_full, t0 + 10_000_000, side='right')
        if i_hi > i_lo:
            imbalance = round(float(np.mean(bdata['imbalance'][i_lo:i_hi])), 3)

        # Strategy signal: flow+imb thr=0.2
        if flow > 0.2 and imbalance > 0:
            fi_signal = 'BUY UP'
        elif flow < -0.2 and imbalance < 0:
            fi_signal = 'BUY DN'
        elif abs(flow) >= 0.2:
            fi_signal = 'SKIP (imb)'
        else:
            fi_signal = 'SKIP (flow)'

        # Compute Binance distance to target at entry time (t0 + 30s)
        bn_distance = 0
        bn_price = None
        coin = w['coin']
        bn_sym = COIN_TO_SYMBOL.get(coin, '')
        bn_data = bn_by_symbol.get(bn_sym)
        if bn_data is not None and w.get('target_price', 0) > 0:
            t_entry = t0 + 30_000_000
            bn_ts = bn_data['ts_us']
            bn_idx = np.searchsorted(bn_ts, t_entry, side='right') - 1
            if bn_idx >= 0:
                bn_price = float(bn_data['price'][bn_idx])
                bn_distance = round((bn_price - w['target_price']) / w['target_price'] * 10000, 1)

        # Mixed strategy: UP(fi+d>=0) or DOWN(d<=-5)
        if fi_signal == 'BUY UP' and bn_distance >= 0:
            signal = 'BUY UP'
        elif bn_distance <= -5:
            signal = 'BUY DN'
        else:
            signal = 'SKIP'

        # PnL: entry at first mid after 30s, settle at outcome
        entry_price = None
        dn_entry_price = None
        pnl = None
        # Find mid price just after 30s observation
        entry_idx = np.searchsorted(bdata['ts_us'], t0 + 30_000_000, side='left')
        if entry_idx < len(bdata['mid']):
            entry_price = round(float(bdata['mid'][entry_idx]), 4)
            dn_entry_price = round(1.0 - entry_price, 4)  # DOWN ≈ 1 - UP

        if signal == 'BUY UP' and entry_price is not None:
            settle_price = 1.0 if w['outcome'] == 1 else 0.0
            pnl = round((settle_price / entry_price - 1.0) * 50, 2)
        elif signal == 'BUY DN' and dn_entry_price is not None and dn_entry_price > 0:
            settle_price = 1.0 if w['outcome'] == 0 else 0.0
            pnl = round((settle_price / dn_entry_price - 1.0) * 50, 2)

        dt_start = datetime.fromtimestamp(t0 / 1e6, tz=timezone.utc)

        results.append({
            'slug': slug,
            'coin': w['coin'],
            'date': dt_start.strftime('%Y-%m-%d'),
            'time': dt_start.strftime('%H:%M'),
            'outcome': 'UP' if w['outcome'] == 1 else 'DOWN',
            'outcome_val': w['outcome'],
            'target': round(w['target_price'], 2),
            'flow_30s': flow,
            'imbalance': imbalance,
            'obs_vol': obs_vol,
            'fi_signal': fi_signal,
            'signal': signal,
            'bn_distance': bn_distance,
            'entry_price': entry_price if signal == 'BUY UP' else dn_entry_price,
            'pnl': pnl,
            'n_points': len(rel_s),
            'n_trades': len(trade_points),
            'ts': rel_s,
            'mid': mids,
            'bid': bids,
            'ask': asks,
            'trades': trade_points,
        })

    print(f"  Extracted {len(results)} windows with data")
    return results


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>15min Window Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0a0e17; --s1: #0f1523; --s2: #151d2e; --s3: #1a2540;
  --border: #1e2d4a; --border2: #283a5c;
  --green: #22d98e; --red: #ff4070; --gold: #f5c842;
  --blue: #4d94ff; --purple: #a06cff;
  --text: #d4dff5; --text2: #5a7099; --muted: #2a3a5a;
  --btc: #f7931a; --eth: #6b8ef0; --sol: #a855f7; --xrp: #00b4e4;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; }

.header {
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  background: var(--s1); display: flex; align-items: center; gap: 20px;
  flex-wrap: wrap;
}
.header h1 {
  font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700;
  letter-spacing: 0.08em; color: var(--gold);
}
.controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }

.pill-group { display: flex; gap: 0; }
.pill {
  padding: 6px 14px; font-size: 11px; font-family: 'DM Mono', monospace;
  font-weight: 500; letter-spacing: 0.06em; cursor: pointer;
  border: 1px solid var(--border2); background: var(--s2); color: var(--text2);
  transition: all 0.15s;
}
.pill:first-child { border-radius: 4px 0 0 4px; }
.pill:last-child { border-radius: 0 4px 4px 0; }
.pill:not(:first-child) { border-left: none; }
.pill.active { background: var(--gold); color: #000; border-color: var(--gold); }
.pill:hover:not(.active) { background: var(--s3); color: var(--text); }

.pill.coin-btc.active { background: var(--btc); border-color: var(--btc); }
.pill.coin-eth.active { background: var(--eth); border-color: var(--eth); color: #fff; }
.pill.coin-sol.active { background: var(--purple); border-color: var(--purple); color: #fff; }
.pill.coin-xrp.active { background: var(--xrp); border-color: var(--xrp); }

.nav-btn {
  width: 32px; height: 32px; border-radius: 4px; border: 1px solid var(--border2);
  background: var(--s2); color: var(--text); font-size: 14px; cursor: pointer;
  display: flex; align-items: center; justify-content: center; transition: 0.15s;
}
.nav-btn:hover { background: var(--s3); border-color: var(--gold); color: var(--gold); }
.nav-btn:disabled { opacity: 0.3; cursor: default; }

.window-label {
  font-size: 11px; color: var(--text2); min-width: 80px; text-align: center;
}

.chart-area {
  padding: 20px 24px; flex: 1; display: flex; flex-direction: column;
}
.info-bar {
  display: flex; gap: 24px; padding: 12px 16px; background: var(--s1);
  border: 1px solid var(--border); border-radius: 6px; margin-bottom: 16px;
  flex-wrap: wrap;
}
.info-item { display: flex; flex-direction: column; gap: 2px; }
.info-label { font-size: 9px; letter-spacing: 0.12em; color: var(--text2); text-transform: uppercase; }
.info-value { font-size: 14px; font-weight: 500; }
.info-value.up { color: var(--green); }
.info-value.down { color: var(--red); }
.info-value.neutral { color: var(--text); }

.chart-wrap {
  flex: 1; position: relative; background: var(--s1);
  border: 1px solid var(--border); border-radius: 6px; padding: 12px;
  min-height: 400px;
}

.slider-row {
  padding: 12px 24px; display: flex; align-items: center; gap: 12px;
  border-top: 1px solid var(--border); background: var(--s1);
}
.slider-row input[type=range] {
  flex: 1; accent-color: var(--gold); height: 4px;
}
.slider-label { font-size: 10px; color: var(--text2); min-width: 60px; }

.overlay-toggle {
  display: flex; align-items: center; gap: 6px; font-size: 11px;
  color: var(--text2); cursor: pointer;
}
.overlay-toggle input { accent-color: var(--gold); }

.legend-row {
  display: flex; gap: 16px; padding: 8px 24px; flex-wrap: wrap;
}
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: var(--text2); }
.legend-dot { width: 10px; height: 3px; border-radius: 1px; }
</style>
</head>
<body>

<div class="header">
  <h1>15min Window Viewer</h1>
  <div class="controls">
    <div class="pill-group" id="coin-pills"></div>
    <div class="pill-group" id="outcome-pills">
      <div class="pill active" data-val="all">ALL</div>
      <div class="pill" data-val="UP">UP</div>
      <div class="pill" data-val="DOWN">DN</div>
    </div>
    <div class="pill-group" id="signal-pills">
      <div class="pill active" data-val="all">ALL</div>
      <div class="pill" data-val="trade">TRADED</div>
      <div class="pill" data-val="skip">SKIP</div>
      <div class="pill" data-val="win">WIN</div>
      <div class="pill" data-val="lose">LOSE</div>
    </div>
    <button class="nav-btn" id="prev-btn" title="Previous">&#9664;</button>
    <div class="window-label" id="win-label">1 / 1</div>
    <button class="nav-btn" id="next-btn" title="Next">&#9654;</button>
    <label class="overlay-toggle">
      <input type="checkbox" id="show-trades" checked/> Trades
    </label>
    <label class="overlay-toggle">
      <input type="checkbox" id="show-spread"/> Bid/Ask
    </label>
  </div>
</div>

<div class="info-bar" id="info-bar"></div>
<div id="running-stats" style="padding:6px 24px;font-size:11px;color:var(--text2);background:var(--s2);border-bottom:1px solid var(--border)"></div>

<div class="chart-wrap">
  <canvas id="chart"></canvas>
</div>

<div class="legend-row">
  <div class="legend-item"><div class="legend-dot" style="background:#f5c842"></div> Mid price</div>
  <div class="legend-item"><div class="legend-dot" style="background:#22d98e"></div> Bid</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff4070"></div> Ask</div>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(77,148,255,0.6);border-radius:50%;width:8px;height:8px"></div> Trades</div>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(255,255,255,0.15)"></div> 30s obs</div>
</div>

<div class="slider-row">
  <div class="slider-label" id="slider-label">1 / 1</div>
  <input type="range" id="slider" min="0" max="0" value="0"/>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const COIN_COLORS = { btc: '#f7931a', eth: '#6b8ef0', sol: '#a855f7', xrp: '#00b4e4' };

// State
let currentCoin = null;
let outcomeFilter = 'all';
let signalFilter = 'all';
let filtered = [];
let idx = 0;
let chart = null;

// Discover coins
const coins = [...new Set(DATA.map(d => d.coin))].sort();

// Build coin pills
const coinPills = document.getElementById('coin-pills');
coins.forEach((c, i) => {
  const p = document.createElement('div');
  p.className = `pill coin-${c}` + (i === 0 ? ' active' : '');
  p.textContent = c.toUpperCase();
  p.dataset.val = c;
  p.onclick = () => selectCoin(c);
  coinPills.appendChild(p);
});

// Outcome pills
document.querySelectorAll('#outcome-pills .pill').forEach(p => {
  p.onclick = () => {
    document.querySelectorAll('#outcome-pills .pill').forEach(x => x.classList.remove('active'));
    p.classList.add('active');
    outcomeFilter = p.dataset.val;
    applyFilter();
  };
});

// Signal pills
document.querySelectorAll('#signal-pills .pill').forEach(p => {
  p.onclick = () => {
    document.querySelectorAll('#signal-pills .pill').forEach(x => x.classList.remove('active'));
    p.classList.add('active');
    signalFilter = p.dataset.val;
    applyFilter();
  };
});

function selectCoin(c) {
  currentCoin = c;
  document.querySelectorAll('#coin-pills .pill').forEach(p => {
    p.classList.toggle('active', p.dataset.val === c);
  });
  applyFilter();
}

function applyFilter() {
  filtered = DATA.filter(d => {
    if (currentCoin && d.coin !== currentCoin) return false;
    if (outcomeFilter !== 'all' && d.outcome !== outcomeFilter) return false;
    if (signalFilter === 'trade' && !d.signal.startsWith('BUY')) return false;
    if (signalFilter === 'skip' && d.signal.startsWith('BUY')) return false;
    if (signalFilter === 'win' && !(d.pnl !== null && d.pnl > 0)) return false;
    if (signalFilter === 'lose' && !(d.pnl !== null && d.pnl <= 0)) return false;
    return true;
  });
  idx = 0;
  document.getElementById('slider').max = Math.max(0, filtered.length - 1);
  document.getElementById('slider').value = 0;
  updateStats();
  renderCurrent();
}

function updateStats() {
  const traded = filtered.filter(d => d.pnl !== null);
  const totalPnl = traded.reduce((s, d) => s + d.pnl, 0);
  const wins = traded.filter(d => d.pnl > 0).length;
  const wr = traded.length > 0 ? (wins / traded.length * 100).toFixed(0) : '—';
  const el = document.getElementById('running-stats');
  if (traded.length > 0) {
    el.innerHTML = `<span style="color:var(--text2)">Filtered:</span> ${filtered.length} windows | `
      + `<span style="color:var(--text2)">Traded:</span> ${traded.length} | `
      + `<span style="color:${totalPnl >= 0 ? 'var(--green)' : 'var(--red)'}">PnL: $${totalPnl.toFixed(1)}</span> | `
      + `<span style="color:var(--text2)">WR:</span> ${wr}%`;
  } else {
    el.innerHTML = `<span style="color:var(--text2)">Filtered:</span> ${filtered.length} windows | <span style="color:var(--text2)">No trades</span>`;
  }
}

// Navigation
document.getElementById('prev-btn').onclick = () => { if (idx > 0) { idx--; sync(); } };
document.getElementById('next-btn').onclick = () => { if (idx < filtered.length - 1) { idx++; sync(); } };
document.getElementById('slider').oninput = (e) => { idx = parseInt(e.target.value); renderCurrent(); };
document.getElementById('show-trades').onchange = () => renderCurrent();
document.getElementById('show-spread').onchange = () => renderCurrent();

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' && idx > 0) { idx--; sync(); }
  if (e.key === 'ArrowRight' && idx < filtered.length - 1) { idx++; sync(); }
});

function sync() {
  document.getElementById('slider').value = idx;
  renderCurrent();
}

function renderCurrent() {
  if (filtered.length === 0) {
    document.getElementById('win-label').textContent = '0 / 0';
    document.getElementById('slider-label').textContent = '0 / 0';
    document.getElementById('info-bar').innerHTML = '<div class="info-item"><div class="info-value neutral">No windows match filters</div></div>';
    if (chart) { chart.destroy(); chart = null; }
    return;
  }

  const w = filtered[idx];
  const label = `${idx + 1} / ${filtered.length}`;
  document.getElementById('win-label').textContent = label;
  document.getElementById('slider-label').textContent = label;
  document.getElementById('prev-btn').disabled = idx === 0;
  document.getElementById('next-btn').disabled = idx >= filtered.length - 1;

  // Info bar
  const flowCls = w.flow_30s > 0.2 ? 'up' : w.flow_30s < -0.2 ? 'down' : 'neutral';
  const imbCls = w.imbalance > 0 ? 'up' : w.imbalance < 0 ? 'down' : 'neutral';
  const outCls = w.outcome === 'UP' ? 'up' : 'down';
  const coinColor = COIN_COLORS[w.coin] || '#fff';
  const distCls = w.bn_distance > 3 ? 'up' : w.bn_distance < -3 ? 'down' : 'neutral';
  const sigCls = w.signal.startsWith('BUY') ? (w.pnl > 0 ? 'up' : 'down') : 'neutral';
  const pnlStr = w.pnl !== null ? `$${w.pnl > 0 ? '+' : ''}${w.pnl.toFixed(1)}` : '—';
  const pnlCls = w.pnl !== null ? (w.pnl > 0 ? 'up' : 'down') : 'neutral';
  const entryStr = w.entry_price !== null ? w.entry_price.toFixed(4) : '—';
  const distStr = `${w.bn_distance > 0 ? '+' : ''}${w.bn_distance.toFixed(1)}bps`;

  document.getElementById('info-bar').innerHTML = `
    <div class="info-item"><div class="info-label">Coin</div><div class="info-value" style="color:${coinColor}">${w.coin.toUpperCase()}</div></div>
    <div class="info-item"><div class="info-label">Date</div><div class="info-value neutral">${w.date}</div></div>
    <div class="info-item"><div class="info-label">Time</div><div class="info-value neutral">${w.time} UTC</div></div>
    <div class="info-item"><div class="info-label">Outcome</div><div class="info-value ${outCls}">${w.outcome}</div></div>
    <div class="info-item" style="border-left:2px solid var(--border2);padding-left:12px"><div class="info-label">Flow 30s</div><div class="info-value ${flowCls}">${w.flow_30s > 0 ? '+' : ''}${w.flow_30s.toFixed(3)}</div></div>
    <div class="info-item"><div class="info-label">Imbalance</div><div class="info-value ${imbCls}">${w.imbalance > 0 ? '+' : ''}${w.imbalance.toFixed(3)}</div></div>
    <div class="info-item"><div class="info-label">BN Dist</div><div class="info-value ${distCls}" style="font-weight:700">${distStr}</div></div>
    <div class="info-item" style="border-left:2px solid var(--border2);padding-left:12px"><div class="info-label">FI Signal</div><div class="info-value neutral">${w.fi_signal}</div></div>
    <div class="info-item"><div class="info-label">MIX Signal</div><div class="info-value ${sigCls}" style="font-weight:700">${w.signal}</div></div>
    <div class="info-item"><div class="info-label">Entry</div><div class="info-value neutral">${entryStr}</div></div>
    <div class="info-item"><div class="info-label">PnL ($50)</div><div class="info-value ${pnlCls}" style="font-weight:700">${pnlStr}</div></div>
  `;

  renderChart(w);
}

function renderChart(w) {
  const showTrades = document.getElementById('show-trades').checked;
  const showSpread = document.getElementById('show-spread').checked;
  const ctx = document.getElementById('chart').getContext('2d');

  if (chart) chart.destroy();

  const datasets = [];

  // Mid price
  datasets.push({
    label: 'Mid',
    data: w.ts.map((t, i) => ({ x: t, y: w.mid[i] })),
    borderColor: '#f5c842',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1,
    fill: false,
    order: 1,
  });

  // Bid/Ask
  if (showSpread) {
    datasets.push({
      label: 'Bid',
      data: w.ts.map((t, i) => ({ x: t, y: w.bid[i] })),
      borderColor: 'rgba(34,217,142,0.4)',
      borderWidth: 0.8,
      pointRadius: 0,
      tension: 0.1,
      fill: false,
      order: 2,
    });
    datasets.push({
      label: 'Ask',
      data: w.ts.map((t, i) => ({ x: t, y: w.ask[i] })),
      borderColor: 'rgba(255,64,112,0.4)',
      borderWidth: 0.8,
      pointRadius: 0,
      tension: 0.1,
      fill: false,
      order: 2,
    });
  }

  // Trade scatter
  if (showTrades && w.trades.length > 0) {
    datasets.push({
      label: 'Trades',
      data: w.trades.map(tr => ({ x: tr.t, y: tr.p, r: Math.min(6, 1.5 + Math.sqrt(tr.s) * 0.3) })),
      type: 'bubble',
      backgroundColor: 'rgba(77,148,255,0.35)',
      borderColor: 'rgba(77,148,255,0.6)',
      borderWidth: 0.5,
      order: 0,
    });
  }

  chart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#151d2e',
          borderColor: '#283a5c',
          borderWidth: 1,
          titleFont: { family: 'DM Mono', size: 11 },
          bodyFont: { family: 'DM Mono', size: 11 },
          callbacks: {
            title: (items) => {
              const s = items[0]?.parsed?.x;
              if (s == null) return '';
              const min = Math.floor(s / 60);
              const sec = Math.floor(s % 60);
              return `T+${min}:${sec.toString().padStart(2, '0')}`;
            },
            label: (item) => ` ${item.dataset.label}: ${item.parsed.y.toFixed(4)}`,
          },
        },
        // 30s observation zone
        annotation: undefined,
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Seconds from window start', color: '#5a7099', font: { family: 'DM Mono', size: 10 } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: {
            color: '#5a7099',
            font: { family: 'DM Mono', size: 10 },
            callback: (v) => {
              const min = Math.floor(v / 60);
              const sec = v % 60;
              return sec === 0 ? `${min}m` : '';
            },
            stepSize: 60,
          },
          min: 0,
          max: 900,
        },
        y: {
          title: { display: true, text: 'UP Token Price', color: '#5a7099', font: { family: 'DM Mono', size: 10 } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#5a7099', font: { family: 'DM Mono', size: 10 } },
        },
      },
    },
    plugins: [{
      id: 'obsZone',
      beforeDraw: (chart) => {
        const { ctx, chartArea, scales } = chart;
        if (!chartArea) return;
        const x0 = scales.x.getPixelForValue(0);
        const x1 = scales.x.getPixelForValue(30);
        ctx.save();
        ctx.fillStyle = 'rgba(245,200,66,0.06)';
        ctx.fillRect(x0, chartArea.top, x1 - x0, chartArea.bottom - chartArea.top);
        ctx.strokeStyle = 'rgba(245,200,66,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, chartArea.top);
        ctx.lineTo(x1, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(245,200,66,0.5)';
        ctx.font = '9px "DM Mono"';
        ctx.fillText('30s obs', x1 + 4, chartArea.top + 12);
        ctx.restore();
      },
    }],
  });
}

// Boot
selectCoin(coins[0]);
</script>
</body>
</html>"""


def main(events_dir, output_path="window_viewer.html"):
    print(f"\n  15min Window Viewer — generating...")
    t0 = time.time()

    windows = load_markets(events_dir)
    series = extract_window_series(events_dir, windows)

    if not series:
        print("  No data to display!")
        return

    # Sort by coin then time
    series.sort(key=lambda d: (d['coin'], d['date'], d['time']))

    # Embed as JSON
    json_data = json.dumps(series, separators=(',', ':'))
    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Written: {output_path} ({size_mb:.1f} MB)")
    print(f"  {len(series)} windows, done in {time.time()-t0:.1f}s")

    # Try to open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
        print(f"  Opened in browser")
    except Exception:
        print(f"  Open {output_path} in your browser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('-o', '--output', default='window_viewer.html')
    args = parser.parse_args()
    main(args.events_dir, args.output)