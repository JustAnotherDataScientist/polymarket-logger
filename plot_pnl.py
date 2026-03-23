#!/usr/bin/env python3
"""
plot_pnl.py — Cumulative PnL curves for all explored strategies.

Usage:
  python plot_pnl.py historical_data/events
"""

import sys, os, glob, json, argparse, time, webbrowser
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from backtest_allcoins import (
    load_all_markets, load_books_and_trades, load_bn_trades,
    cache_fills, compute_all_signals, evaluate, COIN_TO_SYMBOL,
)


def build_series(trades_list, label):
    if not trades_list:
        return {'label': label, 'points': []}
    trades_list = sorted(trades_list, key=lambda t: t['start_us'])
    cum_pnl = 0
    points = []
    for t in trades_list:
        cum_pnl += t['pnl']
        dt = datetime.fromtimestamp(t['start_us'] / 1e6, tz=timezone.utc)
        points.append({
            'ts': t['start_us'] // 1000,
            'date': dt.strftime('%Y-%m-%d %H:%M'),
            'pnl': round(t['pnl'], 2),
            'cum': round(cum_pnl, 2),
            'coin': t.get('coin', ''),
            'direction': t.get('direction', ''),
            'entry': round(t.get('entry', 0), 4),
        })
    return {'label': label, 'points': points}


def make_stats(trades_list):
    if not trades_list:
        return {'n': 0, 'total': 0, 'avg': 0, 'wr': 0, 'sharpe': 0, 'maxdd': 0}
    pnls = np.array([t['pnl'] for t in trades_list])
    wins = int((pnls > 0).sum())
    cum = np.cumsum(pnls)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {
        'n': len(pnls), 'total': round(float(pnls.sum()), 2),
        'avg': round(float(pnls.mean()), 2),
        'wr': round(wins / len(pnls) * 100, 1),
        'sharpe': round(float(pnls.mean() / pnls.std()), 3) if pnls.std() > 0 else 0,
        'maxdd': round(dd, 2),
    }


def main(events_dir, output_path="pnl_viewer.html"):
    print(f"\n  PnL Curve Viewer — generating...")
    t0 = time.time()

    coins = ['btc', 'eth', 'sol']

    # Load data
    all_windows, mdf = load_all_markets(events_dir)
    all_tokens = set()
    coin_windows = {}
    for coin in coins:
        cw = {s: w for s, w in all_windows.items() if w['coin'] == coin}
        coin_windows[coin] = cw
        for w in cw.values():
            all_tokens.add(w['up_token'])
            all_tokens.add(w['down_token'])

    print(f"  Loading books + trades...")
    books, trades = load_books_and_trades(events_dir, all_tokens)
    print(f"  Loading Binance trades...")
    bn_by_symbol = load_bn_trades(events_dir)

    # Compute per coin
    coin_data = {}
    for coin in coins:
        resolved = coin_windows[coin]
        if not resolved:
            continue
        ct_set = set()
        for w in resolved.values():
            ct_set.add(w['up_token']); ct_set.add(w['down_token'])
        cb = books[books['token_id'].isin(ct_set)] if len(books) > 0 else books
        ctr = trades[trades['token_id'].isin(ct_set)] if len(trades) > 0 else trades
        bn = bn_by_symbol.get(COIN_TO_SYMBOL.get(coin, ''))
        fills = cache_fills(resolved, cb)
        signals = compute_all_signals(resolved, cb, ctr, bn_data=bn)
        coin_data[coin] = (resolved, fills, signals)

    # Merge
    m_res, m_fills, m_sig = {}, {}, {obs: {} for obs in [15, 30, 45, 60]}
    for coin in coins:
        if coin not in coin_data: continue
        r, f, s = coin_data[coin]
        m_res.update(r); m_fills.update(f)
        for obs in [15, 30, 45, 60]:
            if obs in s: m_sig[obs].update(s[obs])

    # Define all strategies
    fi15 = dict(obs_secs=30, flow_threshold=0.2, use_imbalance=True, interval_filter=15)
    b15 = dict(obs_secs=30, interval_filter=15)

    strategy_defs = {
        # UP strategies
        'fi15':       ('UP: flow+imb (base)',       '#f5c842', None,   fi15),
        'flow15':     ('UP: flow only',             '#5a7099', [6,3],  dict(obs_secs=30, flow_threshold=0.2, interval_filter=15)),
        'fi15_d0':    ('UP: fi + d>=0',             '#22d98e', None,   dict(**fi15, bn_distance_min=0)),
        'fi15_d3':    ('UP: fi + d>=+3bps',         '#4d94ff', None,   dict(**fi15, bn_distance_min=3)),
        'fi15_d5':    ('UP: fi + d>=+5bps',         '#a06cff', None,   dict(**fi15, bn_distance_min=5)),
        # DOWN strategies
        'dn_d3':      ('DOWN: d<=-3bps',            '#ff6b6b', [4,4],  dict(**b15, force_direction="down", bn_distance_max_signed=-3)),
        'dn_d5':      ('DOWN: d<=-5bps',            '#ff4070', None,   dict(**b15, force_direction="down", bn_distance_max_signed=-5)),
        'dn_d7':      ('DOWN: d<=-7bps',            '#cc3360', None,   dict(**b15, force_direction="down", bn_distance_max_signed=-7)),
        # Losers for reference
        'fi15_below': ('UP: fi + d<=0 (bad)',       '#664400', [4,4],  dict(**fi15, bn_distance_max_signed=0)),
        'flow5':      ('5m flow (bad)',             '#663333', [3,3],  dict(obs_secs=30, flow_threshold=0.2, interval_filter=5)),
    }

    # Evaluate all
    strat_results = {}
    for key, (label, color, dash, kwargs) in strategy_defs.items():
        strat_results[key] = evaluate(m_res, m_fills, m_sig, **kwargs)

    for coin in coins:
        r, f, s = coin_data[coin]
        strat_results[f'coin_{coin}'] = evaluate(r, f, s, **fi15)

    # Combined strategies (merge trade lists)
    def combined(up_key, dn_key, label, color):
        trades = sorted(strat_results.get(up_key, []) + strat_results.get(dn_key, []),
                        key=lambda t: t['start_us'])
        return label, color, None, trades

    combined_defs = {
        'mix_safe':  combined('fi15_d0', 'dn_d5', 'MIX: UP(fi+d>=0) + DN(d<=-5)', '#00ffaa'),
        'mix_tight': combined('fi15_d3', 'dn_d5', 'MIX: UP(fi+d>=3) + DN(d<=-5)', '#00ccff'),
        'mix_aggro': combined('fi15',    'dn_d3', 'MIX: UP(fi) + DN(d<=-3)',       '#ffaa00'),
    }

    # Build JSON
    series, summary = {}, {}
    for key, (label, color, dash, kwargs) in strategy_defs.items():
        series[key] = build_series(strat_results[key], label)
        summary[key] = make_stats(strat_results[key])
    for key, (label, color, dash, trades) in combined_defs.items():
        strat_results[key] = trades
        series[key] = build_series(trades, label)
        summary[key] = make_stats(trades)
    for coin in coins:
        k = f'coin_{coin}'
        series[k] = build_series(strat_results[k], f'{coin.upper()} flow+imb')
        summary[k] = make_stats(strat_results[k])

    colors = {k: v[1] for k, v in strategy_defs.items()}
    colors.update({k: v[1] for k, v in combined_defs.items()})
    colors.update({'coin_btc': '#f7931a', 'coin_eth': '#6b8ef0', 'coin_sol': '#a855f7'})
    dashes = {k: v[2] for k, v in strategy_defs.items()}

    json_data = json.dumps({
        'series': series, 'summary': summary, 'colors': colors, 'dashes': dashes,
    }, separators=(',', ':'))

    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    n_strats = sum(1 for s in summary.values() if s['n'] > 0)
    print(f"  Written: {output_path} ({n_strats} strategies)")
    print(f"  Done in {time.time()-t0:.1f}s")

    try:
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
    except Exception:
        pass


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PnL Curves — Strategy Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0a0e17; --s1: #0f1523; --s2: #151d2e;
  --border: #1e2d4a; --border2: #283a5c;
  --green: #22d98e; --red: #ff4070; --gold: #f5c842;
  --blue: #4d94ff; --purple: #a06cff;
  --text: #d4dff5; --text2: #5a7099;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; display: flex; flex-direction: column; height: 100vh; }
.header { padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--s1); display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; letter-spacing: 0.08em; color: var(--gold); }
.stats-row { display: flex; gap: 10px; padding: 10px 24px; background: var(--s1); border-bottom: 1px solid var(--border); flex-wrap: wrap; overflow-x: auto; }
.stat-card { padding: 8px 12px; background: var(--s2); border: 1px solid var(--border); border-radius: 6px; min-width: 120px; flex-shrink: 0; }
.stat-label { font-size: 8px; letter-spacing: 0.1em; color: var(--text2); text-transform: uppercase; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px; }
.stat-value { font-size: 15px; font-weight: 500; }
.stat-sub { font-size: 9px; color: var(--text2); margin-top: 2px; }
.controls { display: flex; align-items: center; gap: 12px; padding: 8px 24px; border-bottom: 1px solid var(--border); background: var(--s2); flex-wrap: wrap; }
.pill-group { display: flex; gap: 0; }
.pill { padding: 5px 12px; font-size: 10px; font-family: 'DM Mono', monospace; font-weight: 500; letter-spacing: 0.04em; cursor: pointer; border: 1px solid var(--border2); background: var(--s2); color: var(--text2); transition: all 0.15s; white-space: nowrap; }
.pill:first-child { border-radius: 4px 0 0 4px; }
.pill:last-child { border-radius: 0 4px 4px 0; }
.pill:not(:first-child) { border-left: none; }
.pill.active { background: var(--gold); color: #000; border-color: var(--gold); }
.pill:hover:not(.active) { background: var(--s1); color: var(--text); }
.view-label { font-size: 10px; color: var(--text2); }
.chart-wrap { flex: 1; padding: 16px 24px; position: relative; min-height: 0; }
.chart-wrap canvas { width: 100% !important; height: 100% !important; }
.trade-table-wrap { max-height: 180px; overflow-y: auto; border-top: 1px solid var(--border); background: var(--s1); padding: 0 24px; }
table { width: 100%; border-collapse: collapse; font-size: 10px; }
th { position: sticky; top: 0; background: var(--s1); color: var(--text2); text-align: left; padding: 5px 8px; font-weight: 500; letter-spacing: 0.08em; border-bottom: 1px solid var(--border); }
td { padding: 3px 8px; border-bottom: 1px solid var(--border); color: var(--text); }
tr:hover td { background: var(--s2); }
.win { color: var(--green); } .lose { color: var(--red); }
</style>
</head>
<body>
<div class="header"><h1>Cumulative PnL — Strategy Comparison</h1></div>
<div class="stats-row" id="stats-row"></div>
<div class="controls">
  <span class="view-label">View:</span>
  <div class="pill-group" id="view-pills">
    <div class="pill active" data-val="main">MAIN</div>
    <div class="pill" data-val="mixed">MIXED</div>
    <div class="pill" data-val="up">UP ONLY</div>
    <div class="pill" data-val="down">DOWN ONLY</div>
    <div class="pill" data-val="per_coin">PER COIN</div>
    <div class="pill" data-val="all">ALL</div>
  </div>
  <span class="view-label" style="margin-left:12px">Size:</span>
  <div class="pill-group" id="size-pills">
    <div class="pill" data-val="5">$5</div>
    <div class="pill active" data-val="50">$50</div>
    <div class="pill" data-val="100">$100</div>
  </div>
</div>
<div class="chart-wrap"><canvas id="chart"></canvas></div>
<div class="trade-table-wrap" id="table-wrap"></div>

<script>
const RAW = __DATA_PLACEHOLDER__;
const VIEWS = {
  main:     ['fi15', 'fi15_d0', 'dn_d5', 'mix_safe'],
  mixed:    ['mix_safe', 'mix_tight', 'mix_aggro', 'fi15'],
  up:       ['fi15', 'fi15_d0', 'fi15_d3', 'fi15_d5'],
  down:     ['dn_d3', 'dn_d5', 'dn_d7'],
  per_coin: ['coin_btc', 'coin_eth', 'coin_sol'],
  all:      ['fi15', 'fi15_d0', 'fi15_d3', 'dn_d5', 'dn_d7', 'mix_safe', 'mix_tight'],
};
let currentView = 'main', sizeMultiplier = 1, chart = null;

document.querySelectorAll('#view-pills .pill').forEach(p => {
  p.onclick = () => { document.querySelectorAll('#view-pills .pill').forEach(x => x.classList.remove('active')); p.classList.add('active'); currentView = p.dataset.val; render(); };
});
document.querySelectorAll('#size-pills .pill').forEach(p => {
  p.onclick = () => { document.querySelectorAll('#size-pills .pill').forEach(x => x.classList.remove('active')); p.classList.add('active'); sizeMultiplier = parseFloat(p.dataset.val) / 50; render(); };
});

function render() { renderStats(); renderChart(); renderTable(); }

function renderStats() {
  const keys = VIEWS[currentView] || [];
  let html = '';
  for (const k of keys) {
    const s = RAW.summary[k]; if (!s || !s.n) continue;
    const total = (s.total * sizeMultiplier).toFixed(1);
    const color = RAW.colors[k] || '#fff';
    const cls = parseFloat(total) >= 0 ? 'win' : 'lose';
    html += `<div class="stat-card">
      <div class="stat-label" style="color:${color}">${RAW.series[k]?.label || k}</div>
      <div class="stat-value ${cls}">$${total}</div>
      <div class="stat-sub">${s.n} trades | WR ${s.wr}% | avg $${(s.avg*sizeMultiplier).toFixed(2)}</div>
      <div class="stat-sub">Sharpe ${s.sharpe} | DD $${(s.maxdd*sizeMultiplier).toFixed(1)}</div>
    </div>`;
  }
  document.getElementById('stats-row').innerHTML = html;
}

function renderChart() {
  const ctx = document.getElementById('chart').getContext('2d');
  if (chart) chart.destroy();
  const keys = VIEWS[currentView] || [];
  const datasets = [];
  for (const k of keys) {
    const s = RAW.series[k]; if (!s || !s.points.length) continue;
    const color = RAW.colors[k] || '#fff';
    const dash = RAW.dashes[k] || null;
    let cum = 0;
    const data = s.points.map(p => { cum += p.pnl * sizeMultiplier; return { x: p.ts, y: Math.round(cum*100)/100 }; });
    data.unshift({ x: data[0].x - 1, y: 0 });
    datasets.push({ label: s.label, data, borderColor: color, borderWidth: dash ? 1.2 : 1.8, borderDash: dash || [], pointRadius: 0, pointHitRadius: 8, tension: 0, fill: false });
  }
  chart = new Chart(ctx, {
    type: 'line', data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { display: true, position: 'top', labels: { color: '#5a7099', font: { family: 'DM Mono', size: 10 }, boxWidth: 20, padding: 12 } },
        tooltip: { backgroundColor: '#151d2e', borderColor: '#283a5c', borderWidth: 1, titleFont: { family: 'DM Mono', size: 10 }, bodyFont: { family: 'DM Mono', size: 10 },
          callbacks: { title: (i) => { const ts = i[0]?.parsed?.x; return ts ? new Date(ts).toUTCString().slice(0,22) : ''; }, label: (i) => ` ${i.dataset.label}: $${i.parsed.y.toFixed(1)}` } },
      },
      scales: {
        x: { type: 'linear', grid: { color: 'rgba(30,45,74,0.5)' }, ticks: { color: '#5a7099', font: { family: 'DM Mono', size: 9 }, callback: (v) => { const d = new Date(v); return d.toISOString().slice(5,10)+' '+d.toISOString().slice(11,16); }, maxTicksLimit: 12 } },
        y: { title: { display: true, text: 'Cumulative PnL ($)', color: '#5a7099', font: { family: 'DM Mono', size: 10 } }, grid: { color: 'rgba(30,45,74,0.5)' }, ticks: { color: '#5a7099', font: { family: 'DM Mono', size: 10 }, callback: (v) => '$'+v.toFixed(0) } },
      },
    },
    plugins: [{ id: 'zeroLine', beforeDraw: (c) => { const {ctx,chartArea,scales}=c; if(!chartArea)return; const y0=scales.y.getPixelForValue(0); if(y0<chartArea.top||y0>chartArea.bottom)return; ctx.save(); ctx.strokeStyle='rgba(255,255,255,0.12)'; ctx.lineWidth=1; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(chartArea.left,y0); ctx.lineTo(chartArea.right,y0); ctx.stroke(); ctx.setLineDash([]); ctx.restore(); } }],
  });
}

function renderTable() {
  const wrap = document.getElementById('table-wrap');
  const keys = VIEWS[currentView] || []; const mainKey = keys[0];
  const s = RAW.series[mainKey]; if (!s || !s.points.length) { wrap.innerHTML = ''; return; }
  const coinColors = { btc: '#f7931a', eth: '#6b8ef0', sol: '#a855f7' };
  let cum = 0;
  let rows = s.points.map(p => { cum += p.pnl * sizeMultiplier; return { ...p, cum_scaled: cum }; }).reverse();
  const total = rows.length;
  let html = '<table><thead><tr><th>#</th><th>Date</th><th>Coin</th><th>Dir</th><th>Entry</th><th>PnL</th><th>Cum</th></tr></thead><tbody>';
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]; const pnl = (r.pnl*sizeMultiplier).toFixed(1); const cls = r.pnl >= 0 ? 'win' : 'lose';
    html += `<tr><td>${total-i}</td><td>${r.date}</td><td style="color:${coinColors[r.coin]||'#fff'}">${(r.coin||'').toUpperCase()}</td><td>${(r.direction||'').toUpperCase()}</td><td>${r.entry}</td><td class="${cls}">$${pnl}</td><td class="${r.cum_scaled>=0?'win':'lose'}">$${r.cum_scaled.toFixed(1)}</td></tr>`;
  }
  html += '</tbody></table>'; wrap.innerHTML = html;
}
render();
</script>
</body>
</html>"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('events_dir', nargs='?', default='historical_data/events')
    parser.add_argument('-o', '--output', default='pnl_viewer.html')
    args = parser.parse_args()
    main(args.events_dir, args.output)