"""
plot_lead_lag.py — Graphique interactif BTC Binance vs Polymarket UP token.
Zoom avec molette souris, pan avec clic-glisser, reset double-clic.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def load(data_dir, coin, interval, minutes=0):
    p = Path(data_dir)
    files = sorted(p.glob("*_*.parquet")) + sorted(p.glob("[0-9]*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["coin"] == coin) & (df["interval_min"] == str(interval))].copy()
    for col in ["coin_price","up_ask","up_bid","up_mid","coin_return_10s_bps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if minutes > 0:
        cutoff = df["timestamp_utc"].max() - pd.Timedelta(minutes=minutes)
        df = df[df["timestamp_utc"] >= cutoff].reset_index(drop=True)
    return df

def build_html(df, coin, interval):
    # Tous les ticks (pas de sous-échantillonnage — zoom permettra de naviguer)
    ts   = [int(t.timestamp() * 1000) for t in df["timestamp_utc"]]
    price = df["coin_price"].tolist()
    # Normalisation sur fenêtre glissante de 120s (480 ticks @ 4Hz)
    # → amplifie les petits moves, centrés autour de 0.5
    price_s = pd.Series(price)
    win = 480
    roll_min = price_s.rolling(win, min_periods=1).min()
    roll_max = price_s.rolling(win, min_periods=1).max()
    roll_range = (roll_max - roll_min).replace(0, np.nan)
    price_norm_s = (price_s - roll_min) / roll_range
    price_norm = [round(v, 5) if not np.isnan(v) else None
                  for v in price_norm_s.tolist()]

    ask = [round(v, 4) if not np.isnan(v) else None for v in df["up_ask"]]
    bid = [round(v, 4) if not np.isnan(v) else None for v in df["up_bid"]]
    mid = [round(v, 4) if not np.isnan(v) else None for v in df["up_mid"]]
    ret = [round(v, 2) if not np.isnan(v) else 0 for v in df.get("coin_return_10s_bps", pd.Series([0]*len(df)))]

    t0 = df["timestamp_utc"].min().strftime("%H:%M")
    t1 = df["timestamp_utc"].max().strftime("%H:%M")
    n  = len(df)
    mid_mean = round(df["up_mid"].mean(), 4)
    n_moves = int((df["coin_return_10s_bps"].abs() >= 15).sum()) if "coin_return_10s_bps" in df.columns else 0

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Lead-Lag {coin.upper()} {interval}m</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-moment/1.0.1/chartjs-adapter-moment.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; color: #e6edf3; font-family: system-ui, -apple-system, sans-serif; padding: 16px; }}
h2 {{ font-size: 12px; font-weight: 400; color: #8b949e; margin-bottom: 4px; letter-spacing: 0.05em; text-transform: uppercase; }}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
.stats {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
.stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; min-width: 120px; }}
.stat-label {{ font-size: 11px; color: #8b949e; margin-bottom: 3px; }}
.stat-value {{ font-size: 17px; font-weight: 600; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
.legend {{ display: flex; gap: 18px; margin-bottom: 10px; flex-wrap: wrap; }}
.leg {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }}
.dot {{ width: 20px; height: 2px; border-radius: 1px; }}
.controls {{ display: flex; gap: 8px; margin-bottom: 10px; align-items: center; }}
button {{ background: #21262d; border: 1px solid #30363d; color: #e6edf3; padding: 5px 12px;
          border-radius: 6px; cursor: pointer; font-size: 12px; }}
button:hover {{ background: #30363d; }}
.hint {{ font-size: 11px; color: #484f58; margin-top: 6px; }}
</style>
</head>
<body>
<h2>Analyse Lead-Lag</h2>
<h1>{coin.upper()} {interval}min — Binance vs Polymarket</h1>

<div class="stats">
  <div class="stat"><div class="stat-label">Période</div><div class="stat-value">{t0} → {t1} UTC</div></div>
  <div class="stat"><div class="stat-label">Ticks</div><div class="stat-value">{n:,}</div></div>
  <div class="stat"><div class="stat-label">UP mid moyen</div><div class="stat-value">{mid_mean:.4f}</div></div>
  <div class="stat"><div class="stat-label">Moves ≥15bps/10s</div><div class="stat-value">{n_moves}</div></div>
</div>

<div class="card">
  <div class="controls">
    <button onclick="resetZoom('main')">Reset zoom</button>
    <button onclick="zoomLast('main', 60)">1 min</button>
    <button onclick="zoomLast('main', 180)">3 min</button>
    <button onclick="zoomLast('main', 600)">10 min</button>
    <span style="font-size:12px;color:#8b949e;margin-left:8px;">Molette = zoom · Clic-glisser = pan</span>
  </div>
  <div class="legend">
    <div class="leg"><div class="dot" style="background:#f0883e;"></div>BTC Binance (normalisé fenêtre 2min)</div>
    <div class="leg"><div class="dot" style="background:#58a6ff;"></div>UP ask</div>
    <div class="leg"><div class="dot" style="background:#3fb950;"></div>UP bid</div>
    <div class="leg"><div class="dot" style="background:#888;border-top:1px dashed #888;"></div>UP mid</div>
  </div>
  <div style="position: relative; height: 340px;">
    <canvas id="mainChart"></canvas>
  </div>
  <div class="hint">Double-clic pour reset · Zoom avec molette · Pan avec clic-glisser</div>
</div>

<div class="card">
  <div class="controls">
    <button onclick="resetZoom('ret')">Reset zoom</button>
    <span style="font-size:12px;color:#8b949e;margin-left:8px;">Retour Binance 10s (bps) — axe X synchronisé</span>
  </div>
  <div style="position: relative; height: 140px;">
    <canvas id="retChart"></canvas>
  </div>
</div>

<script>
const ts   = {json.dumps(ts)};
const pn   = {json.dumps(price_norm)};
const ask  = {json.dumps(ask)};
const bid  = {json.dumps(bid)};
const mid  = {json.dumps(mid)};
const ret  = {json.dumps(ret)};

const mkPts = (vals) => ts.map((t,i) => ({{x:t, y:vals[i]}}));

const zoomPlugin = {{
  pan: {{ enabled: true, mode: 'x' }},
  zoom: {{
    wheel: {{ enabled: true }},
    pinch: {{ enabled: true }},
    mode: 'x',
    onZoom: syncCharts,
    onZoomComplete: syncCharts,
  }},
}};

const gridColor = '#21262d';
const tickColor = '#8b949e';

const mainChart = new Chart(document.getElementById('mainChart'), {{
  type: 'line',
  data: {{
    datasets: [
      {{ label:'BTC norm', data:mkPts(pn),  borderColor:'#f0883e', borderWidth:1.5, pointRadius:0, tension:0 }},
      {{ label:'UP ask',   data:mkPts(ask), borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, tension:0 }},
      {{ label:'UP bid',   data:mkPts(bid), borderColor:'#3fb950', borderWidth:1.5, pointRadius:0, tension:0 }},
      {{ label:'UP mid',   data:mkPts(mid), borderColor:'#555',    borderWidth:1,   pointRadius:0, tension:0, borderDash:[3,3] }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:false,
    interaction: {{ mode:'index', intersect:false }},
    scales: {{
      x: {{ type:'time', time:{{unit:'second', displayFormats:{{second:'HH:mm:ss'}}}},
            grid:{{color:gridColor}}, ticks:{{color:tickColor, maxTicksLimit:10}} }},
      y: {{ min:0, max:1, grid:{{color:gridColor}}, ticks:{{color:tickColor, stepSize:0.1}} }}
    }},
    plugins: {{
      legend:{{display:false}},
      tooltip:{{
        callbacks:{{
          label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.y?.toFixed(4)}}`
        }}
      }},
      zoom: zoomPlugin
    }}
  }}
}});

const retChart = new Chart(document.getElementById('retChart'), {{
  type:'bar',
  data: {{
    datasets: [{{
      label:'Retour 10s (bps)',
      data: ts.map((t,i) => ({{x:t, y:ret[i]}})),
      backgroundColor: ret.map(v => v>=0 ? '#3fb95066' : '#f7816666'),
      borderWidth:0,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:false,
    scales: {{
      x: {{ type:'time', time:{{unit:'second', displayFormats:{{second:'HH:mm:ss'}}}},
            grid:{{color:gridColor}}, ticks:{{color:tickColor, maxTicksLimit:10}} }},
      y: {{ grid:{{color:gridColor}}, ticks:{{color:tickColor}} }}
    }},
    plugins: {{
      legend:{{display:false}},
      zoom: zoomPlugin
    }}
  }}
}});

const charts = {{ main: mainChart, ret: retChart }};

function syncCharts(evt) {{
  const src = evt.chart;
  const {{min, max}} = src.scales.x;
  [mainChart, retChart].forEach(c => {{
    if (c === src) return;
    c.zoomScale('x', {{min, max}}, 'none');
  }});
}}

function resetZoom(id) {{
  mainChart.resetZoom();
  retChart.resetZoom();
}}

function zoomLast(id, seconds) {{
  const maxTs = ts[ts.length-1];
  const minTs = maxTs - seconds*1000;
  mainChart.zoomScale('x', {{min:minTs, max:maxTs}}, 'none');
  retChart.zoomScale('x',  {{min:minTs, max:maxTs}}, 'none');
}}

mainChart.canvas.ondblclick = () => resetZoom('main');
retChart.canvas.ondblclick  = () => resetZoom('ret');
</script>
</body>
</html>"""
    return html

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",      default="historical_data/live")
    p.add_argument("--coin",     default="btc")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--minutes",  type=int, default=0)
    p.add_argument("--out",      default="/home/ubuntu/pm_watcher/lead_lag.html")
    args = p.parse_args()

    df = load(args.dir, args.coin, args.interval, args.minutes)
    print(f"Chargé : {len(df):,} ticks {args.coin}-{args.interval}m")
    html = build_html(df, args.coin, args.interval)
    Path(args.out).write_text(html)
    print(f"Graphique → {args.out}")

if __name__ == "__main__":
    main()
