"""
dashboard.py — PM·BN Watcher
Single-port HTTP+WS server (port 8080).
Nginx proxies /ws → WS, everything else → HTTP.
Tabs: LIVE | DATA | LOGS
"""
from __future__ import annotations

import http.server
import json
import logging
import queue
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from config import DASHBOARD_HOST, DASHBOARD_HTTP_PORT, DASHBOARD_PORT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _data_stats() -> dict:
    """Summary of collected parquet data."""
    try:
        import pandas as pd
        live_dir = Path("historical_data/live")
        if not live_dir.exists():
            return {"total_rows": 0, "files": [], "markets": {}}

        files = sorted(live_dir.glob("*.parquet"))
        total = 0
        markets: dict[str, int] = {}
        dates = []

        for f in files:
            try:
                df = pd.read_parquet(f, columns=["coin", "interval_min", "timestamp_utc"])
                total += len(df)
                dates.append(str(f.stem))
                for (coin, iv), grp in df.groupby(["coin", "interval_min"]):
                    k = f"{coin}-{int(iv)}m"
                    markets[k] = markets.get(k, 0) + len(grp)
            except Exception:
                pass

        return {
            "total_rows":  total,
            "files":       [f.name for f in files],
            "dates":       dates,
            "markets":     markets,
            "date_range":  [dates[0], dates[-1]] if dates else [],
        }
    except Exception as e:
        return {"error": str(e)}


def _data_query(market: str, date: str, limit: int, offset: int) -> dict:
    """Paginated rows for a market on a given date."""
    try:
        import pandas as pd
        live_dir = Path("historical_data/live")
        if not live_dir.exists():
            return {"rows": [], "total": 0}

        files = []
        if date:
            f = live_dir / f"{date}.parquet"
            if f.exists():
                files = [f]
        else:
            files = sorted(live_dir.glob("*.parquet"))

        if not files:
            return {"rows": [], "total": 0}

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception:
                pass

        if not dfs:
            return {"rows": [], "total": 0}

        df = pd.concat(dfs, ignore_index=True)

        if market and market != "all":
            parts = market.split("-")
            if len(parts) == 2:
                coin, iv = parts[0], parts[1].replace("m", "")
                df = df[(df["coin"] == coin) & (df["interval_min"].astype(str) == iv)]

        df = df.sort_values("timestamp_utc", ascending=False)
        total = len(df)
        page  = df.iloc[offset: offset + limit]

        # Only send key columns to keep payload small
        cols = [
            "timestamp_utc", "coin", "interval_min", "slug",
            "seconds_remaining", "window_progress",
            "coin_price", "target_price", "distance_bps",
            "z_score", "analytical_prob_up", "edge_bps",
            "up_mid", "up_bid", "up_ask", "up_spread_bps", "up_depth_imbalance",
            "coin_return_60s_bps", "coin_flow_imbalance_60s",
            "poly_trade_count_60s", "poly_mid_velocity",
            "outcome",
        ]
        existing_cols = [c for c in cols if c in page.columns]
        page = page[existing_cols]

        return {
            "rows":    page.where(page.notna(), None).to_dict(orient="records"),
            "total":   total,
            "columns": existing_cols,
        }
    except Exception as e:
        logger.error("data_query error: %s", e)
        return {"rows": [], "total": 0, "error": str(e)}


def _log_tail(n: int = 100) -> list[str]:
    """Last N lines from the log file."""
    try:
        from config import LOG_FILE
        p = Path(LOG_FILE)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PM·BN Watcher</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=JetBrains+Mono:wght@300;400;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#04070f;--s1:#070c18;--s2:#0c1325;--s3:#101830;
  --border:#172035;--border2:#1e2d4a;
  --green:#00e5a0;--green2:rgba(0,229,160,.1);
  --red:#ff3060;--red2:rgba(255,48,96,.1);
  --gold:#f5c842;--gold2:rgba(245,200,66,.08);
  --blue:#3b82f6;--purple:#a855f7;
  --text:#d4dff5;--text2:#5a7099;--muted:#243050;
  --btc:#f7931a;--eth:#6b8ef0;--sol:#a855f7;--xrp:#00b4e4;
  --mono:'IBM Plex Mono',monospace;--jet:'JetBrains Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:11px;display:flex;flex-direction:column;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}

/* ── HEADER ── */
#hdr{height:40px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between;
  padding:0 14px;border-bottom:1px solid var(--border);background:var(--s1);z-index:100;}
.logo{font-family:var(--jet);font-size:12px;font-weight:700;letter-spacing:.22em;
  text-transform:uppercase;color:var(--gold);display:flex;align-items:center;gap:8px;}
.logo-sep{color:var(--text2);font-weight:300;}
#hdr-r{display:flex;align-items:center;gap:14px;}
#hdr-clk{font-family:var(--jet);font-size:13px;font-weight:400;color:var(--text2);letter-spacing:.12em;}
#hdr-st{font-size:9px;font-weight:700;letter-spacing:.15em;padding:3px 10px;border-radius:2px;
  border:1px solid var(--muted);color:var(--muted);transition:all .3s;}
#hdr-st.live{color:var(--green);border-color:var(--green);box-shadow:0 0 10px rgba(0,229,160,.2);}
#hdr-st.dead{color:var(--red);border-color:var(--red);}
#hdr-snp{font-size:9px;color:var(--muted);letter-spacing:.08em;}

/* ── TABS ── */
#tabs{flex-shrink:0;display:flex;align-items:center;gap:0;border-bottom:2px solid var(--border2);
  background:var(--s1);padding:0 14px;}
.tab{font-family:var(--jet);font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
  padding:8px 16px;cursor:pointer;color:var(--text2);border-bottom:2px solid transparent;
  margin-bottom:-2px;transition:.15s;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--gold);border-bottom-color:var(--gold);}

/* ── PAGES ── */
.page{display:none;flex:1;overflow:hidden;flex-direction:column;}
.page.active{display:flex;}

/* ══════════════ LIVE PAGE ══════════════ */
/* ── ASSET STRIP ── */
#strip{flex-shrink:0;display:grid;grid-template-columns:repeat(4,1fr);
  height:70px;border-bottom:2px solid var(--border2);background:var(--s1);}
.ast{padding:7px 12px;border-right:1px solid var(--border);display:flex;flex-direction:column;justify-content:space-between;}
.ast:last-child{border-right:none;}
.ast-top{display:flex;align-items:baseline;gap:8px;}
.ast-nm{font-family:var(--jet);font-size:10px;font-weight:700;letter-spacing:.12em;}
.ast-px{font-family:var(--jet);font-size:16px;font-weight:700;letter-spacing:-.01em;}
.ast-tg{font-size:9px;font-weight:700;padding:1px 5px;border-radius:2px;letter-spacing:.06em;}
.ast-bot{display:flex;gap:12px;}
.ast-m{display:flex;flex-direction:column;}
.ast-ml{font-size:8px;color:var(--text2);letter-spacing:.08em;text-transform:uppercase;line-height:1;margin-bottom:1px;}
.ast-mv{font-size:10px;font-weight:600;line-height:1;}

/* ── GRID ── */
#grid{flex:1;min-height:0;display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(2,1fr);}
.mc{border-right:1px solid var(--border);border-bottom:1px solid var(--border);
  padding:9px 11px 7px;display:flex;flex-direction:column;gap:0;
  cursor:pointer;position:relative;overflow:hidden;transition:background .12s;}
.mc::after{content:'';position:absolute;inset:0;pointer-events:none;transition:opacity .2s;opacity:0;}
.mc:nth-child(4n){border-right:none;}
.mc:nth-child(n+5){border-bottom:none;}
.mc:hover{background:var(--s1);}
.mc.sel{outline:1px solid var(--gold);outline-offset:-1px;background:var(--s1);}
.mc.sig-up{background:rgba(0,229,160,.025);}
.mc.sig-dn{background:rgba(255,48,96,.025);}
.mc.sig-up.strong::after{background:linear-gradient(135deg,rgba(0,229,160,.07) 0%,transparent 60%);opacity:1;}
.mc.sig-dn.strong::after{background:linear-gradient(135deg,rgba(255,48,96,.07) 0%,transparent 60%);opacity:1;}
.mc.sig-up.strong{box-shadow:inset 2px 0 0 rgba(0,229,160,.4);}
.mc.sig-dn.strong{box-shadow:inset 2px 0 0 rgba(255,48,96,.4);}
.mc-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;flex-shrink:0;}
.mc-badge{display:flex;align-items:center;gap:5px;}
.mc-dot{width:5px;height:5px;border-radius:50%;background:var(--muted);flex-shrink:0;}
.mc-dot.fresh{background:var(--green);box-shadow:0 0 4px var(--green);}
.mc-dot.stale{background:var(--gold);}
.mc-dot.dead,.mc-dot.stopped{background:var(--red);}
.mc-coin{font-family:var(--jet);font-size:10px;font-weight:700;letter-spacing:.1em;}
.mc-iv{font-size:8px;font-weight:700;padding:1px 5px;border-radius:2px;
  background:var(--s3);border:1px solid var(--border2);color:var(--text2);letter-spacing:.08em;}
.mc-tmr{font-family:var(--jet);font-size:11px;font-weight:400;color:var(--text2);letter-spacing:.08em;}
.mc-tmr.urg{color:var(--red);font-weight:700;animation:pls 1s infinite;}
@keyframes pls{0%,100%{opacity:1;}50%{opacity:.45;}}
.mc-mr{display:flex;align-items:center;gap:7px;margin-bottom:6px;flex-shrink:0;}
.mc-mid{font-family:var(--jet);font-size:21px;font-weight:700;line-height:1;letter-spacing:-.02em;}
.mc-act{font-family:var(--jet);font-size:8px;font-weight:700;padding:2px 6px;border-radius:2px;
  letter-spacing:.1em;text-transform:uppercase;border:1px solid transparent;}
.mc-act.up{background:var(--green2);color:var(--green);border-color:rgba(0,229,160,.3);}
.mc-act.dn{background:var(--red2);color:var(--red);border-color:rgba(255,48,96,.3);}
.mc-act.nu{background:var(--s3);color:var(--text2);border-color:var(--border2);}
.mc-sig{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;padding:5px 6px;
  background:var(--s2);border-radius:3px;border:1px solid var(--border);margin-bottom:5px;flex-shrink:0;}
.mc-si{display:flex;flex-direction:column;align-items:center;gap:2px;}
.mc-sl{font-size:7px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;}
.mc-sv{font-family:var(--jet);font-size:10px;font-weight:700;}
.mc-bn{display:flex;justify-content:space-between;font-size:9px;color:var(--text2);margin-bottom:5px;flex-shrink:0;}
.mc-bv{font-weight:600;}
.mc-pw{display:flex;align-items:center;gap:5px;margin-bottom:5px;flex-shrink:0;}
.mc-pb{flex:1;height:2px;background:var(--border);border-radius:1px;overflow:hidden;}
.mc-pf{height:100%;border-radius:1px;background:var(--green);transition:width .8s linear,background .5s;}
.mc-pp{font-size:8px;color:var(--muted);letter-spacing:.06em;width:28px;text-align:right;}
.mc-spk{flex:1;min-height:0;position:relative;}
.mc-spk canvas{position:absolute;inset:0;width:100%!important;height:100%!important;}

/* ── DETAIL PANEL ── */
#det{flex-shrink:0;height:0;overflow:hidden;border-top:2px solid var(--border2);
  background:var(--s1);transition:height .22s cubic-bezier(.4,0,.2,1);display:flex;flex-direction:column;}
#det.open{height:290px;}
#det-hdr{flex-shrink:0;display:flex;align-items:center;justify-content:space-between;
  padding:6px 14px;border-bottom:1px solid var(--border);}
#det-ttl{font-family:var(--jet);font-size:10px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--gold);}
.det-ctrl{font-size:9px;color:var(--text2);cursor:pointer;padding:2px 8px;
  border:1px solid var(--border2);border-radius:2px;background:var(--s2);letter-spacing:.06em;transition:.12s;}
.det-ctrl:hover{color:var(--text);}
#det-body{flex:1;display:grid;grid-template-columns:185px 1fr 1fr 1fr 1fr;overflow:hidden;min-height:0;}
.dc{border-right:1px solid var(--border);padding:8px 10px;overflow:hidden;display:flex;flex-direction:column;}
.dc:last-child{border-right:none;}
.dc:first-child{overflow-y:auto;padding-right:6px;}
.dc-lbl{font-size:8px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--text2);margin-bottom:7px;flex-shrink:0;}
.dc-ch{flex:1;position:relative;min-height:0;}
.dct{width:100%;border-collapse:collapse;}
.dct tr{border-bottom:1px solid var(--border);}
.dct tr:last-child{border-bottom:none;}
.dct td{padding:4px 3px;font-size:10px;white-space:nowrap;}
.dl{color:var(--text2);}
.dv{text-align:right;font-weight:600;font-family:var(--jet);padding-left:8px;}

/* ══════════════ DATA PAGE ══════════════ */
#pg-data{padding:0;overflow:hidden;}
#data-top{flex-shrink:0;padding:12px 16px;border-bottom:1px solid var(--border);background:var(--s1);
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.data-stat{display:flex;flex-direction:column;gap:2px;padding:8px 14px;
  background:var(--s2);border:1px solid var(--border);border-radius:3px;min-width:120px;}
.data-stat-lbl{font-size:8px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text2);}
.data-stat-val{font-family:var(--jet);font-size:18px;font-weight:700;color:var(--gold);}
#data-filters{flex-shrink:0;padding:8px 16px;border-bottom:1px solid var(--border);background:var(--s2);
  display:flex;align-items:center;gap:10px;}
#data-filters label{font-size:9px;color:var(--text2);letter-spacing:.08em;text-transform:uppercase;}
.dfilter{background:var(--s3);border:1px solid var(--border2);color:var(--text);
  font-family:var(--mono);font-size:10px;padding:4px 8px;border-radius:2px;outline:none;}
.dfilter:focus{border-color:var(--gold);}
#data-load-btn{background:var(--s3);border:1px solid var(--gold);color:var(--gold);
  font-family:var(--jet);font-size:9px;font-weight:700;letter-spacing:.12em;padding:4px 14px;
  border-radius:2px;cursor:pointer;text-transform:uppercase;transition:.15s;}
#data-load-btn:hover{background:var(--gold2);}
#data-body{flex:1;overflow:hidden;display:flex;flex-direction:column;}
#data-tbl-wrap{flex:1;overflow:auto;}
#data-tbl{width:100%;border-collapse:collapse;font-size:10px;}
#data-tbl thead tr{background:var(--s2);position:sticky;top:0;z-index:10;}
#data-tbl th{padding:6px 8px;font-size:8px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text2);border-bottom:1px solid var(--border2);white-space:nowrap;text-align:right;cursor:pointer;}
#data-tbl th:first-child{text-align:left;}
#data-tbl th:hover{color:var(--gold);}
#data-tbl td{padding:5px 8px;border-bottom:1px solid var(--border);white-space:nowrap;
  font-family:var(--jet);text-align:right;}
#data-tbl td:first-child{text-align:left;font-family:var(--mono);}
#data-tbl tr:hover td{background:var(--s2);}
.cell-up{color:var(--green)!important;}
.cell-dn{color:var(--red)!important;}
.cell-nu{color:var(--text2)!important;}
.cell-ac{color:var(--gold)!important;}
#data-pager{flex-shrink:0;padding:6px 16px;border-top:1px solid var(--border);background:var(--s1);
  display:flex;align-items:center;gap:10px;font-size:9px;color:var(--text2);}
.pg-btn{background:var(--s3);border:1px solid var(--border2);color:var(--text2);
  font-family:var(--jet);font-size:9px;padding:3px 10px;border-radius:2px;cursor:pointer;transition:.12s;}
.pg-btn:hover:not(:disabled){border-color:var(--gold);color:var(--gold);}
.pg-btn:disabled{opacity:.35;cursor:default;}
#data-status{font-size:9px;color:var(--text2);margin-left:auto;}
.data-empty{text-align:center;padding:40px;color:var(--muted);font-size:12px;}

/* ══════════════ LOGS PAGE ══════════════ */
#pg-logs{padding:0;}
#logs-ctrl{flex-shrink:0;padding:8px 16px;border-bottom:1px solid var(--border);background:var(--s1);
  display:flex;align-items:center;gap:10px;}
#logs-ctrl span{font-size:9px;color:var(--text2);letter-spacing:.08em;}
.logs-btn{background:var(--s3);border:1px solid var(--border2);color:var(--text2);
  font-family:var(--jet);font-size:9px;padding:3px 10px;border-radius:2px;cursor:pointer;transition:.12s;}
.logs-btn:hover{border-color:var(--gold);color:var(--gold);}
#logs-body{flex:1;overflow:auto;padding:10px 16px;font-size:10px;line-height:1.8;background:var(--bg);}
.log-line{color:var(--text2);}
.log-line.INFO{color:var(--text);}
.log-line.ERROR,.log-line.CRITICAL{color:var(--red);}
.log-line.WARNING{color:var(--gold);}
.log-line.DEBUG{color:var(--muted);}

/* ── FOOTER ── */
#ftr{flex-shrink:0;height:22px;display:flex;align-items:center;
  border-top:1px solid var(--border);background:var(--s1);
  font-size:8px;color:var(--text2);letter-spacing:.07em;overflow:hidden;}
.fi{display:flex;align-items:center;gap:4px;padding:0 10px;
  border-right:1px solid var(--border);height:100%;white-space:nowrap;}
.fi:last-child{margin-left:auto;border-right:none;border-left:1px solid var(--border);}
.fdot{width:5px;height:5px;border-radius:50%;background:var(--muted);}
.fdot.fresh{background:var(--green);box-shadow:0 0 4px var(--green);}
.fdot.stale{background:var(--gold);}
.fdot.dead,.fdot.stopped{background:var(--red);}

/* ── UTIL ── */
.up{color:var(--green)!important;}.dn{color:var(--red)!important;}
.ac{color:var(--gold)!important;}.bl{color:var(--blue)!important;}
.nu{color:var(--text2)!important;}.dim{color:var(--muted)!important;}
</style>
</head>
<body>

<div id="hdr">
  <div class="logo">PM<span class="logo-sep">/</span>BN<span class="logo-sep" style="font-weight:300;font-size:10px;letter-spacing:.1em">WATCHER</span></div>
  <div id="hdr-r">
    <span id="hdr-snp" class="dim">snap #0</span>
    <span id="hdr-clk">--:--:--</span>
    <span id="hdr-st">● CONN…</span>
  </div>
</div>

<div id="tabs">
  <div class="tab active" onclick="showTab('live')">● Live</div>
  <div class="tab" onclick="showTab('data')">◈ Data</div>
  <div class="tab" onclick="showTab('logs')">≡ Logs</div>
</div>

<!-- ══ LIVE ══ -->
<div class="page active" id="pg-live">
  <div id="strip"></div>
  <div id="grid"></div>
  <div id="det">
    <div id="det-hdr">
      <span id="det-ttl">— SELECT A MARKET —</span>
      <span class="det-ctrl" onclick="closeDet()">✕ Close</span>
    </div>
    <div id="det-body">
      <div class="dc"><div class="dc-lbl">Detailed metrics</div><table class="dct" id="dc-mtbl"></table></div>
      <div class="dc"><div class="dc-lbl">UP Mid &amp; Φ(z)</div><div class="dc-ch"><canvas id="dc-mid"></canvas></div></div>
      <div class="dc"><div class="dc-lbl">Z-Score</div><div class="dc-ch"><canvas id="dc-z"></canvas></div></div>
      <div class="dc"><div class="dc-lbl">Price vs Target</div><div class="dc-ch"><canvas id="dc-price"></canvas></div></div>
      <div class="dc"><div class="dc-lbl">Flow Imbalance 60s</div><div class="dc-ch"><canvas id="dc-flow"></canvas></div></div>
    </div>
  </div>
</div>

<!-- ══ DATA ══ -->
<div class="page" id="pg-data">
  <div id="data-top">
    <div class="data-stat"><span class="data-stat-lbl">Total rows</span><span class="data-stat-val" id="ds-rows">—</span></div>
    <div class="data-stat"><span class="data-stat-lbl">Days collected</span><span class="data-stat-val" id="ds-days">—</span></div>
    <div class="data-stat"><span class="data-stat-lbl">Active markets</span><span class="data-stat-val" id="ds-mkts">—</span></div>
    <div class="data-stat"><span class="data-stat-lbl">Date range</span><span class="data-stat-val" id="ds-range" style="font-size:11px">—</span></div>
  </div>
  <div id="data-filters">
    <label>Market</label>
    <select class="dfilter" id="df-market">
      <option value="all">All markets</option>
      <option value="btc-5m">BTC 5m</option>
      <option value="btc-15m">BTC 15m</option>
      <option value="eth-5m">ETH 5m</option>
      <option value="eth-15m">ETH 15m</option>
      <option value="sol-5m">SOL 5m</option>
      <option value="sol-15m">SOL 15m</option>
      <option value="xrp-5m">XRP 5m</option>
      <option value="xrp-15m">XRP 15m</option>
    </select>
    <label>Date</label>
    <input class="dfilter" type="date" id="df-date" style="width:130px"/>
    <label>Rows/page</label>
    <select class="dfilter" id="df-limit">
      <option value="50">50</option>
      <option value="100">100</option>
      <option value="200">200</option>
      <option value="500">500</option>
    </select>
    <button id="data-load-btn" onclick="loadData(0)">Load</button>
    <span id="data-status"></span>
  </div>
  <div id="data-body">
    <div id="data-tbl-wrap"><div class="data-empty">Select filters and click Load to browse collected data.</div></div>
    <div id="data-pager" style="display:none">
      <button class="pg-btn" id="pg-prev" onclick="pageStep(-1)">← Prev</button>
      <button class="pg-btn" id="pg-next" onclick="pageStep(1)">Next →</button>
      <span id="pg-info"></span>
    </div>
  </div>
</div>

<!-- ══ LOGS ══ -->
<div class="page" id="pg-logs">
  <div id="logs-ctrl">
    <span>Last 200 log lines</span>
    <button class="logs-btn" onclick="loadLogs()">↻ Refresh</button>
    <button class="logs-btn" onclick="document.getElementById('logs-body').scrollTop=1e9">↓ Bottom</button>
  </div>
  <div id="logs-body"></div>
</div>

<div id="ftr"></div>

<script>
'use strict';
// ── Config ──
const COINS=['btc','eth','sol','xrp'];
const IVS=[5,15];
const CC={btc:'#f7931a',eth:'#6b8ef0',sol:'#a855f7',xrp:'#00b4e4'};
const MAX=180;
const MH={};
const SPARKS={};
const DC={};
let selK=null,lastS=null,snapN=0;
let datPage=0,datTotal=0,datLimit=50;

// ── Tab switch ──
function showTab(t){
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));
  const tabEl=document.querySelector(`.tab[onclick="showTab('${t}')"]`);
  if(tabEl)tabEl.classList.add('active');
  const pgEl=document.getElementById('pg-'+t);
  if(pgEl)pgEl.classList.add('active');
  if(t==='data')loadStats();
  if(t==='logs')loadLogs();
}

// ── Formatters ──
const f1=v=>v==null?'—':(+v).toFixed(1);
const f2=v=>v==null?'—':(+v).toFixed(2);
const f3=v=>v==null?'—':(+v).toFixed(3);
const fp=v=>v==null?'—':((+v)*100).toFixed(1)+'%';
const fbps=v=>v==null?'—':(v>=0?'+':'')+( +v).toFixed(1)+' bp';
const fms=v=>v==null?'—':(+v).toFixed(0)+'ms';
const fmm=v=>{if(v==null)return'—';const s=Math.round(Math.max(0,+v));return s>=60?`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`:`${s}s`;};
const fpx=(v,d)=>v==null?'—':(+v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const PXD={btc:2,eth:2,sol:3,xrp:4};
const clr=v=>v==null?'':+v>0?'up':+v<0?'dn':'';
const push=(a,v)=>{a.push(v??null);if(a.length>MAX)a.shift();};
const mk=(c,iv)=>`${c}-${iv}m`;
function initMH(k){if(!MH[k])MH[k]={ts:[],mid:[],prob:[],z:[],price:[],target:[],fl:[]};}

// ══ LIVE UI ══
function initGrid(){
  const g=document.getElementById('grid');g.innerHTML='';
  for(const iv of IVS)for(const c of COINS){
    const k=mk(c,iv);initMH(k);
    const d=document.createElement('div');
    d.className='mc';d.id=`mc-${k}`;d.onclick=()=>selMkt(k);
    d.innerHTML=`
<div class="mc-hd">
  <div class="mc-badge">
    <div class="mc-dot" id="dot-${k}"></div>
    <span class="mc-coin" style="color:${CC[c]}">${c.toUpperCase()}</span>
    <span class="mc-iv">${iv}m</span>
  </div>
  <span class="mc-tmr" id="tmr-${k}">—</span>
</div>
<div class="mc-mr">
  <span class="mc-mid nu" id="mid-${k}">—.—%</span>
  <span class="mc-act nu" id="act-${k}">◈ WAIT</span>
</div>
<div class="mc-sig">
  <div class="mc-si"><span class="mc-sl">Edge</span><span class="mc-sv" id="edge-${k}">—</span></div>
  <div class="mc-si"><span class="mc-sl">Z</span><span class="mc-sv" id="z-${k}">—</span></div>
  <div class="mc-si"><span class="mc-sl">Φ(z)</span><span class="mc-sv" id="prob-${k}">—</span></div>
  <div class="mc-si"><span class="mc-sl">Sprd</span><span class="mc-sv nu" id="sprd-${k}">—</span></div>
</div>
<div class="mc-bn">
  <span>Ret60s <span class="mc-bv" id="r60-${k}">—</span></span>
  <span>Flow <span class="mc-bv" id="fl-${k}">—</span></span>
  <span>Vol <span class="mc-bv nu" id="vol-${k}">—</span></span>
</div>
<div class="mc-pw">
  <div class="mc-pb"><div class="mc-pf" id="prg-${k}" style="width:0%"></div></div>
  <span class="mc-pp dim" id="prgp-${k}">0%</span>
</div>
<div class="mc-spk"><canvas id="spk-${k}"></canvas></div>`;
    g.appendChild(d);
  }
}

function initStrip(){
  document.getElementById('strip').innerHTML=COINS.map(c=>`
<div class="ast">
  <div class="ast-top">
    <span class="ast-nm" style="color:${CC[c]}">${c.toUpperCase()}</span>
    <span class="ast-px" id="apx-${c}" style="color:${CC[c]}">—</span>
    <span class="ast-tg" id="atg-${c}">—</span>
  </div>
  <div class="ast-bot">
    <div class="ast-m"><span class="ast-ml">Ret 10s</span><span class="ast-mv" id="ar10-${c}">—</span></div>
    <div class="ast-m"><span class="ast-ml">Ret 60s</span><span class="ast-mv" id="ar60-${c}">—</span></div>
    <div class="ast-m"><span class="ast-ml">Flow 60s</span><span class="ast-mv" id="afi-${c}">—</span></div>
    <div class="ast-m"><span class="ast-ml">Vol 60s</span><span class="ast-mv nu" id="avo-${c}">—</span></div>
    <div class="ast-m"><span class="ast-ml">Lag</span><span class="ast-mv nu" id="alg-${c}">—</span></div>
  </div>
</div>`).join('');
}

Chart.defaults.color='#5a7099';
Chart.defaults.borderColor='#172035';
Chart.defaults.font.family="'IBM Plex Mono',monospace";
Chart.defaults.font.size=8;

function hex2rgba(hex,a){
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}
function initSpark(k,c){
  const canvas=document.getElementById(`spk-${k}`);if(!canvas)return;
  const col=CC[c];
  SPARKS[k]=new Chart(canvas,{
    type:'line',
    data:{labels:MH[k].ts,datasets:[{data:MH[k].mid,borderColor:col,borderWidth:1.5,pointRadius:0,tension:.35,
      fill:true,backgroundColor:hex2rgba(col,.07)}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{x:{display:false},y:{display:true,position:'right',min:0,max:1,
        grid:{color:'rgba(23,32,53,.8)'},
        ticks:{callback:v=>(v*100).toFixed(0)+'%',maxTicksLimit:3,padding:2,font:{size:7}},
        border:{display:false}}}}});
}
function initAllSparks(){for(const iv of IVS)for(const c of COINS)initSpark(mk(c,iv),c);}

function mkChart(id,datasets,sy){
  const el=document.getElementById(id);if(!el)return null;
  return new Chart(el,{type:'line',data:{labels:[],datasets},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:datasets.length>1,labels:{boxWidth:6,padding:8,color:'#5a7099',font:{size:8}}},tooltip:{enabled:true}},
      scales:{x:{grid:{color:'#172035'},ticks:{maxTicksLimit:6,maxRotation:0,font:{size:7}}},
        y:{grid:{color:'#172035'},ticks:{maxTicksLimit:4,font:{size:8}},...sy}}}});
}
function initDC(){
  const yP={min:0,max:1,ticks:{callback:v=>(v*100).toFixed(0)+'%',maxTicksLimit:4,font:{size:8}}};
  const yD={ticks:{maxTicksLimit:4,font:{size:8}}};
  const y11={min:-1,max:1,ticks:{callback:v=>(v*100).toFixed(0)+'%',maxTicksLimit:4,font:{size:8}}};
  DC.mid=mkChart('dc-mid',[
    {label:'up_mid',data:[],borderColor:'#f5c842',borderWidth:1.5,pointRadius:0,tension:.3},
    {label:'Φ(z)',data:[],borderColor:'#3b82f6',borderWidth:1.2,borderDash:[4,3],pointRadius:0,tension:.3}],yP);
  DC.z=mkChart('dc-z',[
    {label:'z',data:[],borderColor:'#f5c842',borderWidth:1.5,pointRadius:0,tension:.2},
    {label:'+1.5',data:[],borderColor:'rgba(0,229,160,.45)',borderWidth:1,borderDash:[4,3],pointRadius:0},
    {label:'-1.5',data:[],borderColor:'rgba(255,48,96,.45)',borderWidth:1,borderDash:[4,3],pointRadius:0},
    {label:'+3',data:[],borderColor:'rgba(0,229,160,.2)',borderWidth:1,borderDash:[2,5],pointRadius:0},
    {label:'-3',data:[],borderColor:'rgba(255,48,96,.2)',borderWidth:1,borderDash:[2,5],pointRadius:0}],yD);
  DC.price=mkChart('dc-price',[
    {label:'Price',data:[],borderColor:'#f5c842',borderWidth:1.5,pointRadius:0,tension:.1},
    {label:'Target',data:[],borderColor:'#ff3060',borderWidth:1,borderDash:[5,4],pointRadius:0}],yD);
  DC.flow=mkChart('dc-flow',[
    {label:'Flow 60s',data:[],borderColor:'#3b82f6',borderWidth:1.5,pointRadius:0,tension:.2,
     fill:{target:'origin',above:'rgba(59,130,246,.1)',below:'rgba(255,48,96,.1)'}}],y11);
}

function selMkt(k){
  if(selK===k){closeDet();return;}
  document.querySelectorAll('.mc').forEach(e=>e.classList.remove('sel'));
  document.getElementById(`mc-${k}`)?.classList.add('sel');
  selK=k;
  document.getElementById('det').classList.add('open');
  const[c,ivs]=k.split('-');
  document.getElementById('det-ttl').textContent=`${c.toUpperCase()} · ${ivs.toUpperCase()} — MARKET DETAIL`;
  const col=CC[c]||'#f5c842';
  if(DC.mid)DC.mid.data.datasets[0].borderColor=col;
  if(DC.z)DC.z.data.datasets[0].borderColor=col;
  if(DC.price)DC.price.data.datasets[0].borderColor=col;
  syncDC(k);
  if(lastS)updateDCMetrics(k,lastS);
}
function closeDet(){
  selK=null;
  document.querySelectorAll('.mc').forEach(e=>e.classList.remove('sel'));
  document.getElementById('det').classList.remove('open');
}
function syncDC(k){
  const h=MH[k];if(!h)return;const n=h.ts.length;
  if(DC.mid){DC.mid.data.labels=[...h.ts];DC.mid.data.datasets[0].data=[...h.mid];DC.mid.data.datasets[1].data=[...h.prob];DC.mid.update('none');}
  if(DC.z){DC.z.data.labels=[...h.ts];DC.z.data.datasets[0].data=[...h.z];DC.z.data.datasets[1].data=Array(n).fill(1.5);DC.z.data.datasets[2].data=Array(n).fill(-1.5);DC.z.data.datasets[3].data=Array(n).fill(3);DC.z.data.datasets[4].data=Array(n).fill(-3);DC.z.update('none');}
  if(DC.price){DC.price.data.labels=[...h.ts];DC.price.data.datasets[0].data=[...h.price];DC.price.data.datasets[1].data=[...h.target];DC.price.update('none');}
  if(DC.flow){DC.flow.data.labels=[...h.ts];DC.flow.data.datasets[0].data=[...h.fl];DC.flow.update('none');}
}
function updateDCMetrics(k,s){
  const m=(s._markets||{})[k]||{};
  const[c]=k.split('-');
  const mid=m.up_mid,prob=m.analytical_prob_up,z=m.z_score;
  const edge=mid!=null&&prob!=null?Math.round((prob-mid)*10000):null;
  const tbl=document.getElementById('dc-mtbl');if(!tbl)return;
  const rows=[
    ['UP mid',fp(mid),mid!=null?(+mid>=.5?'up':'dn'):''],
    ['Bid / Ask',m.up_bid!=null?`${fp(m.up_bid)} / ${fp(m.up_ask)}`:'—',''],
    ['Spread',m.up_spread_bps!=null?f1(m.up_spread_bps)+' bp':'—',''],
    ['Edge',edge!=null?(edge>0?'+':'')+edge+' bp':'—',edge!=null?(Math.abs(edge)>300?(edge>0?'up':'dn'):'ac'):''],
    ['Z-score',z!=null?f3(z):'—',z!=null?(z>1.5?'up':z<-1.5?'dn':''):''],
    ['Φ(z)',fp(prob),prob!=null?(+prob>.5?'up':'dn'):''],
    ['Distance',m.distance_bps!=null?fbps(m.distance_bps):'—',clr(m.distance_bps)],
    ['Vol 60s',m.coin_vol_60s_bps!=null?f2(m.coin_vol_60s_bps)+' bp':'—',''],
    ['Mid velocity',m.poly_mid_velocity!=null?f2(m.poly_mid_velocity)+' b/s':'—',clr(m.poly_mid_velocity)],
    ['Trades 10s',m.poly_trade_count_10s??'—',''],
    ['Volume 10s',m.poly_trade_size_10s!=null?'$'+f1(m.poly_trade_size_10s):'—',''],
    ['Time left',fmm(m.seconds_remaining),m.seconds_remaining!=null&&+m.seconds_remaining<30?'dn':'ac'],
    ['Price',m.coin_price!=null?fpx(m.coin_price,PXD[c]):'—','ac'],
    ['Target',m.target_price!=null?fpx(m.target_price,PXD[c]):'—',''],
    ['Flow 60s',s[`${c}_flow_imbalance_60s`]!=null?(s[`${c}_flow_imbalance_60s`]>0?'+':'')+(s[`${c}_flow_imbalance_60s`]*100).toFixed(0)+'%':'—',clr(s[`${c}_flow_imbalance_60s`])],
    ['Buy ratio',s[`${c}_buy_ratio_usdt_60s`]!=null?(s[`${c}_buy_ratio_usdt_60s`]*100).toFixed(0)+'%':'—',s[`${c}_buy_ratio_usdt_60s`]!=null?(+s[`${c}_buy_ratio_usdt_60s`]>.55?'up':+s[`${c}_buy_ratio_usdt_60s`]<.45?'dn':''):''],
  ];
  tbl.innerHTML=rows.map(([l,v,cl])=>`<tr><td class="dl">${l}</td><td class="dv ${cl}">${v}</td></tr>`).join('');
}

function sv(id,txt,cls){
  const e=document.getElementById(id);if(!e)return;
  e.textContent=txt;
  if(cls!==undefined)e.className=e.className.replace(/ *(up|dn|ac|bl|nu|dim)\b/g,'')+(cls?' '+cls:'');
}

function handleSnap(s){
  lastS=s;snapN++;
  const ts=s.timestamp_utc?s.timestamp_utc.slice(11,19):'--:--:--';
  document.getElementById('hdr-clk').textContent=ts;
  document.getElementById('hdr-snp').textContent=`snap #${snapN}`;
  const mkts=s._markets||{};

  for(const c of COINS){
    const px=s[`${c}_price`],r10=s[`${c}_return_10s_bps`],r60=s[`${c}_return_60s_bps`];
    const fi=s[`${c}_flow_imbalance_60s`],vol=s[`${c}_vol_60s_bps`],lag=s._lags?.[c];
    const pxEl=document.getElementById(`apx-${c}`);
    if(pxEl)pxEl.textContent=px!=null?fpx(px,PXD[c]):'—';
    const tgEl=document.getElementById(`atg-${c}`);
    if(tgEl){tgEl.textContent=r60!=null?(r60>0?'+':'')+f1(r60)+'bp':'—';tgEl.className='ast-tg '+clr(r60);}
    sv(`ar10-${c}`,r10!=null?fbps(r10):'—',clr(r10));
    sv(`ar60-${c}`,r60!=null?fbps(r60):'—',clr(r60));
    const fiEl=document.getElementById(`afi-${c}`);
    if(fiEl){fiEl.textContent=fi!=null?(fi>0?'+':'')+(fi*100).toFixed(0)+'%':'—';fiEl.className='ast-mv '+clr(fi);}
    sv(`avo-${c}`,vol!=null?f2(vol)+' bp':'—','nu');
    sv(`alg-${c}`,fms(lag),'nu');
  }

  for(const iv of IVS)for(const c of COINS){
    const k=mk(c,iv),m=mkts[k]||{},h=MH[k];
    push(h.ts,ts);push(h.mid,m.up_mid??null);push(h.prob,m.analytical_prob_up??null);
    push(h.z,m.z_score??null);push(h.price,m.coin_price??null);push(h.target,m.target_price??null);
    push(h.fl,s[`${c}_flow_imbalance_60s`]??null);
    const mid=m.up_mid,prob=m.analytical_prob_up,z=m.z_score;
    const edge=mid!=null&&prob!=null?Math.round((prob-mid)*10000):null;
    const card=document.getElementById(`mc-${k}`);
    if(card){let cl='mc';if(selK===k)cl+=' sel';if(edge!=null&&edge>0)cl+=' sig-up';if(edge!=null&&edge<0)cl+=' sig-dn';if(edge!=null&&Math.abs(edge)>300)cl+=' strong';card.className=cl;}
    const dot=document.getElementById(`dot-${k}`);
    if(dot)dot.className='mc-dot '+(s._asset_health?.[c]||'dead');
    const mEl=document.getElementById(`mid-${k}`);
    if(mEl){mEl.textContent=mid!=null?fp(mid):'—.—%';mEl.className='mc-mid '+(mid!=null?(mid>=.5?'up':'dn'):'nu');}
    const aEl=document.getElementById(`act-${k}`);
    if(aEl){
      if(edge!=null&&edge>300&&z!=null&&z>1.5){aEl.textContent='▲ BUY UP';aEl.className='mc-act up';}
      else if(edge!=null&&edge<-300&&z!=null&&z<-1.5){aEl.textContent='▼ BUY DN';aEl.className='mc-act dn';}
      else{aEl.textContent='◈ WAIT';aEl.className='mc-act nu';}
    }
    sv(`edge-${k}`,edge!=null?(edge>0?'+':'')+edge+' bp':'—',edge!=null?(Math.abs(edge)>300?(edge>0?'up':'dn'):'ac'):'');
    sv(`z-${k}`,z!=null?f3(z):'—',z!=null?(z>1.5?'up':z<-1.5?'dn':''):'');
    sv(`prob-${k}`,fp(prob),prob!=null?(+prob>.6?'up':+prob<.4?'dn':''):'');
    sv(`sprd-${k}`,m.up_spread_bps!=null?f1(m.up_spread_bps)+' bp':'—','nu');
    const r60=s[`${c}_return_60s_bps`],fi=s[`${c}_flow_imbalance_60s`],vol=s[`${c}_vol_60s_bps`];
    sv(`r60-${k}`,r60!=null?(r60>0?'+':'')+f1(r60)+' bp':'—',clr(r60));
    sv(`fl-${k}`,fi!=null?(fi>0?'+':'')+(fi*100).toFixed(0)+'%':'—',clr(fi));
    sv(`vol-${k}`,vol!=null?f2(vol)+' bp':'—','nu');
    const rem=m.seconds_remaining;
    const tmEl=document.getElementById(`tmr-${k}`);
    if(tmEl){tmEl.textContent=fmm(rem);tmEl.className='mc-tmr'+(rem!=null&&+rem<30?' urg':'');}
    const pct=m.window_progress||0;
    const pEl=document.getElementById(`prg-${k}`);
    if(pEl){pEl.style.width=pct+'%';pEl.style.background=pct>80?'var(--red)':pct>55?'var(--gold)':'var(--green)';}
    sv(`prgp-${k}`,Math.round(pct)+'%','dim');
    const spk=SPARKS[k];
    if(spk){spk.data.labels=[...h.ts];spk.data.datasets[0].data=[...h.mid];spk.update('none');}
  }

  if(selK){syncDC(selK);updateDCMetrics(selK,s);}

  const ph=s._poly_health||'dead';
  let fhtml=`<div class="fi"><span class="fdot ${ph}"></span>POLY: ${ph.toUpperCase()}</div>`;
  for(const c of COINS){
    const h=s._asset_health?.[c]||'dead',lag=s._lags?.[c];
    fhtml+=`<div class="fi"><span class="fdot ${h}"></span>${c.toUpperCase()} ${lag!=null?fms(lag):'—'}</div>`;
  }
  const ag60=s.cross_asset_agreement_60s;
  if(ag60!=null)fhtml+=`<div class="fi">AGREE: <span class="${+ag60>.66?'up':+ag60<.33?'dn':'ac'}">${(+ag60*100).toFixed(0)}%</span></div>`;
  const pm=s._storage;
  if(pm)fhtml+=`<div class="fi">STORAGE: ${pm.total_mb??'—'} MB</div>`;
  fhtml+=`<div class="fi" style="margin-left:auto">${snapN} snaps · ${s.timestamp_utc?s.timestamp_utc.slice(0,10):'—'}</div>`;
  document.getElementById('ftr').innerHTML=fhtml;
}

// ══ DATA UI ══
async function loadStats(){
  try{
    const r=await fetch('/api/stats');
    const d=await r.json();
    document.getElementById('ds-rows').textContent=(d.total_rows||0).toLocaleString();
    document.getElementById('ds-days').textContent=(d.dates||[]).length;
    document.getElementById('ds-mkts').textContent=Object.keys(d.markets||{}).length;
    const dr=d.date_range||[];
    document.getElementById('ds-range').textContent=dr.length===2?`${dr[0]} → ${dr[1]}`:'—';
  }catch(e){console.error(e);}
}

async function loadData(page){
  datPage=page;
  datLimit=parseInt(document.getElementById('df-limit').value)||50;
  const market=document.getElementById('df-market').value;
  const date=document.getElementById('df-date').value;
  const offset=page*datLimit;
  document.getElementById('data-status').textContent='Loading…';
  try{
    const qs=new URLSearchParams({market,date,limit:datLimit,offset});
    const r=await fetch('/api/data?'+qs);
    const d=await r.json();
    datTotal=d.total||0;
    renderTable(d.rows||[],d.columns||[]);
    document.getElementById('data-status').textContent=`${datTotal.toLocaleString()} rows total`;
    const pager=document.getElementById('data-pager');
    if(datTotal>0){
      pager.style.display='flex';
      document.getElementById('pg-prev').disabled=page===0;
      document.getElementById('pg-next').disabled=offset+datLimit>=datTotal;
      document.getElementById('pg-info').textContent=`Page ${page+1} / ${Math.ceil(datTotal/datLimit)} (${offset+1}–${Math.min(offset+datLimit,datTotal)})`;
    }else{
      pager.style.display='none';
    }
  }catch(e){
    document.getElementById('data-status').textContent='Error: '+e.message;
  }
}

function pageStep(dir){loadData(datPage+dir);}

const COL_FMT={
  seconds_remaining:v=>v!=null?Math.round(+v)+'s':'—',
  window_progress:v=>v!=null?(+v).toFixed(1)+'%':'—',
  coin_price:v=>v!=null?(+v).toFixed(4):'—',
  target_price:v=>v!=null?(+v).toFixed(4):'—',
  distance_bps:v=>v!=null?((+v)>0?'+':'')+((+v).toFixed(1))+' bp':'—',
  z_score:v=>v!=null?(+v).toFixed(3):'—',
  analytical_prob_up:v=>v!=null?((+v)*100).toFixed(1)+'%':'—',
  edge_bps:v=>v!=null?((+v)>0?'+':'')+((+v).toFixed(0))+' bp':'—',
  up_mid:v=>v!=null?((+v)*100).toFixed(1)+'%':'—',
  up_bid:v=>v!=null?((+v)*100).toFixed(1)+'%':'—',
  up_ask:v=>v!=null?((+v)*100).toFixed(1)+'%':'—',
  up_spread_bps:v=>v!=null?(+v).toFixed(1)+' bp':'—',
  up_depth_imbalance:v=>v!=null?((+v)>0?'+':'')+((+v*100).toFixed(1))+'%':'—',
  coin_return_60s_bps:v=>v!=null?((+v)>0?'+':'')+((+v).toFixed(1))+' bp':'—',
  coin_flow_imbalance_60s:v=>v!=null?((+v)>0?'+':'')+((+v*100).toFixed(0))+'%':'—',
  poly_trade_count_60s:v=>v??'—',
  poly_mid_velocity:v=>v!=null?(+v).toFixed(3):'—',
  outcome:v=>v===''||v==null?'—':v==='1'?'UP':'DN',
};
const COL_CLR={
  distance_bps:v=>+v>0?'cell-up':+v<0?'cell-dn':'',
  z_score:v=>+v>1.5?'cell-up':+v<-1.5?'cell-dn':'',
  edge_bps:v=>Math.abs(+v)>300?(+v>0?'cell-up':'cell-dn'):'cell-ac',
  analytical_prob_up:v=>+v>.55?'cell-up':+v<.45?'cell-dn':'',
  up_mid:v=>+v>.55?'cell-up':+v<.45?'cell-dn':'',
  coin_return_60s_bps:v=>+v>0?'cell-up':+v<0?'cell-dn':'',
  coin_flow_imbalance_60s:v=>+v>0?'cell-up':+v<0?'cell-dn':'',
  outcome:v=>v==='1'?'cell-up':v==='0'?'cell-dn':'cell-nu',
};
const COL_LABELS={
  timestamp_utc:'Time',coin:'Coin',interval_min:'Int',slug:'Slug',
  seconds_remaining:'Rem',window_progress:'Prog',coin_price:'Price',target_price:'Target',
  distance_bps:'Dist',z_score:'Z',analytical_prob_up:'Φ(z)',edge_bps:'Edge',
  up_mid:'Mid',up_bid:'Bid',up_ask:'Ask',up_spread_bps:'Spread',up_depth_imbalance:'Imbal',
  coin_return_60s_bps:'Ret60',coin_flow_imbalance_60s:'Flow60',
  poly_trade_count_60s:'Trades60',poly_mid_velocity:'Vel',outcome:'Out',
};

function renderTable(rows,cols){
  const wrap=document.getElementById('data-tbl-wrap');
  if(!rows.length){wrap.innerHTML='<div class="data-empty">No data found for these filters.</div>';return;}
  const shortCols=cols.filter(c=>c!=='slug'&&c!=='window_start'&&c!=='window_end');
  let html=`<table id="data-tbl"><thead><tr>`;
  for(const c of shortCols)html+=`<th>${COL_LABELS[c]||c}</th>`;
  html+=`</tr></thead><tbody>`;
  for(const row of rows){
    html+='<tr>';
    for(const c of shortCols){
      const v=row[c];
      const fmt=COL_FMT[c];
      const txt=fmt?fmt(v):v??'—';
      const clsFn=COL_CLR[c];
      const cls=clsFn&&v!=null?clsFn(v):'';
      html+=`<td class="${cls}">${txt}</td>`;
    }
    html+='</tr>';
  }
  html+='</tbody></table>';
  wrap.innerHTML=html;
}

// ══ LOGS UI ══
async function loadLogs(){
  try{
    const r=await fetch('/api/logs?n=200');
    const d=await r.json();
    const body=document.getElementById('logs-body');
    body.innerHTML=d.lines.map(l=>{
      let cls='log-line';
      if(l.includes('[ERROR]')||l.includes('[CRITICAL]'))cls+=' ERROR';
      else if(l.includes('[WARNING]'))cls+=' WARNING';
      else if(l.includes('[INFO]'))cls+=' INFO';
      else if(l.includes('[DEBUG]'))cls+=' DEBUG';
      return `<div class="${cls}">${l.replace(/</g,'&lt;')}</div>`;
    }).join('');
    body.scrollTop=body.scrollHeight;
  }catch(e){document.getElementById('logs-body').textContent='Error: '+e;}
}

// ── WebSocket ──
let ws,rTimer;
function connect(){
  const sec=location.protocol==='https:';
  const url=sec?`wss://${location.hostname}/ws`:`ws://${location.hostname||'localhost'}:WS_PORT_PLACEHOLDER`;
  ws=new WebSocket(url);
  ws.onopen=()=>{const e=document.getElementById('hdr-st');e.textContent='● LIVE';e.className='live';};
  ws.onmessage=e=>{try{handleSnap(JSON.parse(e.data));}catch(err){console.error(err);}};
  ws.onclose=()=>{const e=document.getElementById('hdr-st');e.textContent='● OFF';e.className='dead';clearTimeout(rTimer);rTimer=setTimeout(connect,2500);};
  ws.onerror=()=>{};
}

// ── Boot ──
initGrid();initStrip();initAllSparks();initDC();connect();
</script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self) -> None:
        self._clients: set = set()
        self._lock    = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=10)
        self._enabled = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import websockets  # noqa
            return True
        except ImportError:
            logger.warning("websockets missing — WS dashboard disabled. pip install websockets")
            return False

    def start(self) -> None:
        if not self._enabled:
            return
        threading.Thread(target=self._run_ws,   daemon=True, name="dash-ws").start()
        threading.Thread(target=self._run_http, daemon=True, name="dash-http").start()
        logger.info("Dashboard → http://%s:%d  |  ws://%s:%d",
                    DASHBOARD_HOST, DASHBOARD_HTTP_PORT,
                    DASHBOARD_HOST, DASHBOARD_PORT)

    def push_snapshot(self, snap: dict[str, Any]) -> None:
        if not self._enabled:
            return
        payload = json.dumps(snap, default=str)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except queue.Empty:
                pass

    # ── WebSocket server ──────────────────────────────────────────────────────

    def _run_ws(self) -> None:
        import asyncio
        asyncio.run(self._ws_main())

    async def _ws_main(self) -> None:
        import asyncio
        import websockets  # type: ignore

        async def handler(ws: Any) -> None:
            with self._lock: self._clients.add(ws)
            try:    await ws.wait_closed()
            finally:
                with self._lock: self._clients.discard(ws)

        async def broadcaster() -> None:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    payload = await loop.run_in_executor(
                        None, lambda: self._queue.get(timeout=0.5))
                except Exception:
                    continue
                with self._lock:
                    targets = list(self._clients)
                dead: set = set()
                for client in targets:
                    try:    await client.send(payload)
                    except: dead.add(client)
                if dead:
                    with self._lock: self._clients -= dead

        async with websockets.serve(handler, DASHBOARD_HOST, DASHBOARD_PORT):
            await broadcaster()

    # ── HTTP server ───────────────────────────────────────────────────────────

    def _run_http(self) -> None:
        html = _HTML.replace("WS_PORT_PLACEHOLDER", str(DASHBOARD_PORT)).encode("utf-8")

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                path   = parsed.path
                qs     = urllib.parse.parse_qs(parsed.query)

                def param(k, default=""):
                    return qs.get(k, [default])[0]

                if path == "/" or path == "/index.html":
                    self._serve_bytes(html, "text/html; charset=utf-8")

                elif path == "/api/stats":
                    data = _data_stats()
                    self._serve_json(data)

                elif path == "/api/data":
                    market = param("market", "all")
                    date   = param("date", "")
                    limit  = min(int(param("limit", "100")), 1000)
                    offset = int(param("offset", "0"))
                    data   = _data_query(market, date, limit, offset)
                    self._serve_json(data)

                elif path == "/api/logs":
                    n    = min(int(param("n", "100")), 500)
                    data = {"lines": _log_tail(n)}
                    self._serve_json(data)

                elif path.endswith(".html") and path != "/index.html":
                    # Servir les fichiers HTML statiques depuis le dossier pm_watcher
                    import os
                    base_dir = Path(__file__).parent
                    fname = path.lstrip("/")
                    fpath = base_dir / fname
                    if fpath.exists() and fpath.suffix == ".html":
                        content_bytes = fpath.read_bytes()
                        self._serve_bytes(content_bytes, "text/html; charset=utf-8")
                    else:
                        self.send_response(404)
                        self.end_headers()

                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_bytes(self, content: bytes, ct: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)

            def _serve_json(self, data: Any) -> None:
                payload = json.dumps(data, default=str).encode("utf-8")
                self._serve_bytes(payload, "application/json; charset=utf-8")

            def log_message(self, *a: Any) -> None:
                pass  # suppress HTTP access log

        http.server.HTTPServer((DASHBOARD_HOST, DASHBOARD_HTTP_PORT), _H).serve_forever()
