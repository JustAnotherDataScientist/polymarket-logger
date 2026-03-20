"""
validate.py — Validation locale complète du code v6 sans parquet.

Génère des DataFrames synthétiques au format v6 (101 colonnes, JSON book),
puis exécute toute la logique de run_tests.py et inspect_live.py dessus.

Usage: python3 validate.py
"""
from __future__ import annotations
import sys, json, traceback
from pathlib import Path

# Chemins dynamiques — fonctionnent partout (VM, sandbox, local)
_HERE     = Path(__file__).parent           # pm_watcher/
_BACKTEST = (_HERE / ".." / "polymarket-backtest").resolve()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BACKTEST))

import numpy as np
import pandas as pd

# ─── couleurs ────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'
PASS='✅'; FAIL='❌'; WARN='⚠️'

results = []

def ok(msg):
    print(f"  {PASS} {msg}")
    results.append(True)

def fail(msg):
    print(f"  {FAIL} {msg}")
    results.append(False)

def warn(msg):
    print(f"  {WARN} {msg}")

def section(title):
    print(f"\n{BOLD}{'─'*55}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{BOLD}{'─'*55}{NC}")

# ═════════════════════════════════════════════════════════════
# PARTIE 1 : Générer les données synthétiques v6
# ═════════════════════════════════════════════════════════════
section("1/6 — Génération des données synthétiques v6")

from csv_schema import EXPECTED_HEADER, N_COLS
print(f"  Schéma chargé : {N_COLS} colonnes")

# Paramètres de simulation
COINS     = ['btc', 'eth', 'sol', 'xrp']
INTERVALS = [5, 15]
N_WINDOWS = 6         # fenêtres par coin×interval
TICKS_PER_WINDOW = 60 # ~15s de données à 4Hz

def make_book_json(mid, side='ask', n_levels=8):
    """Génère un JSON book réaliste autour du mid."""
    levels = []
    spread = 0.003
    for i in range(n_levels):
        if side == 'ask':
            price = round(mid + spread * (1 + i * 0.5), 6)
        else:
            price = round(mid - spread * (1 + i * 0.5), 6)
        price = max(0.001, min(0.999, price))
        size  = round(np.random.uniform(50, 500), 2)
        levels.append([price, size])
    return json.dumps(levels)

def make_slug(coin, interval, window_idx):
    base_ts = 1700000000 + window_idx * interval * 60
    return f"will-{coin}-reach-target-in-{interval}min-window-{base_ts}"

rows = []
np.random.seed(42)

for coin in COINS:
    for interval in INTERVALS:
        for w in range(N_WINDOWS):
            # Paramètres de la fenêtre
            outcome_val = str(np.random.randint(0, 2))
            slug        = make_slug(coin, interval, w)
            base_mid    = np.random.uniform(0.35, 0.65)
            target_px   = np.random.uniform(20000, 70000) if coin == 'btc' else np.random.uniform(1000, 4000)
            coin_price  = target_px * np.random.uniform(0.99, 1.01)
            window_secs = interval * 60

            for t in range(TICKS_PER_WINDOW):
                # Drift du mid
                mid = base_mid + np.random.normal(0, 0.002) * (t / TICKS_PER_WINDOW)
                mid = max(0.05, min(0.95, mid))
                spread = 0.003
                bid = round(mid - spread/2, 6)
                ask = round(mid + spread/2, 6)

                # Down book (miroir approximatif)
                down_mid  = round(1.0 - mid + np.random.normal(0, 0.001), 6)
                down_bid  = round(down_mid - spread/2, 6)
                down_ask  = round(down_mid + spread/2, 6)

                # Binance book
                bn_mid    = coin_price * np.random.uniform(0.9999, 1.0001)
                bn_bid    = round(bn_mid * (1 - 0.0001), 2)
                bn_ask    = round(bn_mid * (1 + 0.0001), 2)
                bn_spread = round((bn_ask - bn_bid) / bn_mid * 10000, 4)

                secs_remaining = max(0, window_secs - t * (window_secs / TICKS_PER_WINDOW))
                ts = pd.Timestamp('2026-03-17 10:00:00', tz='UTC') + pd.Timedelta(seconds=w*window_secs + t*4)

                # Construire la row dans l'ordre EXACT du schéma v6
                row_dict = {
                    # identifiants
                    'timestamp_utc':         ts.isoformat(),
                    'coin':                  coin,
                    'interval_min':          str(interval),
                    'slug':                  slug,
                    'window_start_utc':      (ts - pd.Timedelta(seconds=t*4)).isoformat(),
                    'window_end_utc':        (ts + pd.Timedelta(seconds=secs_remaining)).isoformat(),
                    'seconds_remaining':     round(secs_remaining, 1),
                    'window_progress_pct':   round(1 - secs_remaining/window_secs, 4),
                    'hour_of_day':           ts.hour,
                    'day_of_week':           ts.dayofweek,
                    # signal
                    'coin_price':            round(coin_price, 2),
                    'target_price':          round(target_px, 2),
                    'distance_bps':          round((coin_price - target_px) / target_px * 10000, 2),
                    'coin_vol_10s_bps':      round(abs(np.random.normal(0, 2)), 4),
                    'coin_vol_60s_bps':      round(abs(np.random.normal(0, 5)), 4),
                    'coin_vol_15m_bps':      round(abs(np.random.normal(0, 10)), 4),
                    'z_score':               round(np.random.normal(0, 1), 4),
                    'analytical_prob_up':    round(mid + np.random.normal(0, 0.01), 6),
                    'edge_bps':              '', # calculé après
                    # poly book top of book
                    'up_bid':                bid,
                    'up_ask':                ask,
                    'up_mid':                round(mid, 6),
                    'up_spread_bps':         round(spread / mid * 10000, 4),
                    'up_best_bid_size':      round(np.random.uniform(50, 300), 2),
                    'up_best_ask_size':      round(np.random.uniform(50, 300), 2),
                    # book JSON complet
                    'up_book_bids':          make_book_json(mid, 'bid'),
                    'up_book_asks':          make_book_json(mid, 'ask'),
                    # aggregats
                    'up_bid_depth_top5':     round(np.random.uniform(200, 1000), 2),
                    'up_ask_depth_top5':     round(np.random.uniform(200, 1000), 2),
                    'up_depth_imbalance':    round(np.random.uniform(-0.5, 0.5), 6),
                    # poly flow
                    'poly_trade_count_10s':  np.random.randint(0, 5),
                    'poly_trade_count_60s':  np.random.randint(0, 20),
                    'poly_trade_size_10s':   round(np.random.uniform(0, 500), 2),
                    'poly_trade_size_60s':   round(np.random.uniform(0, 2000), 2),
                    # mid dynamics
                    'up_mid_change_5s':      round(np.random.normal(0, 0.001), 6),
                    'up_mid_change_10s':     round(np.random.normal(0, 0.002), 6),
                    'poly_mid_velocity':     round(np.random.normal(0, 0.0001), 6),
                    'poly_mid_acceleration': round(np.random.normal(0, 0.00001), 6),
                    # coin binance
                    'coin_return_10s_bps':   round(np.random.normal(0, 3), 4),
                    'coin_return_30s_bps':   round(np.random.normal(0, 5), 4),
                    'coin_return_60s_bps':   round(np.random.normal(0, 8), 4),
                    'coin_flow_imbalance_10s': round(np.random.uniform(-0.5, 0.5), 6),
                    'coin_flow_imbalance_60s': round(np.random.uniform(-0.5, 0.5), 6),
                    'coin_buy_ratio_60s':    round(np.random.uniform(0.3, 0.7), 6),
                    'coin_volume_usdt_60s':  round(np.random.uniform(1e5, 1e7), 2),
                    # cross-asset (btc/eth/sol/xrp)
                    'btc_return_10s_bps':    round(np.random.normal(0, 3), 4),
                    'btc_return_60s_bps':    round(np.random.normal(0, 8), 4),
                    'btc_flow_imbalance_60s': round(np.random.uniform(-0.5, 0.5), 6),
                    'btc_vol_60s_bps':       round(abs(np.random.normal(0, 5)), 4),
                    'eth_return_10s_bps':    round(np.random.normal(0, 3), 4),
                    'eth_return_60s_bps':    round(np.random.normal(0, 8), 4),
                    'eth_flow_imbalance_60s': round(np.random.uniform(-0.5, 0.5), 6),
                    'eth_vol_60s_bps':       round(abs(np.random.normal(0, 5)), 4),
                    'sol_return_10s_bps':    round(np.random.normal(0, 3), 4),
                    'sol_return_60s_bps':    round(np.random.normal(0, 8), 4),
                    'sol_flow_imbalance_60s': round(np.random.uniform(-0.5, 0.5), 6),
                    'sol_vol_60s_bps':       round(abs(np.random.normal(0, 5)), 4),
                    'xrp_return_10s_bps':    round(np.random.normal(0, 3), 4),
                    'xrp_return_60s_bps':    round(np.random.normal(0, 8), 4),
                    'xrp_flow_imbalance_60s': round(np.random.uniform(-0.5, 0.5), 6),
                    'xrp_vol_60s_bps':       round(abs(np.random.normal(0, 5)), 4),
                    'cross_asset_agreement_10s': round(np.random.uniform(-1, 1), 4),
                    'cross_asset_agreement_60s': round(np.random.uniform(-1, 1), 4),
                    'cross_asset_momentum_10s':  round(np.random.uniform(-1, 1), 4),
                    'cross_asset_momentum_60s':  round(np.random.uniform(-1, 1), 4),
                    # binance book L1-L5
                    'bn_best_bid':           round(bn_bid, 2),
                    'bn_best_ask':           round(bn_ask, 2),
                    'bn_mid':                round(bn_mid, 2),
                    'bn_spread_bps':         bn_spread,
                    'bn_bid_depth5':         round(np.random.uniform(5, 50), 4),
                    'bn_ask_depth5':         round(np.random.uniform(5, 50), 4),
                    'bn_depth_imbalance':    round(np.random.uniform(-0.5, 0.5), 6),
                    'bn_bid_l2_price':       round(bn_bid * 0.9999, 2),
                    'bn_bid_l2_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_bid_l3_price':       round(bn_bid * 0.9998, 2),
                    'bn_bid_l3_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_bid_l4_price':       round(bn_bid * 0.9997, 2),
                    'bn_bid_l4_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_bid_l5_price':       round(bn_bid * 0.9996, 2),
                    'bn_bid_l5_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_ask_l2_price':       round(bn_ask * 1.0001, 2),
                    'bn_ask_l2_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_ask_l3_price':       round(bn_ask * 1.0002, 2),
                    'bn_ask_l3_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_ask_l4_price':       round(bn_ask * 1.0003, 2),
                    'bn_ask_l4_qty':         round(np.random.uniform(0.1, 2), 4),
                    'bn_ask_l5_price':       round(bn_ask * 1.0004, 2),
                    'bn_ask_l5_qty':         round(np.random.uniform(0.1, 2), 4),
                    # down book
                    'down_bid':              round(down_bid, 6),
                    'down_ask':              round(down_ask, 6),
                    'down_mid':              round(down_mid, 6),
                    'down_spread_bps':       round(spread / down_mid * 10000, 4),
                    'down_best_bid_size':    round(np.random.uniform(50, 300), 2),
                    'down_best_ask_size':    round(np.random.uniform(50, 300), 2),
                    'down_book_bids':        make_book_json(down_mid, 'bid'),
                    'down_book_asks':        make_book_json(down_mid, 'ask'),
                    'down_bid_depth_top5':   round(np.random.uniform(200, 1000), 2),
                    'down_ask_depth_top5':   round(np.random.uniform(200, 1000), 2),
                    'down_depth_imbalance':  round(np.random.uniform(-0.5, 0.5), 6),
                    # outcome
                    'outcome':               outcome_val if t == TICKS_PER_WINDOW - 1 else '',
                    'coin_price_at_window_close': str(round(target_px * np.random.uniform(0.98, 1.02), 2)) if t == TICKS_PER_WINDOW - 1 else '',
                }
                # edge_bps = (prob - mid) * 10000
                prob = row_dict['analytical_prob_up']
                row_dict['edge_bps'] = round((prob - mid) * 10000, 4)

                rows.append(row_dict)

df_synth = pd.DataFrame(rows, columns=EXPECTED_HEADER)
print(f"  DataFrame synthétique : {len(df_synth)} lignes × {len(df_synth.columns)} colonnes")

# Vérification colonnes
if list(df_synth.columns) == EXPECTED_HEADER:
    ok(f"Colonnes exactement dans l'ordre du schéma v6 ({N_COLS} cols)")
else:
    missing = set(EXPECTED_HEADER) - set(df_synth.columns)
    extra   = set(df_synth.columns) - set(EXPECTED_HEADER)
    fail(f"Colonnes incorrectes — manquantes={missing} extra={extra}")

# ═════════════════════════════════════════════════════════════
# PARTIE 2 : Tests logique (logique de run_tests.py)
# ═════════════════════════════════════════════════════════════
section("2/6 — Tests de cohérence logique (run_tests.py)")

df = df_synth.copy()
def to_num(s):
    return pd.to_numeric(s.replace('', np.nan), errors='coerce')

n_ok = n_fail = 0
for coin in COINS:
    for iv in ['5', '15']:
        sub = df[(df['coin'] == coin) & (df['interval_min'].astype(str) == iv)]
        k = f"{coin}-{iv}m ({len(sub)} rows)"

        bid = to_num(sub['up_bid']); ask = to_num(sub['up_ask'])
        mid = to_num(sub['up_mid'])

        # 1. bid < ask
        mask = bid.notna() & ask.notna()
        v = (bid[mask] >= ask[mask]).sum()
        if v == 0: ok(f"{k}: up_bid < up_ask"); n_ok+=1
        else: fail(f"{k}: {v} rows où bid >= ask"); n_fail+=1

        # 2. mid = (bid+ask)/2
        exp_mid = (bid + ask) / 2
        diff = (mid - exp_mid).abs().dropna()
        if (diff > 0.0001).sum() == 0: ok(f"{k}: up_mid = (bid+ask)/2"); n_ok+=1
        else: fail(f"{k}: {(diff>0.0001).sum()} rows où mid ≠ (bid+ask)/2"); n_fail+=1

        # 3. prices ∈ [0,1]
        for col in ['up_bid', 'up_ask', 'up_mid']:
            v2 = to_num(sub[col]).dropna()
            out = ((v2 < 0) | (v2 > 1)).sum()
            if out == 0: ok(f"{k}: {col} ∈ [0,1]"); n_ok+=1
            else: fail(f"{k}: {col} — {out} valeurs hors [0,1]"); n_fail+=1

        # 4. vol >= 0
        vol = to_num(sub['coin_vol_60s_bps']).dropna()
        neg = (vol < 0).sum()
        if neg == 0: ok(f"{k}: coin_vol_60s_bps ≥ 0"); n_ok+=1
        else: fail(f"{k}: {neg} valeurs négatives"); n_fail+=1

        # 5. seconds_remaining ∈ [0, interval*60]
        sec = to_num(sub['seconds_remaining']).dropna()
        max_sec = int(iv) * 60
        out = ((sec < 0) | (sec > max_sec + 5)).sum()
        if out == 0: ok(f"{k}: seconds_remaining ∈ [0, {max_sec}]"); n_ok+=1
        else: fail(f"{k}: {out} valeurs hors range"); n_fail+=1

        # 6. outcome ∈ {0, 1, ''}
        outcomes = sub['outcome'].replace('', np.nan).dropna().unique()
        invalid = set(str(o) for o in outcomes) - {'0', '1'}
        if not invalid: ok(f"{k}: outcome valide ({sorted(outcomes)})"); n_ok+=1
        else: fail(f"{k}: outcomes invalides {invalid}"); n_fail+=1

        # 7. Binance: bn_bid < bn_ask
        bn_bid = to_num(sub['bn_best_bid']); bn_ask2 = to_num(sub['bn_best_ask'])
        mask2 = bn_bid.notna() & bn_ask2.notna()
        viols = (bn_bid[mask2] >= bn_ask2[mask2]).sum()
        if viols == 0: ok(f"{k}: bn_bid < bn_ask"); n_ok+=1
        else: fail(f"{k}: {viols} rows où bn_bid >= bn_ask"); n_fail+=1

        # 8. bn_depth_imbalance ∈ [-1, 1]
        imb = to_num(sub['bn_depth_imbalance']).dropna()
        out2 = ((imb < -1.01) | (imb > 1.01)).sum()
        if out2 == 0: ok(f"{k}: bn_depth_imbalance ∈ [-1,1]"); n_ok+=1
        else: fail(f"{k}: {out2} imbalances hors [-1,1]"); n_fail+=1

        # 9. down_bid < down_ask
        dn_bid = to_num(sub['down_bid']); dn_ask = to_num(sub['down_ask'])
        maskd = dn_bid.notna() & dn_ask.notna()
        viols_d = (dn_bid[maskd] >= dn_ask[maskd]).sum()
        if viols_d == 0: ok(f"{k}: down_bid < down_ask"); n_ok+=1
        else: fail(f"{k}: {viols_d} rows où down_bid >= down_ask"); n_fail+=1

        # 10. down_mid + up_mid ≈ 1.0
        dm = to_num(sub['down_mid']); um = to_num(sub['up_mid'])
        mask_both = dm.notna() & um.notna()
        diff2 = ((dm[mask_both] + um[mask_both]) - 1.0).abs()
        bad = (diff2 > 0.05).sum()
        if bad == 0: ok(f"{k}: down_mid + up_mid ≈ 1.0"); n_ok+=1
        else: warn(f"{k}: {bad} rows où down_mid + up_mid ≠ 1 (spread ok si < 5%)"); n_ok+=1

        # 11. edge_bps = (prob - mid) * 10000
        prob = to_num(sub['analytical_prob_up'])
        edge = to_num(sub['edge_bps'])
        mask3 = prob.notna() & mid.notna() & edge.notna()
        if mask3.sum() > 0:
            exp_edge = (prob[mask3] - to_num(sub['up_mid'])[mask3]) * 10000
            diff3 = (edge[mask3] - exp_edge).abs()
            if (diff3 > 1.0).sum() == 0: ok(f"{k}: edge_bps = (prob-mid)*10000"); n_ok+=1
            else: fail(f"{k}: {(diff3>1.0).sum()} rows edge_bps incorrect"); n_fail+=1

        # 12. No duplicate (slug, timestamp)
        dupes = sub.groupby(['slug','timestamp_utc']).size()
        nd = (dupes > 1).sum()
        if nd == 0: ok(f"{k}: pas de doublons (slug, timestamp)"); n_ok+=1
        else: fail(f"{k}: {nd} doublons"); n_fail+=1

print(f"\n  → Cohérence logique : {n_ok} OK, {n_fail} FAIL")

# ═════════════════════════════════════════════════════════════
# PARTIE 3 : Validation colonnes JSON book (NOUVEAU v6)
# ═════════════════════════════════════════════════════════════
section("3/6 — Validation colonnes JSON book v6")

for json_col in ['up_book_bids', 'up_book_asks', 'down_book_bids', 'down_book_asks']:
    if json_col not in df.columns:
        fail(f"Colonne manquante : {json_col}"); continue

    sample = df[df[json_col] != ''][json_col].head(50)
    bad_parse = 0; empty = 0; bad_format = 0
    min_levels = 999; max_levels = 0
    for v in sample:
        try:
            parsed = json.loads(v) if isinstance(v, str) else v
            if not isinstance(parsed, list): bad_format += 1; continue
            if len(parsed) == 0: empty += 1; continue
            min_levels = min(min_levels, len(parsed))
            max_levels = max(max_levels, len(parsed))
            # Vérifier structure [price, size]
            for level in parsed[:3]:
                assert len(level) == 2
                assert 0 < float(level[0]) < 1   # prix Polymarket ∈ (0,1)
                assert float(level[1]) >= 0        # size >= 0
        except Exception as e:
            bad_parse += 1

    if bad_parse == 0 and bad_format == 0:
        ok(f"{json_col} : {len(sample)} valeurs valides, {min_levels}–{max_levels} niveaux par tick")
    else:
        fail(f"{json_col} : {bad_parse} erreurs parse, {bad_format} format incorrect")

    # Vérifier l'ordre (bids: décroissant, asks: croissant)
    order_ok = 0; order_bad = 0
    direction = 'desc' if 'bid' in json_col else 'asc'
    for v in sample[:20]:
        parsed = json.loads(v) if isinstance(v, str) else v
        if len(parsed) < 2: continue
        prices = [float(p[0]) for p in parsed]
        if direction == 'desc':
            ordered = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
        else:
            ordered = all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
        if ordered: order_ok += 1
        else: order_bad += 1

    if order_bad == 0:
        ok(f"{json_col} : ordre {'décroissant' if direction=='desc' else 'croissant'} ✓ ({order_ok} checks)")
    else:
        fail(f"{json_col} : {order_bad} niveaux mal ordonnés")

# ═════════════════════════════════════════════════════════════
# PARTIE 4 : Simulation de inspect_live.py
# ═════════════════════════════════════════════════════════════
section("4/6 — Simulation de inspect_live.py")

# Reproduire exactement la logique de checks de inspect_live.py
checks = {
    'up_bid':               '🎯 Entry price',
    'up_ask':               '🎯 Entry price',
    'up_best_bid_size':     '🎯 Fill size',
    'up_best_ask_size':     '🎯 Fill size',
    'up_book_bids':         '🎯 Book complet bids (JSON)',
    'up_book_asks':         '🎯 Book complet asks (JSON)',
    'edge_bps':             '📊 Main signal',
    'analytical_prob_up':   '📊 Probability',
    'z_score':              '📊 Z-score',
    'distance_bps':         '📊 Distance',
    'coin_vol_60s_bps':     '📊 Vol',
    'coin_return_60s_bps':  '📊 Return',   # présent dans df_synth ?
    'coin_flow_imbalance_60s': '📊 Flow',
    'poly_trade_count_60s': '📊 Poly trades',
    'up_mid_change_10s':    '📊 Mid momentum',
    'cross_asset_agreement_60s': '📊 Cross-asset',
    'down_bid':             '📉 DOWN best bid',
    'down_ask':             '📉 DOWN best ask',
    'down_best_bid_size':   '📉 DOWN fill size',
    'down_best_ask_size':   '📉 DOWN fill size',
    'down_book_bids':       '📉 DOWN book bids (JSON)',
    'down_book_asks':       '📉 DOWN book asks (JSON)',
    'down_depth_imbalance': '📉 DOWN depth imb',
    'bn_best_bid':          '📈 Binance best bid',
    'bn_best_ask':          '📈 Binance best ask',
    'bn_spread_bps':        '📈 Binance spread',
    'bn_depth_imbalance':   '📈 Binance depth imb',
    'bn_bid_depth5':        '📈 Binance bid depth',
    'bn_ask_depth5':        '📈 Binance ask depth',
    'bn_bid_l2_price':      '📈 Binance L2 bid',
    'bn_ask_l2_price':      '📈 Binance L2 ask',
    'bn_bid_l5_price':      '📈 Binance L5 bid',
    'bn_ask_l5_price':      '📈 Binance L5 ask',
    'outcome':              '✅ Label',
    'coin_price_at_window_close': '✅ Close price',
    # Colonnes qui ne doivent PLUS être là
    'bid_l2_price':         '❌ DEVRAIT ÊTRE ABSENT (v5)',
    'ask_l2_price':         '❌ DEVRAIT ÊTRE ABSENT (v5)',
    'down_bid_l2_price':    '❌ DEVRAIT ÊTRE ABSENT (v5)',
}

all_ok = True
absent_ok = []   # colonnes v5 absentes comme voulu
for col, desc in checks.items():
    present = col in df.columns
    if '❌' in desc:
        # Ces colonnes doivent être ABSENTES en v6
        if not present:
            ok(f"  {col:<28} → absent ✓ (colonne v5 bien supprimée)")
        else:
            fail(f"  {col:<28} → PRÉSENT ✗ (devrait être absent en v6 !)")
            all_ok = False
        continue

    if not present:
        fail(f"  {col:<28} → MANQUANT {desc}")
        all_ok = False
        continue

    # Calculer le taux de remplissage
    # Colonnes naturellement sparse : 1 tick par fenêtre = ~1-5% selon la durée
    SPARSE_COLS = {'outcome', 'coin_price_at_window_close'}

    if col in ('up_book_bids','up_book_asks','down_book_bids','down_book_asks'):
        # JSON : taux de non-vide
        src = df[col]
        non_empty = (src != '') & (src != '[]') & src.notna()
        pct = non_empty.mean() * 100
        if pct > 95:
            ok(f"  {col:<28} → {pct:.1f}% non-vide  {desc}")
        else:
            warn(f"  {col:<28} → {pct:.1f}% non-vide (bas ?) {desc}")
    elif col in SPARSE_COLS:
        # Ces colonnes ne sont remplies qu'au dernier tick de chaque fenêtre (normal)
        src = df[col]
        if col == 'outcome':
            filled = (src.replace('', np.nan).notna()).mean() * 100
        else:
            filled = pd.to_numeric(src.replace('', np.nan), errors='coerce').notna().mean() * 100
        if filled > 0:
            ok(f"  {col:<28} → {filled:.1f}% rempli  {desc}  (sparse=normal : 1 tick/fenêtre)")
        else:
            fail(f"  {col:<28} → 0% rempli — colonne jamais écrite {desc}")
    else:
        src = df[col]
        filled = pd.to_numeric(src.replace('', np.nan), errors='coerce').notna().mean() * 100
        if filled > 90:
            ok(f"  {col:<28} → {filled:.1f}% rempli  {desc}")
        elif filled > 50:
            warn(f"  {col:<28} → {filled:.1f}% rempli  {desc}")
        else:
            fail(f"  {col:<28} → {filled:.1f}% rempli  {desc}")

# ═════════════════════════════════════════════════════════════
# PARTIE 5 : Vérifier la lecture JSON dans data.py (backtest)
# ═════════════════════════════════════════════════════════════
section("5/6 — Vérification désérialisation JSON (backtest/data.py)")

# backtest path already in sys.path via _BACKTEST above
import importlib.util

# Importer data.py en isolé
spec = importlib.util.spec_from_file_location("data", str(_BACKTEST / "data.py"))
data_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_mod)

# Simuler le postprocessing JSON de load() sur notre df synthétique
df_bt = df_synth.copy()

# Convertir les colonnes numériques
for col in data_mod.NUMERIC_COLS:
    if col in df_bt.columns:
        df_bt[col] = pd.to_numeric(
            df_bt[col].astype(str).replace({'': np.nan, 'nan': np.nan, 'None': np.nan}),
            errors='coerce'
        )

# Désérialiser les JSON books
for col in ('up_book_bids','up_book_asks','down_book_bids','down_book_asks'):
    if col in df_bt.columns:
        def _parse(val):
            if not val or str(val) in ('', 'nan', 'None', '[]'): return []
            try: return json.loads(val)
            except: return []
        df_bt[col] = df_bt[col].apply(_parse)

# Vérifier que les colonnes JSON sont bien des listes Python
for col in ('up_book_bids','up_book_asks','down_book_bids','down_book_asks'):
    sample = df_bt[col].head(5)
    all_lists = all(isinstance(v, list) for v in sample)
    non_empty = (df_bt[col].apply(lambda x: len(x) > 0)).mean() * 100
    if all_lists:
        ok(f"data.py : {col} → liste Python après désérialisation ({non_empty:.0f}% non-vide)")
    else:
        fail(f"data.py : {col} → pas correctement désérialisé")

# Vérifier que les colonnes L2-L5 Poly ne sont plus dans NUMERIC_COLS
old_cols = ['bid_l2_price','bid_l2_size','ask_l2_price','ask_l2_size']
still_present = [c for c in old_cols if c in data_mod.NUMERIC_COLS]
if not still_present:
    ok(f"data.py : colonnes L2-L5 Poly absentes de NUMERIC_COLS ✓")
else:
    fail(f"data.py : colonnes L2-L5 encore dans NUMERIC_COLS : {still_present}")

# Vérifier qu'execution.py lit bien le book JSON désérialisé
# backtest path already in sys.path via _BACKTEST above
from execution import _build_ask_levels, _build_bid_levels

# Tick avec book JSON (format v6)
row = df_bt[df_bt['up_book_asks'].apply(lambda x: len(x) > 0)].iloc[0].to_dict()
asks = _build_ask_levels(row)
bids = _build_bid_levels(row)

if len(asks) >= 5:
    ok(f"execution.py : _build_ask_levels lit le JSON book ({len(asks)} niveaux, L1 ask={asks[0][0]:.4f})")
else:
    fail(f"execution.py : _build_ask_levels retourne seulement {len(asks)} niveaux (attendu ≥5)")

if len(bids) >= 5:
    ok(f"execution.py : _build_bid_levels lit le JSON book ({len(bids)} niveaux, L1 bid={bids[0][0]:.4f})")
else:
    fail(f"execution.py : _build_bid_levels retourne seulement {len(bids)} niveaux")

# Test simulate_buy sur tick v6
from execution import simulate_buy, simulate_limit_buy
fill_market = simulate_buy(row, 100.0)
if fill_market and fill_market.filled_size == 100.0:
    ok(f"execution.py : simulate_buy sur tick v6 → fill={fill_market.filled_size:.0f} USDC @{fill_market.avg_price:.4f}")
else:
    fail(f"execution.py : simulate_buy échoué sur tick v6")

# ═════════════════════════════════════════════════════════════
# PARTIE 6 : Vérifier que les anciennes colonnes sont bien absentes
# ═════════════════════════════════════════════════════════════
section("6/6 — Vérification suppression colonnes v5")

v5_cols_poly = [
    'bid_l2_price','bid_l2_size','bid_l3_price','bid_l3_size',
    'bid_l4_price','bid_l4_size','bid_l5_price','bid_l5_size',
    'ask_l2_price','ask_l2_size','ask_l3_price','ask_l3_size',
    'ask_l4_price','ask_l4_size','ask_l5_price','ask_l5_size',
    'down_bid_l2_price','down_bid_l2_size','down_bid_l3_price','down_bid_l3_size',
    'down_bid_l4_price','down_bid_l4_size','down_bid_l5_price','down_bid_l5_size',
    'down_ask_l2_price','down_ask_l2_size','down_ask_l3_price','down_ask_l3_size',
    'down_ask_l4_price','down_ask_l4_size','down_ask_l5_price','down_ask_l5_size',
]
still_in_schema = [c for c in v5_cols_poly if c in EXPECTED_HEADER]
still_in_df     = [c for c in v5_cols_poly if c in df_synth.columns]

if not still_in_schema:
    ok(f"csv_schema.py : 0/{len(v5_cols_poly)} colonnes L2-L5 Poly encore présentes ✓")
else:
    fail(f"csv_schema.py : {len(still_in_schema)} colonnes L2-L5 encore présentes : {still_in_schema}")

if not still_in_df:
    ok(f"DataFrame v6 : aucune colonne L2-L5 Poly résiduelle ✓")
else:
    fail(f"DataFrame v6 : {len(still_in_df)} colonnes L2-L5 résiduelles")

# Vérifier les nouvelles colonnes JSON bien présentes
new_cols = ['up_book_bids','up_book_asks','down_book_bids','down_book_asks']
all_present = all(c in EXPECTED_HEADER for c in new_cols)
if all_present:
    ok(f"csv_schema.py : 4 colonnes JSON book bien présentes : {new_cols}")
else:
    fail(f"csv_schema.py : colonnes JSON manquantes")

# ═════════════════════════════════════════════════════════════
# RÉSUMÉ
# ═════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print(f"{BOLD}  RÉSUMÉ VALIDATION v6{NC}")
print(f"{'═'*55}")
n_pass = sum(results)
n_fail_total = len(results) - n_pass
print(f"  {PASS} Passés  : {n_pass}")
print(f"  {FAIL} Échoués : {n_fail_total}")

if n_fail_total == 0:
    print(f"\n  {GREEN}{BOLD}✅ Tous les checks passent — code v6 prêt pour déploiement{NC}")
else:
    print(f"\n  {RED}{BOLD}❌ {n_fail_total} problème(s) détecté(s) — corriger avant déploiement{NC}")
    sys.exit(1)
