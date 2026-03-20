"""
inspect_live.py — Vérifie la qualité des données enregistrées par pm_watcher v4
Usage: python3 inspect_live.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

live_dir = Path("historical_data/live")
files = sorted(live_dir.glob("*.parquet"))

if not files:
    print("No parquet files found in historical_data/live/")
    exit(1)

print(f"Files found: {[f.name for f in files]}")
print()

dfs = []
for f in files:
    df = pd.read_parquet(f)
    dfs.append(df)
    print(f"{f.name}: {len(df):,} rows | {len(df.columns)} cols")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df):,} rows | {len(df.columns)} columns")
print(f"Period: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
print(f"Coins: {sorted(df['coin'].unique())}")
print(f"Intervals: {sorted(df['interval_min'].astype(str).unique())}")

print("\n" + "="*60)
print("SCHEMA CHECK")
print("="*60)
try:
    from csv_schema import EXPECTED_HEADER, N_COLS
    missing = [c for c in EXPECTED_HEADER if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_HEADER]
    print(f"Expected: {N_COLS} | Got: {len(df.columns)}")
    print(f"Missing cols: {missing if missing else 'none ✓'}")
    print(f"Extra cols:   {extra if extra else 'none ✓'}")
except ImportError:
    print(f"Got: {len(df.columns)} columns")

print("\n" + "="*60)
print("DATA QUALITY — BTC 5m")
print("="*60)
btc = df[(df['coin']=='btc') & (df['interval_min'].astype(str)=='5')].copy()
print(f"Rows: {len(btc):,}")

def fill_rate(col, src=None):
    src = src if src is not None else btc
    if col not in src.columns:
        return "MISSING"
    # Colonnes JSON book : vérifier non-vide plutôt que numérique
    if col in ("up_book_bids","up_book_asks","down_book_bids","down_book_asks"):
        filled = src[col].apply(lambda x:
            bool(x) if isinstance(x, list)
            else str(x).strip() not in ("","[]","None","nan")
        ).mean()*100
        return f"{filled:.1f}%"
    s = pd.to_numeric(src[col].replace('', np.nan), errors='coerce')
    return f"{s.notna().mean()*100:.1f}%"

checks = {
    # ── Backtest critical ──
    "up_bid":               "🎯 Entry price",
    "up_ask":               "🎯 Entry price",
    "up_best_bid_size":     "🎯 Fill size",
    "up_best_ask_size":     "🎯 Fill size",
    "up_book_bids":         "🎯 Book complet bids (JSON)",
    "up_book_asks":         "🎯 Book complet asks (JSON)",
    # ── ML features — Polymarket ──
    "edge_bps":             "📊 Main signal",
    "analytical_prob_up":   "📊 Probability",
    "z_score":              "📊 Z-score",
    "distance_bps":         "📊 Distance",
    "coin_vol_60s_bps":     "📊 Vol",
    "coin_return_60s_bps":  "📊 Return",
    "coin_flow_imbalance_60s": "📊 Flow imbalance",
    "poly_trade_count_60s": "📊 Poly trades",
    "up_mid_change_10s":    "📊 Mid momentum",
    "cross_asset_agreement_60s": "📊 Cross-asset",
    # ── DOWN book ──
    "down_bid":             "📉 DOWN best bid",
    "down_ask":             "📉 DOWN best ask",
    "down_best_bid_size":   "📉 DOWN fill size",
    "down_best_ask_size":   "📉 DOWN fill size",
    "down_book_bids":       "📉 DOWN book complet bids (JSON)",
    "down_book_asks":       "📉 DOWN book complet asks (JSON)",
    "down_depth_imbalance": "📉 DOWN depth imb",
    # ── ML features — Binance book ──
    "bn_best_bid":          "📈 Binance best bid",
    "bn_best_ask":          "📈 Binance best ask",
    "bn_spread_bps":        "📈 Binance spread",
    "bn_depth_imbalance":   "📈 Binance depth imb",
    "bn_bid_depth5":        "📈 Binance bid depth",
    "bn_ask_depth5":        "📈 Binance ask depth",
    "bn_bid_l2_price":      "📈 Binance L2 bid",
    "bn_ask_l2_price":      "📈 Binance L2 ask",
    "bn_bid_l5_price":      "📈 Binance L5 bid",
    "bn_ask_l5_price":      "📈 Binance L5 ask",
    # ── Outcome ──
    "outcome":              "✅ Label",
    "coin_price_at_window_close": "✅ Close price",
}

all_ok = True
for col, desc in checks.items():
    rate = fill_rate(col)
    if rate == "MISSING":
        status = "✗"; all_ok = False
    elif rate == "0.0%":
        status = "✗"; all_ok = False
    elif float(rate.rstrip('%')) < 50:
        status = "⚠"; all_ok = False
    else:
        status = "✓"
    print(f"  {status} {col:<30} {rate:>6}  {desc}")

print("\n" + "="*60)
print("TICK RATE CHECK")
print("="*60)
btc_ts = pd.to_datetime(
    btc['timestamp_utc'].astype(str).str.replace('Z','+00:00'),
    utc=True, errors='coerce'
).dropna().sort_values()

if len(btc_ts) > 1:
    diffs = btc_ts.diff().dropna().dt.total_seconds()
    median_tick = diffs.median()
    print(f"  Median tick interval: {median_tick:.3f}s (expected 0.25s)")
    print(f"  Mean tick interval:   {diffs.mean():.3f}s")
    print(f"  Tick rate:            ~{1/median_tick:.1f} Hz")
    gaps = diffs[diffs > 5]
    print(f"  Gaps > 5s:            {len(gaps)}")
    if median_tick <= 0.30:
        print("  ✓ Tick rate correct (≤ 0.30s)")
    else:
        print(f"  ⚠ Tick rate slow ({median_tick:.2f}s > 0.30s)")

print("\n" + "="*60)
print("BINANCE BOOK SAMPLE VALUES")
print("="*60)
for col in ["bn_best_bid","bn_spread_bps","bn_depth_imbalance","bn_bid_depth5"]:
    if col in btc.columns:
        s = pd.to_numeric(btc[col].replace('', np.nan), errors='coerce').dropna()
        if len(s) > 0:
            print(f"  {col}: min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")
        else:
            print(f"  {col}: all empty ← depthUpdate bug still present?")

print("\n" + "="*60)
print("VERDICT")
print("="*60)
if all_ok:
    print(f"✅ All checks passed — {len(btc):,} BTC 5m rows ready for backtest + ML")
else:
    print("⚠️  Some columns have issues — see above")
