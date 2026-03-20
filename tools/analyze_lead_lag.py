"""
analyze_lead_lag.py — Mesure empirique du lead Binance → Polymarket

Utilise les colonnes pré-calculées coin_return_Xs_bps du logger
(rolling returns sur 10s, 30s, 60s) pour détecter les moves Binance.

Usage : python3 analyze_lead_lag.py --dir historical_data/live [--coin btc] [--interval 5]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WINDOW_TICKS  = 20    # fenêtre de recherche après le move (~5s à 4Hz)
MIN_SAMPLES   = 10    # minimum d'événements pour afficher les stats
MID_MOVE_BPS  = 3     # seuil de réaction du mid Polymarket (bps)

# Colonnes de retour Binance disponibles dans nos données
BN_RETURN_COLS = {
    "10s":  "coin_return_10s_bps",
    "30s":  "coin_return_30s_bps",
    "60s":  "coin_return_60s_bps",
}

def load_parquets(data_dir: str) -> pd.DataFrame:
    p = Path(data_dir)
    files = sorted(p.glob("*.parquet")) + sorted(p.glob("*_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Aucun parquet dans {data_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  ⚠ {f.name} : {e}")
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Colonnes numériques
    for col in list(BN_RETURN_COLS.values()) + \
               ["up_mid", "up_bid", "up_ask", "coin_price",
                "coin_flow_imbalance_60s", "up_depth_imbalance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp_utc").reset_index(drop=True)


def analyze_one(sub: pd.DataFrame, return_col: str, thresholds: list) -> dict:
    """
    Pour chaque seuil, cherche les moves Binance et mesure le délai
    avant que up_mid bouge de MID_MOVE_BPS bps.
    """
    if return_col not in sub.columns:
        return {}

    results = {}
    for thresh in thresholds:
        lags = []

        # Événements UP : retour Binance dépasse +thresh
        up_events = sub.index[
            (sub[return_col] >= thresh) &
            (sub[return_col].shift(1).fillna(0) < thresh)
        ]
        # Événements DOWN : retour Binance dépasse -thresh
        dn_events = sub.index[
            (sub[return_col] <= -thresh) &
            (sub[return_col].shift(1).fillna(0) > -thresh)
        ]

        for idx in list(up_events) + list(dn_events):
            if idx + WINDOW_TICKS >= len(sub):
                continue
            mid0 = sub.loc[idx, "up_mid"]
            if pd.isna(mid0) or mid0 <= 0:
                continue

            for lag in range(1, WINDOW_TICKS + 1):
                mid_after = sub.loc[idx + lag, "up_mid"]
                if pd.isna(mid_after):
                    continue
                if abs(mid_after - mid0) / mid0 * 10000 >= MID_MOVE_BPS:
                    lags.append(lag * 250)  # en ms
                    break

        n_events = len(up_events) + len(dn_events)
        if len(lags) < MIN_SAMPLES:
            results[thresh] = {"n_events": n_events, "n_reacted": len(lags),
                               "insufficient": True}
            continue

        lags_arr = np.array(lags)
        results[thresh] = {
            "n_events":    n_events,
            "n_reacted":   len(lags),
            "react_rate":  round(len(lags) / n_events * 100, 1),
            "p25_ms":      int(np.percentile(lags_arr, 25)),
            "median_ms":   int(np.median(lags_arr)),
            "p75_ms":      int(np.percentile(lags_arr, 75)),
            "p95_ms":      int(np.percentile(lags_arr, 95)),
        }

    return results


def analyze_ask_bid_lead(sub: pd.DataFrame, thresh: float, return_col: str) -> dict:
    """
    Lequel réagit en premier : ask ou bid ?
    Sur un move UP Binance : ask devrait monter avant bid.
    """
    if return_col not in sub.columns:
        return {}

    ask_lags, bid_lags = [], []
    events = sub.index[sub[return_col] >= thresh]

    for idx in events:
        if idx + WINDOW_TICKS >= len(sub):
            continue
        ask0 = sub.loc[idx, "up_ask"]
        bid0 = sub.loc[idx, "up_bid"]
        if pd.isna(ask0) or pd.isna(bid0) or ask0 <= 0 or bid0 <= 0:
            continue

        ask_lag = bid_lag = None
        for lag in range(1, WINDOW_TICKS + 1):
            if ask_lag is None:
                v = sub.loc[idx + lag, "up_ask"]
                if not pd.isna(v) and abs(v - ask0) / ask0 * 10000 >= MID_MOVE_BPS:
                    ask_lag = lag * 250
            if bid_lag is None:
                v = sub.loc[idx + lag, "up_bid"]
                if not pd.isna(v) and abs(v - bid0) / bid0 * 10000 >= MID_MOVE_BPS:
                    bid_lag = lag * 250
            if ask_lag and bid_lag:
                break

        if ask_lag and bid_lag:
            ask_lags.append(ask_lag)
            bid_lags.append(bid_lag)

    if len(ask_lags) < MIN_SAMPLES:
        return {"n": len(ask_lags), "insufficient": True}

    return {
        "n":              len(ask_lags),
        "ask_median_ms":  int(np.median(ask_lags)),
        "bid_median_ms":  int(np.median(bid_lags)),
        "ask_first_pct":  round(sum(a < b for a, b in
                          zip(ask_lags, bid_lags)) / len(ask_lags) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",      default="historical_data/live")
    parser.add_argument("--coin",     default=None)
    parser.add_argument("--interval", type=int, default=None)
    args = parser.parse_args()

    print("\n" + "="*62)
    print("  Analyse Lead-Lag Binance → Polymarket")
    print("="*62)

    df = load_parquets(args.dir)
    n  = len(df)
    t0 = df["timestamp_utc"].min().strftime("%H:%M")
    t1 = df["timestamp_utc"].max().strftime("%H:%M")
    print(f"  {n:,} ticks | {t0} → {t1} UTC\n")

    coins     = [args.coin]     if args.coin     else ["btc","eth","sol","xrp"]
    intervals = [args.interval] if args.interval else [5, 15]

    # Seuils adaptés aux rolling returns (10s, 30s)
    THRESHOLDS = {
        "10s":  [10, 20, 30, 50],    # bps sur 10 secondes
        "30s":  [20, 30, 50, 100],   # bps sur 30 secondes
    }

    for coin in coins:
        for iv in intervals:
            mask = (df["coin"] == coin) & (df["interval_min"] == str(iv))
            sub  = df[mask].copy().reset_index(drop=True)
            if len(sub) < 200:
                continue

            key = f"{coin}-{iv}m"
            print(f"{'─'*62}")
            print(f"  {key}  ({len(sub):,} ticks)")
            print(f"{'─'*62}")

            for window, return_col in [("10s", "coin_return_10s_bps"),
                                       ("30s", "coin_return_30s_bps")]:
                if return_col not in sub.columns:
                    continue
                thresholds = THRESHOLDS[window]
                results = analyze_one(sub, return_col, thresholds)
                if not results:
                    continue

                print(f"\n  Retour Binance {window} :")
                print(f"  {'Seuil':>8}  {'Evt':>5}  {'Réact':>6}  "
                      f"{'p25':>6}  {'Médian':>8}  {'p75':>6}  {'p95':>6}")
                for thresh, r in results.items():
                    if r.get("insufficient"):
                        print(f"  {thresh:>6}bps  {r['n_events']:>5}  "
                              f"{r['n_reacted']:>6}  (insuffisant)")
                    else:
                        print(f"  {thresh:>6}bps  {r['n_events']:>5}  "
                              f"{r['react_rate']:>5.0f}%  "
                              f"{r['p25_ms']:>5}ms  {r['median_ms']:>7}ms  "
                              f"{r['p75_ms']:>5}ms  {r['p95_ms']:>5}ms")

            # Ask vs Bid lead
            ret_col = "coin_return_10s_bps"
            if ret_col in sub.columns:
                avb = analyze_ask_bid_lead(sub, 20, ret_col)
                if not avb.get("insufficient"):
                    print(f"\n  Ask vs Bid après move ≥20bps/10s "
                          f"(n={avb['n']}) :")
                    print(f"    Ask médian : {avb['ask_median_ms']}ms  |  "
                          f"Bid médian : {avb['bid_median_ms']}ms  |  "
                          f"Ask réagit en 1er : {avb['ask_first_pct']}% du temps")

            print()

    print("="*62)
    print("  Interprétation :")
    print("  • Médian < 1000ms → lead signal exploitable")
    print("  • Médian 250-750ms → idéal pour limit orders anticipatoires")
    print("  • Ask réagit avant bid → les vendeurs anticipent les moves UP")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
