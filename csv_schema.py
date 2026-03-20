"""
csv_schema.py
─────────────────
101 colonnes orientées ML + backtest précis (UP + DOWN book).

Nouveautés v3 :
  - up_best_bid_size / up_best_ask_size : taille USDC au meilleur niveau
  - bid/ask_l2..l5_price / bid/ask_l2..l5_size : niveaux 2-5 du book
  - Source : WebSocket event-driven (plus de REST polling pour le book)
  - Tick rate : 250ms (4 Hz) — fix: next_log_mono += (pas = now+)
  - Binance order book L1-L5 (bid/ask depth, spread, imbalance) @ 100ms
"""

EXPECTED_HEADER = [
    # ── Identifiants marché ───────────────────────────────────────────────────
    "timestamp_utc",
    "coin",
    "interval_min",
    "slug",
    "window_start_utc",
    "window_end_utc",
    "seconds_remaining",
    "window_progress_pct",
    "hour_of_day",
    "day_of_week",

    # ── Signal principal ──────────────────────────────────────────────────────
    "coin_price",
    "target_price",
    "distance_bps",
    "coin_vol_10s_bps",
    "coin_vol_60s_bps",
    "coin_vol_15m_bps",
    "z_score",
    "analytical_prob_up",
    "edge_bps",

    # ── Polymarket book — top of book ─────────────────────────────────────────
    "up_bid",                   # meilleur bid ∈ [0,1]
    "up_ask",                   # meilleur ask ∈ [0,1]
    "up_mid",
    "up_spread_bps",
    "up_best_bid_size",         # ★ USDC dispo au meilleur bid
    "up_best_ask_size",         # ★ USDC dispo au meilleur ask

    # ── Polymarket book — complet JSON (remplace les niveaux 2-5 fixes) ─────
    "up_book_bids",   # JSON : "[[0.55,150],[0.54,200],...]" — book complet bids
    "up_book_asks",   # JSON : "[[0.56,100],[0.57,300],...]" — book complet asks

    # ── Polymarket book — agrégés top 5 ──────────────────────────────────────
    "up_bid_depth_top5",
    "up_ask_depth_top5",
    "up_depth_imbalance",

    # ── Polymarket flow ───────────────────────────────────────────────────────
    "poly_trade_count_10s",
    "poly_trade_count_60s",
    "poly_trade_size_10s",
    "poly_trade_size_60s",

    # ── Dynamique mid ─────────────────────────────────────────────────────────
    "up_mid_change_5s",
    "up_mid_change_10s",
    "poly_mid_velocity",
    "poly_mid_acceleration",

    # ── Coin Binance ──────────────────────────────────────────────────────────
    "coin_return_10s_bps",
    "coin_return_30s_bps",
    "coin_return_60s_bps",
    "coin_flow_imbalance_10s",
    "coin_flow_imbalance_60s",
    "coin_buy_ratio_60s",
    "coin_volume_usdt_60s",

    # ── Cross-asset ───────────────────────────────────────────────────────────
    "btc_return_10s_bps",  "btc_return_60s_bps",
    "btc_flow_imbalance_60s", "btc_vol_60s_bps",
    "eth_return_10s_bps",  "eth_return_60s_bps",
    "eth_flow_imbalance_60s", "eth_vol_60s_bps",
    "sol_return_10s_bps",  "sol_return_60s_bps",
    "sol_flow_imbalance_60s", "sol_vol_60s_bps",
    "xrp_return_10s_bps",  "xrp_return_60s_bps",
    "xrp_flow_imbalance_60s", "xrp_vol_60s_bps",

    # ── Cross-asset résumé ────────────────────────────────────────────────────
    "cross_asset_agreement_10s",
    "cross_asset_agreement_60s",
    "cross_asset_momentum_10s",
    "cross_asset_momentum_60s",

    # ── Binance order book ───────────────────────────────────────────────────
    "bn_best_bid",          # Binance best bid price
    "bn_best_ask",          # Binance best ask price
    "bn_mid",               # Binance mid price
    "bn_spread_bps",        # Binance spread in bps
    "bn_bid_depth5",        # total qty bid top 5 (base asset)
    "bn_ask_depth5",        # total qty ask top 5
    "bn_depth_imbalance",   # (bid-ask)/(bid+ask)
    "bn_bid_l2_price", "bn_bid_l2_qty",
    "bn_bid_l3_price", "bn_bid_l3_qty",
    "bn_bid_l4_price", "bn_bid_l4_qty",
    "bn_bid_l5_price", "bn_bid_l5_qty",
    "bn_ask_l2_price", "bn_ask_l2_qty",
    "bn_ask_l3_price", "bn_ask_l3_qty",
    "bn_ask_l4_price", "bn_ask_l4_qty",
    "bn_ask_l5_price", "bn_ask_l5_qty",

    # ── DOWN token book ──────────────────────────────────────────────────────
    "down_bid",             # meilleur bid token DOWN ∈ [0,1]
    "down_ask",             # meilleur ask token DOWN ∈ [0,1]
    "down_mid",             # mid token DOWN
    "down_spread_bps",      # spread token DOWN en bps
    "down_best_bid_size",   # USDC dispo au meilleur bid DOWN
    "down_best_ask_size",   # USDC dispo au meilleur ask DOWN
    "down_book_bids",  # JSON : "[[0.45,100],[0.44,200],...]" — book complet bids
    "down_book_asks",  # JSON : "[[0.46,80],[0.47,150],...]" — book complet asks
    "down_bid_depth_top5",
    "down_ask_depth_top5",
    "down_depth_imbalance",

    # ── Outcome ───────────────────────────────────────────────────────────────
    "outcome",
    "coin_price_at_window_close",
]

assert len(EXPECTED_HEADER) == len(set(EXPECTED_HEADER)), \
    f"Colonnes dupliquées : {[c for c in EXPECTED_HEADER if EXPECTED_HEADER.count(c) > 1]}"

N_COLS = len(EXPECTED_HEADER)

# Helper for backtesting — how to simulate order execution
TAKER_FEE_BPS = 2.0   # Polymarket taker fee

if __name__ == "__main__":
    print(f"Total colonnes : {N_COLS}")
    for i, col in enumerate(EXPECTED_HEADER, 1):
        new = "★" if any(x in col for x in ["best_bid_size","best_ask_size","_l2_","_l3_","_l4_","_l5_"]) else " "
        print(f"  {new} {i:3d}. {col}")
