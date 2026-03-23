from __future__ import annotations

# ──────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────
PARQUET_OUTPUT_DIR  = "historical_data"
PARQUET_FLUSH_SECONDS = 120   # compacte le journal en Parquet toutes les 2 minutes
DEBUG_TAIL_CSV      = "debug_tail.csv"
DEBUG_TAIL_MAX_ROWS = 500
TIMESCALE_DSN       = ""
TIMESCALE_TABLE     = "market_ticks"

# ──────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 1.0
LOG_INTERVAL_SECONDS  = 5.0    # 0.2 Hz — event store captures every tick anyway

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
GAMMA_BASE   = "https://gamma-api.polymarket.com"
CLOB_BASE    = "https://clob.polymarket.com"
BINANCE_BASE = "https://api.binance.com"

POLYMARKET_WSS_URL    = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_MULTI_WSS_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams=btcusdt@trade/ethusdt@trade/solusdt@trade/xrpusdt@trade"
    "/btcusdt@depth20@1000ms/ethusdt@depth20@1000ms"
    "/solusdt@depth20@1000ms/xrpusdt@depth20@1000ms"
)
REQUEST_TIMEOUT = 10

# ──────────────────────────────────────────────
# Polymarket markets to track
# Each entry: (poly_coin, binance_symbol, interval_minutes)
# The logger will try to find an active market for each combo.
# ──────────────────────────────────────────────
POLY_MARKETS = [
    # coin      binance_sym   interval_min
    ("btc",    "BTCUSDT",    5),
    ("btc",    "BTCUSDT",    15),
    ("eth",    "ETHUSDT",    5),
    ("eth",    "ETHUSDT",    15),
    ("sol",    "SOLUSDT",    5),
    ("sol",    "SOLUSDT",    15),
    ("xrp",    "XRPUSDT",    5),
    ("xrp",    "XRPUSDT",    15),
]

# ──────────────────────────────────────────────
# Binance assets (for WebSocket + vol tracking)
# ──────────────────────────────────────────────
TRACKED_ASSETS = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

# ──────────────────────────────────────────────
# WebSocket health thresholds (ms)
# ──────────────────────────────────────────────
POLY_STALE_MS  = 5_000
POLY_DEAD_MS   = 20_000
ASSET_STALE_MS = 1_500
ASSET_DEAD_MS  = 5_000
ASSET_PRICE_MAX_STALENESS_MS = 1_500

# ──────────────────────────────────────────────
# In-memory history
# ──────────────────────────────────────────────
ASSET_BUCKET_HISTORY_SECONDS   = 3600
ASSET_TRADE_ID_HISTORY_SECONDS = 7200
POLY_TRADE_DEQUE_MAXLEN        = 10_000

# ──────────────────────────────────────────────
# Market / cache
# ──────────────────────────────────────────────
MARKET_REFRESH_SECONDS   = 10.0
TARGET_CACHE_TTL_SECONDS = 3 * 3600

# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────
DASHBOARD_HOST      = "0.0.0.0"
DASHBOARD_PORT      = 8765
DASHBOARD_HTTP_PORT = 8080

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL        = "INFO"
LOG_FILE         = "logger.log"
LOG_MAX_BYTES    = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
