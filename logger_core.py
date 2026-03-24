from __future__ import annotations

import logging
import math
import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

from api import ApiClient
from config import (
    ASSET_PRICE_MAX_STALENESS_MS,
    LOG_INTERVAL_SECONDS,
    MARKET_REFRESH_SECONDS,
    POLL_INTERVAL_SECONDS,
    TARGET_CACHE_TTL_SECONDS,
    TRACKED_ASSETS,
)
from storage import StorageManager
from utils import (
    analytical_prob_up,
    compute_time_window_metrics,
    cross_asset_agreement,
    cross_asset_momentum,
    floor_to_interval_seconds,
    fmt_bool,
    fmt_float,
    fmt_int,
    iso_utc,
    market_window_from_slug,
    mid_velocity_bps_per_second,
    normalize_text,
    prune_asset_target_cache,
    prune_market_target_cache,
    sanitize_csv_field,
    utc_now,
    z_score_binary,
)
from ws_clients import MultiAssetBinanceTradeWS, PolymarketBookWS

logger = logging.getLogger(__name__)

_PolyMidEntry = tuple[datetime, float]


def build_empty_poly_metrics() -> dict[str, Any]:
    return {
        "poly_trade_count_10s": 0, "poly_trade_count_30s": 0, "poly_trade_count_60s": 0,
        "poly_trade_size_10s": 0.0, "poly_trade_size_30s": 0.0, "poly_trade_size_60s": 0.0,
        "poly_ws_last_trade_utc": "", "poly_ws_lag_ms": None,
        "poly_ws_status": "stopped", "poly_ws_health": "dead",
    }


def asset_price_decimals(asset: str) -> int:
    if asset == "xrp":         return 6
    if asset in {"sol","eth"}: return 4
    return 2


def build_asset_row_values(
    asset: str, symbol: str, client: ApiClient,
    multi_ws: MultiAssetBinanceTradeWS, snap_dt: datetime,
    asset_target_cache: dict[str, Optional[float]],
) -> tuple[list[Any], dict[str, Any]]:

    price = multi_ws.get_last_price(asset, snap_dt, ASSET_PRICE_MAX_STALENESS_MS)
    if price is None:
        price = client.get_spot_price(symbol)

    a15_floor = floor_to_interval_seconds(snap_dt, 15 * 60)
    a1h_floor  = floor_to_interval_seconds(snap_dt, 60 * 60)

    (a15_start, a15_end, s15, p15) = compute_time_window_metrics(snap_dt, a15_floor, 15 * 60)
    (a1h_start, a1h_end, s1h, p1h) = compute_time_window_metrics(snap_dt, a1h_floor, 60 * 60)

    k15 = f"{asset}:15m:{int(a15_start.timestamp())}"
    k1h = f"{asset}:1h:{int(a1h_start.timestamp())}"
    if k15 not in asset_target_cache:
        asset_target_cache[k15] = client.get_symbol_target_price_at_window_start(symbol, a15_start)
    if k1h not in asset_target_cache:
        asset_target_cache[k1h] = client.get_symbol_target_price_at_window_start(symbol, a1h_start)

    t15, t1h = asset_target_cache[k15], asset_target_cache[k1h]

    def dist(p, t):
        if p is None or t is None: return None, None
        d = p - t
        return d, (d / t * 10000.0 if t != 0 else None)

    d15, d15b = dist(price, t15)
    d1h, d1hb = dist(price, t1h)

    m = multi_ws.get_metrics(asset, snap_dt)
    dec = asset_price_decimals(asset)

    row = [
        fmt_float(price, dec),
        iso_utc(a15_start), iso_utc(a15_end),
        fmt_float(s15, 3), fmt_float(p15, 3),
        fmt_float(t15, dec), fmt_float(d15, dec), fmt_float(d15b, 4),
        iso_utc(a1h_start), iso_utc(a1h_end),
        fmt_float(s1h, 3), fmt_float(p1h, 3),
        fmt_float(t1h, dec), fmt_float(d1h, dec), fmt_float(d1hb, 4),
        fmt_float(m["return_10s_bps"], 4), fmt_float(m["return_30s_bps"], 4),
        fmt_float(m["return_60s_bps"], 4), fmt_float(m["return_15m_bps"], 4),
        fmt_float(m["return_1h_bps"], 4),
        m["trade_count_10s"], m["trade_count_30s"], m["trade_count_60s"],
        m["trade_count_15m"], m["trade_count_1h"],
        fmt_float(m["volume_asset_10s"], 6), fmt_float(m["volume_asset_30s"], 6),
        fmt_float(m["volume_asset_60s"], 6), fmt_float(m["volume_asset_15m"], 6),
        fmt_float(m["volume_asset_1h"], 6),
        fmt_float(m["volume_usdt_10s"], 2), fmt_float(m["volume_usdt_30s"], 2),
        fmt_float(m["volume_usdt_60s"], 2), fmt_float(m["volume_usdt_15m"], 2),
        fmt_float(m["volume_usdt_1h"], 2),
        fmt_float(m["buy_volume_asset_10s"], 6), fmt_float(m["buy_volume_asset_30s"], 6),
        fmt_float(m["buy_volume_asset_60s"], 6), fmt_float(m["buy_volume_asset_15m"], 6),
        fmt_float(m["buy_volume_asset_1h"], 6),
        fmt_float(m["sell_volume_asset_10s"], 6), fmt_float(m["sell_volume_asset_30s"], 6),
        fmt_float(m["sell_volume_asset_60s"], 6), fmt_float(m["sell_volume_asset_15m"], 6),
        fmt_float(m["sell_volume_asset_1h"], 6),
        fmt_float(m["buy_volume_usdt_10s"], 2), fmt_float(m["buy_volume_usdt_30s"], 2),
        fmt_float(m["buy_volume_usdt_60s"], 2), fmt_float(m["buy_volume_usdt_15m"], 2),
        fmt_float(m["buy_volume_usdt_1h"], 2),
        fmt_float(m["sell_volume_usdt_10s"], 2), fmt_float(m["sell_volume_usdt_30s"], 2),
        fmt_float(m["sell_volume_usdt_60s"], 2), fmt_float(m["sell_volume_usdt_15m"], 2),
        fmt_float(m["sell_volume_usdt_1h"], 2),
        fmt_float(m["buy_ratio_usdt_10s"], 6), fmt_float(m["buy_ratio_usdt_30s"], 6),
        fmt_float(m["buy_ratio_usdt_60s"], 6), fmt_float(m["buy_ratio_usdt_15m"], 6),
        fmt_float(m["buy_ratio_usdt_1h"], 6),
        fmt_float(m["flow_imbalance_usdt_10s"], 6), fmt_float(m["flow_imbalance_usdt_30s"], 6),
        fmt_float(m["flow_imbalance_usdt_60s"], 6), fmt_float(m["flow_imbalance_usdt_15m"], 6),
        fmt_float(m["flow_imbalance_usdt_1h"], 6),
        fmt_float(m["vwap_10s"], dec), fmt_float(m["vwap_30s"], dec),
        fmt_float(m["vwap_60s"], dec), fmt_float(m["vwap_15m"], dec),
        fmt_float(m["vwap_1h"], dec),
        m["ws_last_trade_utc"], fmt_int(m["ws_lag_ms"]),
        m["ws_status"], m["ws_health"],
    ]
    return row, {"price": price, "metrics": m, "target_15m": t15, "target_1h": t1h}


class _MarketFetchResult:
    __slots__ = ("market", "window_start", "window_end")
    def __init__(self, market, window_start, window_end):
        self.market = market; self.window_start = window_start; self.window_end = window_end


class BackgroundMarketFetcher:
    """Fetches one Polymarket updown market (coin+interval) in background."""
    def __init__(self, client: ApiClient, coin: str, interval_min: int) -> None:
        self._client       = client
        self._coin         = coin
        self._interval_min = interval_min
        self._result_q: queue.Queue[_MarketFetchResult] = queue.Queue(maxsize=1)
        self._req_event  = threading.Event()
        self._stop_event = threading.Event()
        name = f"mkt-{coin}-{interval_min}m"
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)

    def start(self) -> None:   self._thread.start()
    def stop(self) -> None:    self._stop_event.set(); self._req_event.set()
    def request_fetch(self) -> None: self._req_event.set()

    def pop_result(self) -> Optional[_MarketFetchResult]:
        try:    return self._result_q.get_nowait()
        except queue.Empty: return None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._req_event.wait(timeout=1.5)
            self._req_event.clear()
            if self._stop_event.is_set(): break
            try:
                now_dt = utc_now()
                market = self._client.find_active_market(self._coin, self._interval_min, now_dt)
                if market is not None:
                    slug = normalize_text(market.get("slug"))
                    ws, we = market_window_from_slug(slug)
                else:
                    ws = we = None
                result = _MarketFetchResult(market, ws, we)
            except Exception as exc:
                logger.debug("BackgroundMarketFetcher[%s-%dm]: %s",
                             self._coin, self._interval_min, exc)
                result = _MarketFetchResult(None, None, None)
            try:    self._result_q.get_nowait()
            except queue.Empty: pass
            try:    self._result_q.put_nowait(result)
            except queue.Full: pass


class BackgroundBookFetcher:
    """Fetches Polymarket order book in a background thread so it never blocks the main loop."""
    def __init__(self, client: Any) -> None:
        self._client = client
        self._result_q: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._req_event = threading.Event()
        self._stop_event = threading.Event()
        self._last_token_id: Optional[str] = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="book-fetcher")

    def start(self) -> None:   self._thread.start()
    def stop(self) -> None:    self._stop_event.set(); self._req_event.set()

    def request_fetch(self, token_id: str) -> None:
        self._last_token_id = token_id
        self._req_event.set()

    def pop_result(self) -> Optional[dict]:
        try:    return self._result_q.get_nowait()
        except queue.Empty: return None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._req_event.wait(timeout=1.0)
            self._req_event.clear()
            if self._stop_event.is_set(): break
            token_id = self._last_token_id
            if not token_id: continue
            try:
                book = self._client.get_up_book_metrics(token_id)
            except Exception as exc:
                logger.debug("BookFetcher error: %s", exc)
                book = {k: None for k in ["up_bid","up_ask","up_mid","up_spread","up_spread_bps_mid",
                                           "up_best_bid_size","up_best_ask_size",
                                           "up_bid_depth_top5","up_ask_depth_top5","up_depth_imbalance_top5"]}
            try:    self._result_q.get_nowait()
            except queue.Empty: pass
            try:    self._result_q.put_nowait(book)
            except queue.Full: pass


class PolyMidTracker:
    def __init__(self, max_history_seconds: int = 60) -> None:
        self._history: deque[_PolyMidEntry] = deque()
        self._max_seconds = max_history_seconds
        self._last_velocity: Optional[float] = None

    def push(self, now_dt: datetime, mid: Optional[float]) -> None:
        if mid is None: return
        self._history.append((now_dt, mid))
        cutoff = now_dt.timestamp() - self._max_seconds
        while self._history and self._history[0][0].timestamp() < cutoff:
            self._history.popleft()

    def mid_n_seconds_ago(self, now_dt: datetime, n: float) -> Optional[float]:
        target_ts = now_dt.timestamp() - n
        best: Optional[tuple[float, float]] = None
        for dt, val in self._history:
            delta = abs(dt.timestamp() - target_ts)
            if best is None or delta < best[0]:
                best = (delta, val)
        if best is None or best[0] > n * 0.5:
            return None
        return best[1]

    def velocity_bps_per_second(self, now_dt: datetime, mid_now: Optional[float]) -> Optional[float]:
        mid_5s = self.mid_n_seconds_ago(now_dt, 5.0)
        return mid_velocity_bps_per_second(mid_now, mid_5s, 5.0)

    def acceleration(self, now_dt: datetime, current_velocity: Optional[float]) -> Optional[float]:
        if current_velocity is None or self._last_velocity is None:
            self._last_velocity = current_velocity
            return None
        accel = current_velocity - self._last_velocity
        self._last_velocity = current_velocity
        return accel

    def reset(self) -> None:
        self._history.clear()
        self._last_velocity = None


# ─────────────────────────────────────────────────────────────────────────────
# MarketTracker — one instance per (coin, interval)
# ─────────────────────────────────────────────────────────────────────────────

class MarketTracker:
    """
    Encapsulates all state for one Polymarket updown market (coin + interval).
    e.g. BTC-5m, ETH-15m, SOL-5m ...
    """
    def __init__(self, coin: str, symbol: str, interval_min: int,
                 client: ApiClient, event_store: Any = None) -> None:
        self.coin         = coin          # "btc", "eth", "sol"
        self.symbol       = symbol        # "BTCUSDT" etc
        self.interval_min = interval_min
        self.key          = f"{coin}-{interval_min}m"
        self._event_store = event_store   # for recording market metadata + outcomes

        self.fetcher      = BackgroundMarketFetcher(client, coin, interval_min)
        self.poly_ws = PolymarketBookWS()  # UP + DOWN tokens in one WS
        self.mid_tracker  = PolyMidTracker()

        self.cached_market:       Optional[dict] = None
        self.cached_window_start: Optional[datetime] = None
        self.cached_window_end:   Optional[datetime] = None
        self.cached_book: dict = {k: None for k in [
            "up_bid","up_ask","up_mid","up_spread","up_spread_bps_mid",
            "up_best_bid_size","up_best_ask_size",
            "up_bid_depth_top5","up_ask_depth_top5","up_depth_imbalance_top5"]}

        self.last_slug:       Optional[str]      = None
        self.last_up_token:   Optional[str]      = None
        self.last_down_token: Optional[str]      = None
        self.last_window_end: Optional[datetime] = None
        self.target_cache:    dict[str, Optional[float]] = {}

        self._next_refresh_mono = 0.0

    def start(self) -> None:
        self.fetcher.start()
        # poly_ws started on demand when up_token_id is known

    def stop(self) -> None:
        self.fetcher.stop()
        self.poly_ws.stop()

    def tick(self, now_dt: datetime, now_mono: float,
             client: ApiClient, snap_dt: datetime) -> dict:
        """
        Call every poll tick. Returns a dict with all market-specific signals.
        Never blocks — all network calls are in background threads.
        """
        # ── Refresh market ────────────────────────────────────────────────
        should_fetch = (
            self.cached_market is None
            or (self.cached_window_end and now_dt >= self.cached_window_end)
            or now_mono >= self._next_refresh_mono
        )
        if should_fetch:
            self.fetcher.request_fetch()
            self._next_refresh_mono = now_mono + MARKET_REFRESH_SECONDS

        fr = self.fetcher.pop_result()
        if fr is not None:
            self.cached_market       = fr.market
            self.cached_window_start = fr.window_start
            self.cached_window_end   = fr.window_end

        market_found = self.cached_market is not None
        market_slug = up_token_id = ""
        window_start_str = window_end_str = ""
        seconds_remaining = window_progress = None
        target_price = distance_bps = None
        outcome_value = ""
        btc_close_px = None
        just_switched = False

        # ── Detect window close ───────────────────────────────────────────
        if (self.last_window_end and now_dt >= self.last_window_end
                and self.last_slug and self.last_slug in self.target_cache):
            close_px  = client.get_symbol_target_price_at_window_start(self.symbol, self.last_window_end)
            target_px = self.target_cache.get(self.last_slug)
            if close_px is not None and target_px is not None:
                btc_close_px  = close_px
                outcome_value = "1" if close_px > target_px else "0"
                logger.info("Window closed [%s] close=%.4f target=%.4f → %s",
                            self.key, close_px, target_px, "UP" if outcome_value=="1" else "DOWN")
                # ★ Record outcome in event store
                if self._event_store is not None:
                    try:
                        self._event_store.poly_markets.append(
                            ts_us=int(now_dt.timestamp() * 1_000_000),
                            slug=self.last_slug,
                            coin=self.coin,
                            interval_min=self.interval_min,
                            up_token_id=self.last_up_token or "",
                            down_token_id=self.last_down_token or "",
                            window_start_us=0,
                            window_end_us=0,
                            target_price=target_px,
                            close_price=close_px,
                            outcome=1 if outcome_value == "1" else 0,
                        )
                    except Exception as e:
                        logger.debug("Outcome recording error: %s", e)
            self.last_window_end = None

        down_token_id = ""  # initialized before if/else
        if market_found:
            market_slug  = normalize_text(self.cached_market.get("slug"))
            up_token_id  = client.get_up_token_id(self.cached_market) or ""
            window_start_str = iso_utc(self.cached_window_start) if self.cached_window_start else ""
            window_end_str   = iso_utc(self.cached_window_end)   if self.cached_window_end   else ""

            just_switched = market_slug != self.last_slug
            if just_switched:
                logger.info("New market [%s]: %s %s → %s",
                            self.key, market_slug, window_start_str, window_end_str)
                self.last_slug       = market_slug
                self.last_window_end = self.cached_window_end
                self.mid_tracker.reset()

            down_token_id = client.get_down_token_id(self.cached_market) or ""

            if up_token_id != self.last_up_token or down_token_id != self.last_down_token:
                if up_token_id:
                    self.poly_ws.start(up_token_id, down_asset_id=down_token_id or None)
                else:
                    self.poly_ws.stop()
                self.cached_book = {}
                self.last_up_token   = up_token_id
                self.last_down_token = down_token_id

            if market_slug not in self.target_cache and self.cached_window_start:
                self.target_cache[market_slug] = client.get_symbol_target_price_at_window_start(
                    self.symbol, self.cached_window_start)

            target_price = self.target_cache.get(market_slug)

            # ★ Record market metadata on switch (slug, tokens, target, window)
            if just_switched and self._event_store is not None:
                try:
                    ws_us = int(self.cached_window_start.timestamp() * 1_000_000) if self.cached_window_start else 0
                    we_us = int(self.cached_window_end.timestamp() * 1_000_000) if self.cached_window_end else 0
                    self._event_store.poly_markets.append(
                        ts_us=int(now_dt.timestamp() * 1_000_000),
                        slug=market_slug,
                        coin=self.coin,
                        interval_min=self.interval_min,
                        up_token_id=up_token_id,
                        down_token_id=down_token_id,
                        window_start_us=ws_us,
                        window_end_us=we_us,
                        target_price=target_price or 0.0,
                        close_price=0.0,
                        outcome=-1,    # unknown at switch time
                    )
                except Exception as e:
                    logger.debug("Market metadata recording error: %s", e)

            # ── Book (background) ────────────────────────────────────────
            # Book is now updated via WebSocket in PolymarketBookWS — no REST call needed

            # ── Window timing ────────────────────────────────────────────
            if self.cached_window_start and self.cached_window_end:
                total_s = (self.cached_window_end - self.cached_window_start).total_seconds()
                rem_s   = max(0.0, (self.cached_window_end - snap_dt).total_seconds())
                seconds_remaining = rem_s
                if total_s > 0:
                    window_progress = max(0.0, min(100.0, (total_s - rem_s) / total_s * 100))

        else:
            if self.last_up_token:
                self.poly_ws.stop()
                self.last_up_token = None
                self.last_down_token = None
            down_token_id = ""

        # ── Prune old target cache entries ────────────────────────────────
        prune_market_target_cache(self.target_cache, now_dt, TARGET_CACHE_TTL_SECONDS)

        # ── Mid dynamics ──────────────────────────────────────────────────
        # Get fresh book data from WS (10-50ms latency vs 300ms REST before)
        book = self.poly_ws.get_book_metrics(snap_dt) if up_token_id and self.poly_ws.connected else self.cached_book
        if book:
            self.cached_book = book

        # DOWN book — same WS as UP, separate book state
        down_book = self.poly_ws.get_down_book_metrics(snap_dt) if down_token_id and self.poly_ws.connected else {}
        up_mid_now = book.get("up_mid")
        self.mid_tracker.push(snap_dt, up_mid_now)
        mid_5s_ago   = self.mid_tracker.mid_n_seconds_ago(snap_dt, 5.0)
        mid_10s_ago  = self.mid_tracker.mid_n_seconds_ago(snap_dt, 10.0)
        velocity     = self.mid_tracker.velocity_bps_per_second(snap_dt, up_mid_now)
        acceleration = self.mid_tracker.acceleration(snap_dt, velocity)

        return {
            "key":               self.key,
            "coin":              self.coin,
            "interval_min":      self.interval_min,
            "market_found":      market_found,
            "just_switched":     just_switched,
            "market_slug":       market_slug,
            "window_start_str":  window_start_str,
            "window_end_str":    window_end_str,
            "seconds_remaining": seconds_remaining,
            "window_progress":   window_progress,
            "target_price":      target_price,
            "book":              book,
            "down_book":         down_book,
            "up_token_id":       up_token_id,
            "poly_metrics":      (self.poly_ws.get_trade_metrics(snap_dt)
                                  if market_found and up_token_id
                                  else build_empty_poly_metrics()),
            "mid_5s_ago":        mid_5s_ago,
            "mid_10s_ago":       mid_10s_ago,
            "poly_mid_velocity": velocity,
            "poly_mid_accel":    acceleration,
            "outcome":           outcome_value,
            "btc_close_px":      btc_close_px,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_logger(dashboard: Any = None, event_store: Any = None) -> None:
    from config import POLY_MARKETS
    storage  = StorageManager()
    client   = ApiClient()
    multi_ws = MultiAssetBinanceTradeWS(TRACKED_ASSETS)

    # ★ Attach event recording for backtesting
    if event_store is not None:
        from event_recording import attach_binance_recorder, attach_poly_recorder
        attach_binance_recorder(multi_ws, event_store)

    # One MarketTracker per (coin, interval)
    trackers: list[MarketTracker] = [
        MarketTracker(coin, symbol, interval_min, client, event_store=event_store)
        for coin, symbol, interval_min in POLY_MARKETS
    ]

    # ★ Attach Polymarket event recording to each tracker's WS
    if event_store is not None:
        for tr in trackers:
            attach_poly_recorder(tr.poly_ws, event_store)

    asset_target_cache: dict[str, Optional[float]] = {}

    # ── Paper trader ──────────────────────────────────────────────────────
    from paper_trader import PaperTrader
    _paper = PaperTrader()
    _btc_15m_tracker = next((tr for tr in trackers if tr.key == "btc-15m"), None)

    next_tick_mono = time.monotonic()
    next_log_mono  = time.monotonic()
    next_storage_report_mono = time.monotonic() + 3600 * 6
    _next_mem_diag_mono = time.monotonic() + 60  # first diag after 1 min

    # ── Shutdown ───────────────────────────────────────────────────────────
    shutdown_flag = threading.Event()
    def _shutdown(signum: int, frame: Any) -> None:
        logger.info("Signal %d → graceful shutdown", signum)
        shutdown_flag.set()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    multi_ws.start()
    for tr in trackers:
        tr.start()

    logger.info("Logger started | poll=%.2fs | log_interval=%.1fs | markets=%s",
                POLL_INTERVAL_SECONDS, LOG_INTERVAL_SECONDS,
                ", ".join(f"{tr.key}" for tr in trackers))

    # ── Watchdog — restarts Binance WS if silent > 90s ────────────────────
    _last_healthy_mono = [time.monotonic()]

    def _watchdog() -> None:
        while not shutdown_flag.is_set():
            time.sleep(30)
            age = time.monotonic() - _last_healthy_mono[0]
            if age > 90:
                logger.warning("Watchdog: Binance WS silent for %.0fs — restarting", age)
                try:
                    multi_ws.stop()
                    time.sleep(2)
                    multi_ws.start()
                    _last_healthy_mono[0] = time.monotonic()
                    logger.info("Watchdog: Binance WS restarted OK")
                except Exception as exc:
                    logger.error("Watchdog restart failed: %s", exc)

    threading.Thread(target=_watchdog, daemon=True, name="watchdog").start()

    try:
        while not shutdown_flag.is_set():
            loop_started_dt  = utc_now()
            snap_str = iso_utc(loop_started_dt)

            try:
                now_dt   = utc_now()
                now_mono = time.monotonic()

                prune_asset_target_cache(asset_target_cache, now_dt, TARGET_CACHE_TTL_SECONDS)

                # ── Storage size report ────────────────────────────────────
                if now_mono >= next_storage_report_mono:
                    report = storage.stats()
                    if report:
                        logger.info("Storage: %s", report)
                    next_storage_report_mono = now_mono + 3600 * 6

                # ── Memory diagnostics (every 10 min) ─────────────────────
                if now_mono >= _next_mem_diag_mono:
                    import os as _os
                    try:
                        _rss_kb = 0
                        with open(f"/proc/{_os.getpid()}/status") as _f:
                            for _line in _f:
                                if _line.startswith("VmRSS:"):
                                    _rss_kb = int(_line.split()[1])
                                    break
                        _rss_mb = _rss_kb // 1024
                    except Exception:
                        _rss_mb = -1
                    _bn_diag = {}
                    with multi_ws.lock:
                        for _asset, _st in multi_ws.state.items():
                            _bn_diag[_asset] = {
                                "buckets": len(_st["second_buckets"]),
                                "trade_ids": len(_st["recent_trade_ids"]),
                                "trade_id_set": len(_st["recent_trade_id_set"]),
                            }
                    _poly_diag = {}
                    for _tr in trackers:
                        with _tr.poly_ws._lock:
                            _poly_diag[_tr.key] = len(_tr.poly_ws.trade_events)
                    _n_threads = threading.active_count()
                    logger.info(
                        "MEMDIAG rss=%dMB threads=%d bn=%s poly_trades=%s",
                        _rss_mb, _n_threads, _bn_diag, _poly_diag,
                    )
                    _next_mem_diag_mono = now_mono + 600  # every 10 min

                snap_dt = utc_now()

                # ── Asset metrics (shared across all markets) ──────────────
                # Single lock acquisition for all Binance metrics
                bn_snap = multi_ws.snapshot_all(snap_dt)

                def _mk_dbg(asset):
                    s = bn_snap.get(asset, {})
                    return {"price": s.get("price"), "metrics": s}

                btc_dbg = _mk_dbg("btc")
                eth_dbg = _mk_dbg("eth")
                sol_dbg = _mk_dbg("sol")
                xrp_dbg = _mk_dbg("xrp")

                btc_price   = btc_dbg["price"]
                # Reset watchdog timer when we get a live BTC price
                if btc_price is not None:
                    _last_healthy_mono[0] = time.monotonic()
                btc_vol_10s = bn_snap.get("btc", {}).get("vol_10s_bps")
                btc_vol_60s = bn_snap.get("btc", {}).get("vol_60s_bps")
                btc_vol_15m = bn_snap.get("btc", {}).get("vol_15m_bps")
                eth_vol_60s = bn_snap.get("eth", {}).get("vol_60s_bps")
                sol_vol_60s = bn_snap.get("sol", {}).get("vol_60s_bps")
                xrp_vol_60s = bn_snap.get("xrp", {}).get("vol_60s_bps")

                btc_r10 = btc_dbg["metrics"]["return_10s_bps"]
                btc_r60 = btc_dbg["metrics"]["return_60s_bps"]
                eth_r10 = eth_dbg["metrics"]["return_10s_bps"]
                sol_r10 = sol_dbg["metrics"]["return_10s_bps"]
                xrp_r10 = xrp_dbg["metrics"]["return_10s_bps"]
                agree_10s = cross_asset_agreement(btc_r10, [eth_r10, sol_r10, xrp_r10])
                agree_60s = cross_asset_agreement(btc_r60, [eth_dbg["metrics"]["return_60s_bps"],
                                                             sol_dbg["metrics"]["return_60s_bps"],
                                                             xrp_dbg["metrics"]["return_60s_bps"]])
                mom_10s = cross_asset_momentum([eth_r10, sol_r10, xrp_r10])
                mom_60s = cross_asset_momentum([eth_dbg["metrics"]["return_60s_bps"],
                                                sol_dbg["metrics"]["return_60s_bps"],
                                                xrp_dbg["metrics"]["return_60s_bps"]])

                # ── Per-market ticks ──────────────────────────────────────
                asset_dbg = {
                    "btc": btc_dbg,
                    "eth": eth_dbg,
                    "sol": sol_dbg,
                    "xrp": xrp_dbg,
                }
                asset_vol60 = {
                    "btc": btc_vol_60s,
                    "eth": eth_vol_60s,
                    "sol": sol_vol_60s,
                    "xrp": xrp_vol_60s,
                }
                market_snaps: dict[str, dict] = {}

                for tr in trackers:
                    coin_dbg     = asset_dbg[tr.coin]
                    coin_vol_60s = asset_vol60[tr.coin]
                    coin_price   = coin_dbg["price"]

                    t = tr.tick(now_dt, now_mono, client, snap_dt)

                    target_price = t["target_price"]
                    distance_bps = None
                    if coin_price is not None and target_price is not None and target_price != 0:
                        distance_bps = (coin_price - target_price) / target_price * 10000.0

                    z   = z_score_binary(distance_bps, coin_vol_60s, t["seconds_remaining"])
                    pup = analytical_prob_up(z)

                    book = t["book"]
                    poly = t["poly_metrics"]

                    market_snaps[tr.key] = {
                        "key":              tr.key,
                        "coin":             tr.coin,
                        "interval_min":     tr.interval_min,
                        "market_found":     t["market_found"],
                        "slug":             t["market_slug"],
                        "window_start":     t["window_start_str"],
                        "window_end":       t["window_end_str"],
                        "seconds_remaining": t["seconds_remaining"],
                        "window_progress":  t["window_progress"],
                        "coin_price":       coin_price,
                        "target_price":     target_price,
                        "distance_bps":     distance_bps,
                        "coin_vol_60s_bps": coin_vol_60s,
                        "z_score":          z,
                        "analytical_prob_up": pup,
                        "up_mid":              book.get("up_mid"),
                        "up_bid":              book.get("up_bid"),
                        "up_ask":              book.get("up_ask"),
                        "up_spread_bps":       book.get("up_spread_bps") or book.get("up_spread_bps_mid"),
                        "up_best_bid_size":    book.get("up_best_bid_size"),
                        "up_best_ask_size":    book.get("up_best_ask_size"),
                        "up_book_bids":        book.get("up_book_bids"),
                        "up_book_asks":        book.get("up_book_asks"),
                        "up_bid_depth_top5":   book.get("up_bid_depth_top5"),
                        "up_ask_depth_top5":   book.get("up_ask_depth_top5"),
                        "up_depth_imbalance":  book.get("up_depth_imbalance") or book.get("up_depth_imbalance_top5"),
                        "poly_trade_count_10s": poly["poly_trade_count_10s"],
                        "poly_trade_count_60s": poly["poly_trade_count_60s"],
                        "poly_trade_size_10s":  poly["poly_trade_size_10s"],
                        "poly_trade_size_60s":  poly["poly_trade_size_60s"],
                        "poly_mid_5s_ago":      t["mid_5s_ago"],
                        "poly_mid_10s_ago":     t["mid_10s_ago"],
                        "poly_mid_velocity":    t["poly_mid_velocity"],
                        "poly_mid_accel":       t["poly_mid_accel"],
                        "poly_health":          poly["poly_ws_health"],
                        "poly_lag_ms":          poly["poly_ws_lag_ms"],
                        # DOWN book
                        "down_bid":             t["down_book"].get("down_bid"),
                        "down_ask":             t["down_book"].get("down_ask"),
                        "down_mid":             t["down_book"].get("down_mid"),
                        "down_spread_bps":      t["down_book"].get("down_spread_bps"),
                        "down_best_bid_size":   t["down_book"].get("down_best_bid_size"),
                        "down_best_ask_size":   t["down_book"].get("down_best_ask_size"),
                        "down_book_bids":       t["down_book"].get("down_book_bids"),
                        "down_book_asks":       t["down_book"].get("down_book_asks"),
                        "down_bid_depth_top5":  t["down_book"].get("down_bid_depth_top5"),
                        "down_ask_depth_top5":  t["down_book"].get("down_ask_depth_top5"),
                        "down_depth_imbalance": t["down_book"].get("down_depth_imbalance"),
                        "outcome":              t["outcome"],
                        "coin_close_px":        t["btc_close_px"],
                    }

                    # ── Log to console ──────────────────────────────────────
                    if t["market_found"]:
                        logger.info(
                            "[%s] price=%.4f dist=%.1f z=%s p_up=%s vol60=%.2f | "
                            "up_mid=%s | poly=%s",
                            tr.key,
                            coin_price or 0,
                            distance_bps or 0,
                            f"{z:.2f}" if z is not None else "NA",
                            f"{pup:.3f}" if pup is not None else "NA",
                            coin_vol_60s or 0,
                            f"{book.get('up_mid'):.4f}" if book.get("up_mid") is not None else "NA",
                            poly["poly_ws_health"],
                        )

                # ── Paper trader tick ─────────────────────────────────────
                if _btc_15m_tracker is not None:
                    try:
                        _paper.tick(market_snaps, _btc_15m_tracker.poly_ws, snap_dt)
                    except Exception as _pe:
                        logger.debug("PaperTrader error: %s", _pe)

                # ── Dashboard snapshot ────────────────────────────────────
                if dashboard is not None:
                    dashboard.push_snapshot({
                        "timestamp_utc":       snap_str,
                        # Per-coin fields
                        "btc_price":           btc_dbg["price"],
                        "btc_return_10s_bps":  btc_dbg["metrics"]["return_10s_bps"],
                        "btc_return_60s_bps":  btc_dbg["metrics"]["return_60s_bps"],
                        "btc_return_15m_bps":  btc_dbg["metrics"]["return_15m_bps"],
                        "btc_flow_imbalance_10s": btc_dbg["metrics"]["flow_imbalance_usdt_10s"],
                        "btc_flow_imbalance_60s": btc_dbg["metrics"]["flow_imbalance_usdt_60s"],
                        "btc_buy_ratio_usdt_60s": btc_dbg["metrics"]["buy_ratio_usdt_60s"],
                        "btc_volume_usdt_60s": btc_dbg["metrics"]["volume_usdt_60s"],
                        "btc_vol_60s_bps":     btc_vol_60s,
                        "eth_price":           eth_dbg["price"],
                        "eth_return_10s_bps":  eth_dbg["metrics"]["return_10s_bps"],
                        "eth_return_60s_bps":  eth_dbg["metrics"]["return_60s_bps"],
                        "eth_flow_imbalance_60s": eth_dbg["metrics"]["flow_imbalance_usdt_60s"],
                        "eth_buy_ratio_usdt_60s": eth_dbg["metrics"]["buy_ratio_usdt_60s"],
                        "eth_volume_usdt_60s": eth_dbg["metrics"]["volume_usdt_60s"],
                        "eth_vol_60s_bps":     eth_vol_60s,
                        "sol_price":           sol_dbg["price"],
                        "sol_return_10s_bps":  sol_dbg["metrics"]["return_10s_bps"],
                        "sol_return_60s_bps":  sol_dbg["metrics"]["return_60s_bps"],
                        "sol_flow_imbalance_60s": sol_dbg["metrics"]["flow_imbalance_usdt_60s"],
                        "sol_buy_ratio_usdt_60s": sol_dbg["metrics"]["buy_ratio_usdt_60s"],
                        "sol_volume_usdt_60s": sol_dbg["metrics"]["volume_usdt_60s"],
                        "sol_vol_60s_bps":     sol_vol_60s,
                        "xrp_price":           xrp_dbg["price"],
                        "xrp_return_10s_bps":  xrp_dbg["metrics"]["return_10s_bps"],
                        "xrp_return_60s_bps":  xrp_dbg["metrics"]["return_60s_bps"],
                        "xrp_flow_imbalance_60s": xrp_dbg["metrics"]["flow_imbalance_usdt_60s"],
                        "xrp_buy_ratio_usdt_60s": xrp_dbg["metrics"]["buy_ratio_usdt_60s"],
                        "xrp_volume_usdt_60s": xrp_dbg["metrics"]["volume_usdt_60s"],
                        "xrp_vol_60s_bps":     xrp_vol_60s,
                        "cross_asset_agreement_10s": agree_10s,
                        "cross_asset_agreement_60s": agree_60s,
                        "cross_asset_momentum_10s":  mom_10s,
                        "cross_asset_momentum_60s":  mom_60s,
                        # All markets (grid)
                        "_markets": market_snaps,
                        # Health
                        "_poly_health": next(
                            (ms["poly_health"] for ms in market_snaps.values()
                             if ms.get("market_found")), "dead"),
                        "_asset_health": {
                            sym: dbg["metrics"]["ws_health"]
                            for sym, dbg in [("btc", btc_dbg), ("eth", eth_dbg),
                                             ("sol", sol_dbg), ("xrp", xrp_dbg)]
                        },
                        "_lags": {
                            sym: dbg["metrics"]["ws_lag_ms"]
                            for sym, dbg in [("btc", btc_dbg), ("eth", eth_dbg),
                                             ("sol", sol_dbg), ("xrp", xrp_dbg)]
                        },
                        "_storage": storage.stats(),
                    })

                # ── Write to Parquet every LOG_INTERVAL_SECONDS ───────────
                if now_mono >= next_log_mono:
                    next_log_mono += LOG_INTERVAL_SECONDS

                    for tr_key, ms in market_snaps.items():
                        if not ms["market_found"]:
                            continue

                        c       = ms["coin"]
                        cdbg    = asset_dbg[c]
                        cvol10  = bn_snap.get(c, {}).get("vol_10s_bps")
                        cvol60  = asset_vol60[c]
                        cvol15m = bn_snap.get(c, {}).get("vol_15m_bps")
                        bn_book = multi_ws.get_book(c)

                        # Derived
                        pup      = ms["analytical_prob_up"]
                        up_mid   = ms["up_mid"]
                        edge_bps = (
                            round((pup - up_mid) * 10000, 4)
                            if pup is not None and up_mid is not None else None
                        )
                        mid_5s  = ms["poly_mid_5s_ago"]
                        mid_10s = ms["poly_mid_10s_ago"]
                        chg5s   = (
                            round((up_mid - mid_5s) * 10000, 4)
                            if up_mid is not None and mid_5s is not None else None
                        )
                        chg10s  = (
                            round((up_mid - mid_10s) * 10000, 4)
                            if up_mid is not None and mid_10s is not None else None
                        )

                        row = [
                            # 0-9 identity
                            snap_str,
                            c,
                            ms["interval_min"],
                            ms["slug"],
                            ms["window_start"], ms["window_end"],
                            fmt_float(ms["seconds_remaining"], 1),
                            fmt_float(ms["window_progress"], 2),
                            loop_started_dt.hour,
                            loop_started_dt.weekday(),
                            # 10-18 signal
                            fmt_float(ms["coin_price"], 6),
                            fmt_float(ms["target_price"], 6),
                            fmt_float(ms["distance_bps"], 4),
                            fmt_float(cvol10, 4),
                            fmt_float(cvol60, 4),
                            fmt_float(cvol15m, 4),
                            fmt_float(ms["z_score"], 6),
                            fmt_float(pup, 6),
                            fmt_float(edge_bps, 4),
                            # 19-43 book (top of book + L2 levels + aggregates)
                            fmt_float(ms["up_bid"], 6),
                            fmt_float(ms["up_ask"], 6),
                            fmt_float(up_mid, 6),
                            fmt_float(ms["up_spread_bps"], 4),
                            fmt_float(ms.get("up_best_bid_size"), 2),
                            fmt_float(ms.get("up_best_ask_size"), 2),
                            # book complet UP sérialisé en JSON
                            ms.get("up_book_bids") or "[]",
                            ms.get("up_book_asks") or "[]",
                            # aggregates
                            fmt_float(ms["up_bid_depth_top5"], 2),
                            fmt_float(ms["up_ask_depth_top5"], 2),
                            fmt_float(ms["up_depth_imbalance"], 6),
                            # 26-29 poly flow
                            fmt_int(ms["poly_trade_count_10s"]),
                            fmt_int(ms["poly_trade_count_60s"]),
                            fmt_float(ms["poly_trade_size_10s"], 2),
                            fmt_float(ms["poly_trade_size_60s"], 2),
                            # 30-33 mid dynamics
                            fmt_float(chg5s, 4),
                            fmt_float(chg10s, 4),
                            fmt_float(ms["poly_mid_velocity"], 6),
                            fmt_float(ms["poly_mid_accel"], 6),
                            # 34-40 coin binance (coin-specific, not btc hardcoded)
                            fmt_float(cdbg["metrics"]["return_10s_bps"], 4),
                            fmt_float(cdbg["metrics"]["return_30s_bps"], 4),
                            fmt_float(cdbg["metrics"]["return_60s_bps"], 4),
                            fmt_float(cdbg["metrics"]["flow_imbalance_usdt_10s"], 6),
                            fmt_float(cdbg["metrics"]["flow_imbalance_usdt_60s"], 6),
                            fmt_float(cdbg["metrics"]["buy_ratio_usdt_60s"], 6),
                            fmt_float(cdbg["metrics"]["volume_usdt_60s"], 2),
                            # 41-44 btc context
                            fmt_float(btc_dbg["metrics"]["return_10s_bps"], 4),
                            fmt_float(btc_dbg["metrics"]["return_60s_bps"], 4),
                            fmt_float(btc_dbg["metrics"]["flow_imbalance_usdt_60s"], 6),
                            fmt_float(btc_vol_60s, 4),
                            # 45-48 eth context
                            fmt_float(eth_dbg["metrics"]["return_10s_bps"], 4),
                            fmt_float(eth_dbg["metrics"]["return_60s_bps"], 4),
                            fmt_float(eth_dbg["metrics"]["flow_imbalance_usdt_60s"], 6),
                            fmt_float(eth_vol_60s, 4),
                            # 49-52 sol context
                            fmt_float(sol_dbg["metrics"]["return_10s_bps"], 4),
                            fmt_float(sol_dbg["metrics"]["return_60s_bps"], 4),
                            fmt_float(sol_dbg["metrics"]["flow_imbalance_usdt_60s"], 6),
                            fmt_float(sol_vol_60s, 4),
                            # 53-56 xrp context
                            fmt_float(xrp_dbg["metrics"]["return_10s_bps"], 4),
                            fmt_float(xrp_dbg["metrics"]["return_60s_bps"], 4),
                            fmt_float(xrp_dbg["metrics"]["flow_imbalance_usdt_60s"], 6),
                            fmt_float(xrp_vol_60s, 4),
                            # 57-60 cross-asset
                            fmt_float(agree_10s, 4),
                            fmt_float(agree_60s, 4),
                            fmt_float(mom_10s, 4),
                            fmt_float(mom_60s, 4),
                            # Binance order book
                            fmt_float(bn_book.get("bn_best_bid"),       6),
                            fmt_float(bn_book.get("bn_best_ask"),       6),
                            fmt_float(bn_book.get("bn_mid"),            6),
                            fmt_float(bn_book.get("bn_spread_bps"),     4),
                            fmt_float(bn_book.get("bn_bid_depth5"),     4),
                            fmt_float(bn_book.get("bn_ask_depth5"),     4),
                            fmt_float(bn_book.get("bn_depth_imbalance"),6),
                            fmt_float(bn_book.get("bn_bid_l2_price"),   6),
                            fmt_float(bn_book.get("bn_bid_l2_qty"),     4),
                            fmt_float(bn_book.get("bn_bid_l3_price"),   6),
                            fmt_float(bn_book.get("bn_bid_l3_qty"),     4),
                            fmt_float(bn_book.get("bn_bid_l4_price"),   6),
                            fmt_float(bn_book.get("bn_bid_l4_qty"),     4),
                            fmt_float(bn_book.get("bn_bid_l5_price"),   6),
                            fmt_float(bn_book.get("bn_bid_l5_qty"),     4),
                            fmt_float(bn_book.get("bn_ask_l2_price"),   6),
                            fmt_float(bn_book.get("bn_ask_l2_qty"),     4),
                            fmt_float(bn_book.get("bn_ask_l3_price"),   6),
                            fmt_float(bn_book.get("bn_ask_l3_qty"),     4),
                            fmt_float(bn_book.get("bn_ask_l4_price"),   6),
                            fmt_float(bn_book.get("bn_ask_l4_qty"),     4),
                            fmt_float(bn_book.get("bn_ask_l5_price"),   6),
                            fmt_float(bn_book.get("bn_ask_l5_qty"),     4),
                            # DOWN book
                            fmt_float(ms.get("down_bid"),            6),
                            fmt_float(ms.get("down_ask"),            6),
                            fmt_float(ms.get("down_mid"),            6),
                            fmt_float(ms.get("down_spread_bps"),     4),
                            fmt_float(ms.get("down_best_bid_size"),  2),
                            fmt_float(ms.get("down_best_ask_size"),  2),
                            # book complet DOWN sérialisé en JSON
                            ms.get("down_book_bids") or "[]",
                            ms.get("down_book_asks") or "[]",
                            fmt_float(ms.get("down_bid_depth_top5"), 2),
                            fmt_float(ms.get("down_ask_depth_top5"), 2),
                            fmt_float(ms.get("down_depth_imbalance"),6),
                            # outcome
                            ms["outcome"],
                            fmt_float(ms["coin_close_px"], 6),
                        ]
                        assert len(row) == 101, f"Row {tr_key}: got {len(row)} cols, expected 101"
                        storage.append(row)

            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)

            next_tick_mono += POLL_INTERVAL_SECONDS
            now_after = time.monotonic()
            if next_tick_mono <= now_after:
                missed = int((now_after - next_tick_mono) // POLL_INTERVAL_SECONDS) + 1
                logger.warning("Missed %d tick(s)", missed)
                next_tick_mono += missed * POLL_INTERVAL_SECONDS
            sleep_for = max(0.0, next_tick_mono - time.monotonic())
            if not shutdown_flag.is_set():
                time.sleep(sleep_for)

    finally:
        logger.info("Shutting down…")
        for tr in trackers:
            tr.stop()
        multi_ws.stop()
        storage.close()
        logger.info("Done.")
