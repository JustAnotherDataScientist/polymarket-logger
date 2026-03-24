"""
ws_clients.py
─────────────────
Deux WebSocket clients :

1. PolymarketBookWS (remplace TradeCounterWS + BackgroundBookFetcher REST)
   - Souscrit au channel "market" du CLOB WS
   - Traite price_change  → mise à jour best bid/ask + niveaux L2 en temps réel
   - Traite book_snapshot → mise à jour complète du L2 book
   - Traite last_trade_price → comptage des trades
   - Latence : ~10-50ms (vs ~300ms REST avant)

2. MultiAssetBinanceTradeWS (inchangé)
   - Trades Binance en temps réel via WebSocket
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

import websocket

from config import (
    ASSET_BUCKET_HISTORY_SECONDS,
    ASSET_DEAD_MS,
    ASSET_PRICE_MAX_STALENESS_MS,
    ASSET_STALE_MS,
    ASSET_TRADE_ID_HISTORY_SECONDS,
    BINANCE_MULTI_WSS_URL,
    POLY_DEAD_MS,
    POLY_STALE_MS,
    POLY_TRADE_DEQUE_MAXLEN,
    POLYMARKET_WSS_URL,
)
from utils import (
    bps_change,
    from_unix_ts,
    iso_utc,
    lag_ms,
    normalize_text,
    realized_vol_bps_per_second,
    safe_float,
    safe_int,
    utc_now,
    ws_health_from_lag,
)

logger = logging.getLogger(__name__)

N_LEVELS = 5   # number of L2 levels to track per side


# ─────────────────────────────────────────────────────────────────────────────
# PolymarketBookWS — replaces TradeCounterWS + BackgroundBookFetcher REST
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketBookWS:
    """
    Single WebSocket connection per market that handles:
      - Real-time best bid/ask + L2 book (from price_change / book_snapshot events)
      - Trade counting (from last_trade_price events)

    Replaces both the old TradeCounterWS (trades only) and the
    BackgroundBookFetcher (REST polling every ~300ms).
    Book data is now ~10-50ms fresh instead of ~300ms.
    """

    def __init__(self) -> None:
        self.current_asset_id:  Optional[str] = None
        self.down_asset_id:     Optional[str] = None  # DOWN token (same WS)

        # ── Book state — UP token ─────────────────────────────────────────────
        self._lock = threading.Lock()
        self._bids: list[tuple[float, float]] = []
        self._asks: list[tuple[float, float]] = []
        self._book_ts: Optional[datetime] = None

        # ── Book state — DOWN token ───────────────────────────────────────────
        self._down_bids: list[tuple[float, float]] = []
        self._down_asks: list[tuple[float, float]] = []
        self._down_book_ts: Optional[datetime] = None

        # ── Trade state ───────────────────────────────────────────────────────
        self.trade_events: deque[tuple[datetime, float, str, str]] = deque(
            maxlen=POLY_TRADE_DEQUE_MAXLEN
        )
        self.last_trade_dt: Optional[datetime] = None

        # ── WS plumbing ───────────────────────────────────────────────────────
        self.ws_app: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.connected = False
        self.status = "disconnected"

    # ── WS lifecycle ──────────────────────────────────────────────────────────

    def _send_subscribe(self, ws: websocket.WebSocketApp) -> None:
        asset_id = self.current_asset_id
        if not asset_id:
            return
        ids = [asset_id]
        if self.down_asset_id:
            ids.append(self.down_asset_id)
        ws.send(json.dumps({"assets_ids": ids, "type": "market"}))

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        self.status = "connected"
        logger.info("Polymarket WS connected, subscribing to %s", self.current_asset_id)
        self._send_subscribe(ws)

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            data = json.loads(message)
        except Exception:
            return
        self._process_parsed(data)

    def _process_parsed(self, data: Any) -> None:
        """Process pre-parsed JSON data (avoids double json.loads when recording)."""
        events = data if isinstance(data, list) else [data]
        for event in events:
            if isinstance(event, dict):
                self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = normalize_text(event.get("event_type") or event.get("update_type") or "").lower()
        now = utc_now()

        if event_type == "price_change":
            self._handle_price_change(event, now)

        elif event_type in ("book", "book_snapshot"):
            self._handle_book_snapshot(event, now)

        elif event_type == "last_trade_price":
            self._handle_trade(event, now)

        elif event_type == "subscribed":
            self.status = "subscribed"

    def _is_down_token(self, asset_id: str) -> bool:
        return bool(self.down_asset_id and asset_id == self.down_asset_id)

    def _handle_price_change(self, event: dict, now: datetime) -> None:
        asset_id = normalize_text(event.get("asset_id") or event.get("token_id") or "")
        with self._lock:
            is_down = self._is_down_token(asset_id)
            if asset_id and asset_id != self.current_asset_id and not is_down:
                return

            best_bid = safe_float(event.get("best_bid"))
            best_ask = safe_float(event.get("best_ask"))
            bids_raw = event.get("bids")
            asks_raw = event.get("asks")

            if is_down:
                if bids_raw and asks_raw:
                    self._update_down_book(bids_raw, asks_raw, now)
                elif best_bid is not None and best_ask is not None:
                    self._update_down_top(best_bid, best_ask, now)
                self._down_book_ts = now
            else:
                if bids_raw and asks_raw:
                    self._update_full_book(bids_raw, asks_raw, now)
                elif best_bid is not None and best_ask is not None:
                    self._update_top_of_book(best_bid, best_ask, now)
                self._book_ts = now

            # Trade signal (only track for UP token)
            if not is_down:
                chg_size = safe_float(event.get("change_size"))
                if chg_size and chg_size > 0:
                    ts = from_unix_ts(event.get("timestamp")) or now
                    tx = normalize_text(event.get("transaction_hash") or
                                        f"pc_{ts.timestamp():.6f}_{chg_size}")
                    self.trade_events.append((ts, chg_size, tx, asset_id or ""))
                    self.last_trade_dt = ts
                    self._clear_old_trades(now)

    def _handle_book_snapshot(self, event: dict, now: datetime) -> None:
        """Full L2 book snapshot."""
        asset_id = normalize_text(event.get("asset_id") or event.get("token_id") or "")
        with self._lock:
            is_down = self._is_down_token(asset_id)
            if asset_id and asset_id != self.current_asset_id and not is_down:
                return
            bids_raw = event.get("bids", [])
            asks_raw = event.get("asks", [])
            if is_down:
                self._update_down_book(bids_raw, asks_raw, now)
                self._down_book_ts = now
            else:
                self._update_full_book(bids_raw, asks_raw, now)
                self._book_ts = now

    def _handle_trade(self, event: dict, now: datetime) -> None:
        """last_trade_price: confirmed on-chain trade."""
        asset_id = normalize_text(event.get("asset_id") or "")
        trade_dt = from_unix_ts(event.get("timestamp")) or now
        size     = safe_float(event.get("size")) or 0.0
        tx_hash  = normalize_text(event.get("transaction_hash") or "")
        with self._lock:
            self.trade_events.append((trade_dt, size, tx_hash, asset_id))
            if asset_id == self.current_asset_id:
                self.last_trade_dt = trade_dt
            self._clear_old_trades(now)

    def _update_full_book(self, bids_raw: list, asks_raw: list,
                           now: datetime) -> None:
        """Parse and store full L2 book (call under lock)."""
        def _parse(levels: list) -> list[tuple[float, float]]:
            result = []
            for lv in levels:
                if isinstance(lv, dict):
                    p = safe_float(lv.get("price"))
                    s = safe_float(lv.get("size") or lv.get("quantity") or 0)
                elif isinstance(lv, (list, tuple)) and len(lv) >= 2:
                    p = safe_float(lv[0])
                    s = safe_float(lv[1])
                else:
                    continue
                if p is not None:
                    result.append((p, s or 0.0))
            return result

        bids = sorted(_parse(bids_raw), key=lambda x: x[0], reverse=True)
        asks = sorted(_parse(asks_raw), key=lambda x: x[0])
        self._bids = bids   # book complet — pas de troncature
        self._asks = asks

    def _update_top_of_book(self, best_bid: float, best_ask: float,
                             now: datetime) -> None:
        """Update only level 1 when full book not available (call under lock)."""
        if self._bids:
            self._bids[0] = (best_bid, self._bids[0][1])
        else:
            self._bids = [(best_bid, 0.0)]
        if self._asks:
            self._asks[0] = (best_ask, self._asks[0][1])
        else:
            self._asks = [(best_ask, 0.0)]

    def _update_down_book(self, bids_raw: list, asks_raw: list,
                           now: datetime) -> None:
        """Parse and store DOWN token L2 book (call under lock)."""
        def _parse(levels):
            result = []
            for lv in levels:
                if isinstance(lv, dict):
                    p = safe_float(lv.get("price"))
                    s = safe_float(lv.get("size") or lv.get("quantity") or 0)
                elif isinstance(lv, (list, tuple)) and len(lv) >= 2:
                    p = safe_float(lv[0]); s = safe_float(lv[1])
                else:
                    continue
                if p is not None:
                    result.append((p, s or 0.0))
            return result
        bids = sorted(_parse(bids_raw), key=lambda x: x[0], reverse=True)
        asks = sorted(_parse(asks_raw), key=lambda x: x[0])
        self._down_bids = bids   # book complet — pas de troncature
        self._down_asks = asks

    def _update_down_top(self, best_bid: float, best_ask: float,
                          now: datetime) -> None:
        """Update DOWN token level 1 only."""
        if self._down_bids:
            self._down_bids[0] = (best_bid, self._down_bids[0][1])
        else:
            self._down_bids = [(best_bid, 0.0)]
        if self._down_asks:
            self._down_asks[0] = (best_ask, self._down_asks[0][1])
        else:
            self._down_asks = [(best_ask, 0.0)]

    def _clear_old_trades(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=60)
        while self.trade_events and self.trade_events[0][0] < cutoff:
            self.trade_events.popleft()

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        self.connected = False
        self.status = f"error: {error}"
        logger.warning("Polymarket WS error: %s", error)

    def _on_close(self, ws: websocket.WebSocketApp,
                  close_status_code: Any, close_msg: Any) -> None:
        self.connected = False
        if self.stop_event.is_set():
            self.status = "closed"
        else:
            self.status = "reconnecting"
            logger.warning("Polymarket WS closed unexpectedly (will reconnect)")

    def _run_forever(self) -> None:
        backoff = 2  # secondes, avec backoff exponentiel
        while not self.stop_event.is_set():
            try:
                self.ws_app = websocket.WebSocketApp(
                    POLYMARKET_WSS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws_app.run_forever(ping_interval=20, ping_timeout=10)
                backoff = 2  # reset backoff après une connexion réussie
            except Exception as exc:
                self.status = f"error: {exc}"
                logger.error("Polymarket WS exception: %s", exc)
                backoff = min(backoff * 2, 60)  # max 60s entre tentatives
            if self.stop_event.is_set():
                break
            logger.info("Polymarket WS reconnecting in %ds", backoff)
            time.sleep(backoff)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, asset_id: str, down_asset_id: Optional[str] = None) -> None:
        self.stop()
        with self._lock:
            self.current_asset_id = asset_id
            self.down_asset_id    = down_asset_id
            self._bids = []; self._asks = []; self._book_ts = None
            self._down_bids = []; self._down_asks = []; self._down_book_ts = None
        self.stop_event = threading.Event()
        self.status = "starting"
        self.last_trade_dt = None
        self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
        self.ws_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.ws_app is not None:
            try:
                self.ws_app.close()
            except Exception:
                pass
        if self.ws_thread is not None and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)
        self.connected = False
        self.status = "stopped"
        self.current_asset_id = None
        self.last_trade_dt = None
        self.ws_app = None
        self.ws_thread = None

    def get_book_metrics(self, now_dt: datetime) -> dict[str, Any]:
        """
        Returns all book data needed for the parquet row.
        Mirrors the old api.get_up_book_metrics() interface but much fresher.
        """
        with self._lock:
            bids = list(self._bids)
            asks = list(self._asks)
            book_ts = self._book_ts

        def _lv(levels: list, idx: int, field: int) -> Optional[float]:
            """Get price (field=0) or size (field=1) at level idx (0-based)."""
            return levels[idx][field] if idx < len(levels) else None

        best_bid = _lv(bids, 0, 0)
        best_ask = _lv(asks, 0, 0)
        best_bid_size = _lv(bids, 0, 1)
        best_ask_size = _lv(asks, 0, 1)
        best_mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread_bps = (
            (best_ask - best_bid) / best_mid * 10000
            if best_bid and best_ask and best_mid else None
        )
        bid_depth5 = sum(s for _, s in bids[:5]) if bids else None
        ask_depth5 = sum(s for _, s in asks[:5]) if asks else None
        depth_imb  = (
            (bid_depth5 - ask_depth5) / (bid_depth5 + ask_depth5)
            if bid_depth5 is not None and ask_depth5 is not None
            and (bid_depth5 + ask_depth5) > 0
            else None
        )

        lag_value = lag_ms(now_dt, book_ts)
        health    = ws_health_from_lag(self.status, lag_value, POLY_STALE_MS, POLY_DEAD_MS)

        # Book complet sérialisé en JSON (remplace bid/ask_l2..l5_price/size)
        up_book_bids_json = json.dumps([[p, s] for p, s in bids])
        up_book_asks_json = json.dumps([[p, s] for p, s in asks])

        return {
            "up_bid":            best_bid,
            "up_ask":            best_ask,
            "up_mid":            best_mid,
            "up_spread_bps":     spread_bps,
            "up_best_bid_size":  best_bid_size,
            "up_best_ask_size":  best_ask_size,
            "up_bid_depth_top5": bid_depth5,
            "up_ask_depth_top5": ask_depth5,
            "up_depth_imbalance": depth_imb,
            "book_lag_ms":       lag_value,
            "book_health":       health,
            "up_book_bids":      up_book_bids_json,   # JSON [[price, size], ...]
            "up_book_asks":      up_book_asks_json,   # JSON [[price, size], ...]
        }

    def get_down_book_metrics(self, now_dt: datetime) -> dict[str, Any]:
        """Returns DOWN token book metrics with down_* key names."""
        with self._lock:
            bids = list(self._down_bids)
            asks = list(self._down_asks)
            book_ts = self._down_book_ts

        def _lv(levels, idx, field):
            return levels[idx][field] if idx < len(levels) else None

        best_bid = _lv(bids, 0, 0)
        best_ask = _lv(asks, 0, 0)
        best_bid_size = _lv(bids, 0, 1)
        best_ask_size = _lv(asks, 0, 1)
        best_mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread_bps = (
            (best_ask - best_bid) / best_mid * 10000
            if best_bid and best_ask and best_mid else None
        )
        bid_depth5 = sum(s for _, s in bids[:5]) if bids else None
        ask_depth5 = sum(s for _, s in asks[:5]) if asks else None
        depth_imb = (
            (bid_depth5 - ask_depth5) / (bid_depth5 + ask_depth5)
            if bid_depth5 is not None and ask_depth5 is not None
            and (bid_depth5 + ask_depth5) > 0 else None
        )

        # Book complet sérialisé en JSON (remplace down_bid/ask_l2..l5_price/size)
        down_book_bids_json = json.dumps([[p, s] for p, s in bids])
        down_book_asks_json = json.dumps([[p, s] for p, s in asks])

        return {
            "down_bid":             best_bid,
            "down_ask":             best_ask,
            "down_mid":             best_mid,
            "down_spread_bps":      spread_bps,
            "down_best_bid_size":   best_bid_size,
            "down_best_ask_size":   best_ask_size,
            "down_bid_depth_top5":  bid_depth5,
            "down_ask_depth_top5":  ask_depth5,
            "down_depth_imbalance": depth_imb,
            "down_book_bids":       down_book_bids_json,  # JSON [[price, size], ...]
            "down_book_asks":       down_book_asks_json,  # JSON [[price, size], ...]
        }

    def get_trade_metrics(self, now_dt: datetime) -> dict[str, Any]:
        """Trade counts and sizes (rolling 10s/60s)."""
        with self._lock:
            self._clear_old_trades(now_dt)
            cutoff_10 = now_dt - timedelta(seconds=10)
            cutoff_60 = now_dt - timedelta(seconds=60)
            c10 = c60 = 0
            s10 = s60 = 0.0
            current_asset = self.current_asset_id

            for trade_dt, size, tx_hash, asset_id in self.trade_events:
                if current_asset and asset_id and asset_id != current_asset:
                    continue
                if trade_dt >= cutoff_60:
                    c60 += 1
                    s60 += size
                    if trade_dt >= cutoff_10:
                        c10 += 1
                        s10 += size

            lag_value = lag_ms(now_dt, self.last_trade_dt)
            health    = ws_health_from_lag(self.status, lag_value, POLY_STALE_MS, POLY_DEAD_MS)

            return {
                "poly_trade_count_10s": c10,
                "poly_trade_count_60s": c60,
                "poly_trade_size_10s":  s10,
                "poly_trade_size_60s":  s60,
                "poly_ws_lag_ms":       lag_value,
                "poly_ws_status":       self.status,
                "poly_ws_health":       health,
            }


# ─────────────────────────────────────────────────────────────────────────────
# MultiAssetBinanceTradeWS — unchanged from v2
# ─────────────────────────────────────────────────────────────────────────────

class MultiAssetBinanceTradeWS:
    def __init__(self, symbols_by_asset: dict[str, str]) -> None:
        self.symbols_by_asset = {k.lower(): v.upper() for k, v in symbols_by_asset.items()}
        self.asset_by_symbol  = {v.upper(): k.lower() for k, v in self.symbols_by_asset.items()}
        self.state: dict[str, dict[str, Any]] = {}
        for asset in self.symbols_by_asset:
            self.state[asset] = {
                "second_buckets":       deque(maxlen=4000),
                "recent_trade_ids":     deque(maxlen=500_000),
                "recent_trade_id_set":  set(),
                "last_trade_dt":        None,
                "last_trade_price":     None,
                # Order book depth (from @depth20@100ms stream)
                "book_bids":            [],   # [(price, qty), ...] sorted desc
                "book_asks":            [],   # [(price, qty), ...] sorted asc
                "book_ts":              None, # last update datetime
            }
        self.lock        = threading.Lock()
        self.ws_app:     Optional[websocket.WebSocketApp] = None
        self.ws_thread:  Optional[threading.Thread] = None
        self.stop_event  = threading.Event()
        self.connected   = False
        self.status      = "disconnected"

    def _clear_old_for_asset(self, asset: str, now_dt: datetime) -> None:
        state = self.state[asset]
        cutoff_sec = int(now_dt.timestamp()) - ASSET_BUCKET_HISTORY_SECONDS
        while state["second_buckets"] and state["second_buckets"][0]["sec"] < cutoff_sec:
            state["second_buckets"].popleft()
        trade_id_cutoff = int(now_dt.timestamp()) - ASSET_TRADE_ID_HISTORY_SECONDS
        while (state["recent_trade_ids"]
               and state["recent_trade_ids"][0][1] < trade_id_cutoff):
            old_id, _ = state["recent_trade_ids"].popleft()
            state["recent_trade_id_set"].discard(old_id)
        # Safety cap: rebuild set from deque if it grows too large
        if len(state["recent_trade_id_set"]) > 500_000:
            state["recent_trade_id_set"] = set(tid for tid, _ in state["recent_trade_ids"])

    def _on_open(self, ws):
        self.connected = True
        self.status = "connected"
        logger.info("Binance multi-asset WS connected")

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return
        self._process_parsed(payload)

    def _process_parsed(self, payload: Any) -> None:
        """Process pre-parsed JSON payload (avoids double json.loads when recording)."""
        if not isinstance(payload, dict):
            return
        stream = normalize_text(payload.get("stream") or "").lower()
        data   = payload.get("data", payload)
        if not isinstance(data, dict):
            return

        # Route by stream name (depth stream has no "e" field)
        if "depth" in stream:
            self._handle_depth(data, stream)
            return

        # Fallback: route by event type for trade stream
        event_type = normalize_text(data.get("e") or "").lower()
        if event_type != "trade":
            return

        symbol         = normalize_text(data.get("s")).upper()
        asset          = self.asset_by_symbol.get(symbol)
        if not asset:
            return

        trade_dt       = from_unix_ts(data.get("T"))
        price          = safe_float(data.get("p"))
        qty            = safe_float(data.get("q"))
        buyer_is_maker = bool(data.get("m"))
        trade_id       = safe_int(data.get("t"))

        if trade_dt is None or price is None or qty is None or trade_id is None:
            return

        sec      = int(trade_dt.timestamp())
        notional = price * qty
        is_buy   = not buyer_is_maker

        with self.lock:
            self._clear_old_for_asset(asset, utc_now())
            state = self.state[asset]
            if trade_id in state["recent_trade_id_set"]:
                return
            state["recent_trade_id_set"].add(trade_id)
            state["recent_trade_ids"].append((trade_id, sec))

            if state["second_buckets"] and state["second_buckets"][-1]["sec"] == sec:
                bucket = state["second_buckets"][-1]
            else:
                bucket = {
                    "sec": sec, "count": 0,
                    "vol_asset": 0.0, "vol_usdt": 0.0,
                    "buy_asset": 0.0, "sell_asset": 0.0,
                    "buy_usdt":  0.0, "sell_usdt":  0.0,
                    "close_price": price,
                }
                state["second_buckets"].append(bucket)

            bucket["count"]       += 1
            bucket["vol_asset"]   += qty
            bucket["vol_usdt"]    += notional
            bucket["close_price"]  = price
            if is_buy:
                bucket["buy_asset"] += qty
                bucket["buy_usdt"]  += notional
            else:
                bucket["sell_asset"] += qty
                bucket["sell_usdt"]  += notional

            state["last_trade_dt"]    = trade_dt
            state["last_trade_price"] = price

    def _handle_depth(self, data: dict, stream: str = "") -> None:
        """Handle @depth20@100ms book update."""
        # data has no "s" field for depth20 stream — extract from stream name
        # e.g. "btcusdt@depth20@100ms" → "BTCUSDT"
        symbol = normalize_text(data.get("s") or "").upper()
        if not symbol and stream:
            symbol = stream.split("@")[0].upper()  # "btcusdt" → "BTCUSDT"
        asset = self.asset_by_symbol.get(symbol)
        if not asset:
            return
        # @depth20@100ms uses "bids"/"asks", incremental diff uses "b"/"a"
        bids_raw = data.get("bids") or data.get("b", [])
        asks_raw = data.get("asks") or data.get("a", [])
        def _parse(levels):
            result = []
            for lv in levels:
                try:
                    p, q = float(lv[0]), float(lv[1])
                    if q > 0:
                        result.append((p, q))
                except (ValueError, IndexError):
                    continue
            return result
        bids = sorted(_parse(bids_raw), key=lambda x: x[0], reverse=True)[:10]
        asks = sorted(_parse(asks_raw), key=lambda x: x[0])[:10]
        with self.lock:
            self.state[asset]["book_bids"] = bids
            self.state[asset]["book_asks"] = asks
            self.state[asset]["book_ts"]   = utc_now()

    def _on_error(self, ws, error):
        self.connected = False
        self.status = f"error: {error}"
        logger.warning("Binance WS error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        if self.stop_event.is_set():
            self.status = "closed"
        else:
            self.status = "reconnecting"
            logger.warning("Binance WS closed (will reconnect) — state preserved")

    def _run_forever(self) -> None:
        backoff = 2  # secondes, avec backoff exponentiel
        while not self.stop_event.is_set():
            try:
                self.ws_app = websocket.WebSocketApp(
                    BINANCE_MULTI_WSS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws_app.run_forever(ping_interval=20, ping_timeout=10)
                backoff = 2  # reset backoff après une connexion réussie
            except Exception as exc:
                self.status = f"error: {exc}"
                logger.error("Binance WS exception: %s", exc)
                backoff = min(backoff * 2, 60)  # max 60s entre tentatives
            if self.stop_event.is_set():
                break
            logger.info("Binance WS reconnecting in %ds", backoff)
            time.sleep(backoff)

    def start(self) -> None:
        self.stop()
        self.stop_event = threading.Event()
        self.status = "starting"
        self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
        self.ws_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.ws_app is not None:
            try: self.ws_app.close()
            except Exception: pass
        if self.ws_thread is not None and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)
        self.connected = False
        self.status    = "stopped"
        self.ws_app    = None
        self.ws_thread = None

    def _price_at_or_before_cutoff(self, asset: str,
                                    cutoff_dt: datetime) -> Optional[float]:
        state = self.state[asset]
        cutoff_sec = int(cutoff_dt.timestamp())
        candidate: Optional[float] = None
        for bucket in state["second_buckets"]:
            if bucket["sec"] <= cutoff_sec:
                p = safe_float(bucket.get("close_price"))
                if p is not None:
                    candidate = p
            else:
                break
        return candidate

    def snapshot_all(self, now_dt: datetime) -> dict[str, Any]:
        """
        Acquire lock briefly to copy raw state, then compute metrics outside.

        Optimisations vs v1:
          - Lock held only for shallow copies (~50μs vs ~5ms before)
          - Single-pass bucket iteration instead of 5× filter + 3× vol + 4× price scan
          - WS callbacks unblocked during computation
        """
        # ── Phase 1: copy raw data under lock (fast) ─────────────────────
        raw: dict[str, dict] = {}
        with self.lock:
            for asset in self.symbols_by_asset:
                self._clear_old_for_asset(asset, now_dt)
                state = self.state[asset]
                raw[asset] = {
                    "buckets":    list(state["second_buckets"]),
                    "last_price": state.get("last_trade_price"),
                    "last_dt":    state.get("last_trade_dt"),
                    "book_bids":  list(state.get("book_bids", [])),
                    "book_asks":  list(state.get("book_asks", [])),
                    "book_ts":    state.get("book_ts"),
                }

        # ── Phase 2: compute metrics outside lock ────────────────────────
        now_ts = int(now_dt.timestamp())
        _WINDOWS = (("10s", 10), ("30s", 30), ("60s", 60), ("15m", 900), ("1h", 3600))
        _VOL_WINDOWS = (10, 60, 900)
        _RET_WINDOWS = (10, 30, 60, 900)

        result = {}
        for asset, snap in raw.items():
            buckets    = snap["buckets"]
            last_price = snap["last_price"]
            ws_now     = snap["last_dt"]

            lag = (now_dt - ws_now).total_seconds() * 1000 if ws_now else None
            health = ws_health_from_lag(
                "connected" if ws_now else "stopped",
                lag, ASSET_STALE_MS, ASSET_DEAD_MS,
            )

            # ── Single-pass over buckets ─────────────────────────────────
            cutoffs = {s: now_ts - s for _, s in _WINDOWS}
            win_buy  = {s: 0.0 for _, s in _WINDOWS}
            win_sell = {s: 0.0 for _, s in _WINDOWS}

            vol_prices = {ws: [] for ws in _VOL_WINDOWS}
            ret_prices = {ws: None for ws in _RET_WINDOWS}

            for b in buckets:
                sec = b["sec"]
                buy_u  = b.get("buy_usdt", 0.0)
                sell_u = b.get("sell_usdt", 0.0)
                cp     = b.get("close_price")

                for _, s in _WINDOWS:
                    if sec >= cutoffs[s]:
                        win_buy[s]  += buy_u
                        win_sell[s] += sell_u

                if cp is not None:
                    for ws in _VOL_WINDOWS:
                        if sec >= cutoffs[ws]:
                            vol_prices[ws].append(cp)
                    for ws in _RET_WINDOWS:
                        if sec <= now_ts - ws:
                            ret_prices[ws] = cp

            def _ratio(n, d): return n / d if d > 0 else None
            def _flow(b, s, t): return (b - s) / t if t > 0 else None
            def _bps(cur, past):
                if cur and past and past > 0:
                    return (cur - past) / past * 10000
                return None

            total_60 = win_buy[60] + win_sell[60]

            result[asset] = {
                "price":                last_price,
                "last_trade_dt":        ws_now,
                "ws_lag_ms":            lag,
                "ws_health":            health,
                "return_10s_bps":       _bps(last_price, ret_prices[10]),
                "return_30s_bps":       _bps(last_price, ret_prices[30]),
                "return_60s_bps":       _bps(last_price, ret_prices[60]),
                "return_15m_bps":       _bps(last_price, ret_prices[900]),
                "flow_imbalance_usdt_10s": _flow(win_buy[10], win_sell[10],
                                                  win_buy[10] + win_sell[10]),
                "flow_imbalance_usdt_60s": _flow(win_buy[60], win_sell[60], total_60),
                "buy_ratio_usdt_60s":      _ratio(win_buy[60], total_60),
                "volume_usdt_60s":         total_60,
                "vol_10s_bps":          realized_vol_bps_per_second(vol_prices[10]),
                "vol_60s_bps":          realized_vol_bps_per_second(vol_prices[60]),
                "vol_15m_bps":          realized_vol_bps_per_second(vol_prices[900]),
                # Binance book (copied under lock above)
                "book_bids":            snap["book_bids"],
                "book_asks":            snap["book_asks"],
                "book_ts":              snap["book_ts"],
            }
        return result

    def get_book(self, asset: str) -> dict[str, Any]:
        """Return Binance order book metrics for an asset."""
        with self.lock:
            state = self.state.get(asset, {})
            bids  = state.get("book_bids", [])
            asks  = state.get("book_asks", [])

        if not bids or not asks:
            return {
                "bn_best_bid": None, "bn_best_ask": None,
                "bn_spread_bps": None, "bn_mid": None,
                "bn_bid_depth5": None, "bn_ask_depth5": None,
                "bn_depth_imbalance": None,
                "bn_bid_l2_price": None, "bn_bid_l2_qty": None,
                "bn_ask_l2_price": None, "bn_ask_l2_qty": None,
                "bn_bid_l3_price": None, "bn_bid_l3_qty": None,
                "bn_ask_l3_price": None, "bn_ask_l3_qty": None,
                "bn_bid_l4_price": None, "bn_bid_l4_qty": None,
                "bn_ask_l4_price": None, "bn_ask_l4_qty": None,
                "bn_bid_l5_price": None, "bn_bid_l5_qty": None,
                "bn_ask_l5_price": None, "bn_ask_l5_qty": None,
            }

        def lv(levels, i, f): return levels[i][f] if i < len(levels) else None

        best_bid = lv(bids, 0, 0)
        best_ask = lv(asks, 0, 0)
        mid      = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread   = (best_ask - best_bid) / mid * 10000 if mid else None
        bid_d5   = sum(q for _, q in bids[:5]) if bids else None
        ask_d5   = sum(q for _, q in asks[:5]) if asks else None
        imb      = (bid_d5 - ask_d5) / (bid_d5 + ask_d5) if bid_d5 and ask_d5 and (bid_d5 + ask_d5) > 0 else None

        return {
            "bn_best_bid":      best_bid,
            "bn_best_ask":      best_ask,
            "bn_mid":           mid,
            "bn_spread_bps":    spread,
            "bn_bid_depth5":    bid_d5,
            "bn_ask_depth5":    ask_d5,
            "bn_depth_imbalance": imb,
            "bn_bid_l2_price":  lv(bids, 1, 0), "bn_bid_l2_qty": lv(bids, 1, 1),
            "bn_bid_l3_price":  lv(bids, 2, 0), "bn_bid_l3_qty": lv(bids, 2, 1),
            "bn_bid_l4_price":  lv(bids, 3, 0), "bn_bid_l4_qty": lv(bids, 3, 1),
            "bn_bid_l5_price":  lv(bids, 4, 0), "bn_bid_l5_qty": lv(bids, 4, 1),
            "bn_ask_l2_price":  lv(asks, 1, 0), "bn_ask_l2_qty": lv(asks, 1, 1),
            "bn_ask_l3_price":  lv(asks, 2, 0), "bn_ask_l3_qty": lv(asks, 2, 1),
            "bn_ask_l4_price":  lv(asks, 3, 0), "bn_ask_l4_qty": lv(asks, 3, 1),
            "bn_ask_l5_price":  lv(asks, 4, 0), "bn_ask_l5_qty": lv(asks, 4, 1),
        }

    def get_last_price(self, asset: str, now_dt: datetime,
                        max_staleness_ms: int = ASSET_PRICE_MAX_STALENESS_MS
                        ) -> Optional[float]:
        with self.lock:
            state = self.state[asset]
            lv = lag_ms(now_dt, state["last_trade_dt"])
            if state["last_trade_price"] is None:
                return None
            if lv is None or lv > max_staleness_ms:
                return None
            return state["last_trade_price"]

    def get_realized_vol(self, asset: str, now_dt: datetime,
                          window_seconds: int) -> Optional[float]:
        with self.lock:
            state = self.state[asset]
            cutoff_sec = int(now_dt.timestamp()) - window_seconds
            prices = [
                safe_float(b.get("close_price"))
                for b in state["second_buckets"]
                if b["sec"] >= cutoff_sec
            ]
        prices = [p for p in prices if p is not None]
        return realized_vol_bps_per_second(prices)

    def get_metrics(self, asset: str, now_dt: datetime) -> dict[str, Any]:
        with self.lock:
            self._clear_old_for_asset(asset, now_dt)
            state = self.state[asset]
            window_defs = {"10s": 10, "30s": 30, "60s": 60, "15m": 900, "1h": 3600}
            cutoff_secs = {
                label: int((now_dt - timedelta(seconds=s)).timestamp())
                for label, s in window_defs.items()
            }
            stats: dict[str, dict[str, float]] = {
                label: {"count": 0.0, "vol_asset": 0.0, "vol_usdt": 0.0,
                        "buy_asset": 0.0, "sell_asset": 0.0,
                        "buy_usdt":  0.0, "sell_usdt":  0.0}
                for label in window_defs
            }
            for bucket in state["second_buckets"]:
                sec = bucket["sec"]
                for label in window_defs:
                    if sec >= cutoff_secs[label]:
                        for k in ("count","vol_asset","vol_usdt",
                                  "buy_asset","sell_asset","buy_usdt","sell_usdt"):
                            stats[label][k] += float(bucket[k])

            lag_value = lag_ms(now_dt, state["last_trade_dt"])
            health    = ws_health_from_lag(self.status, lag_value,
                                           ASSET_STALE_MS, ASSET_DEAD_MS)
            current_price = state["last_trade_price"]
            if lag_value is not None and lag_value > ASSET_PRICE_MAX_STALENESS_MS:
                current_price = None

            p10  = self._price_at_or_before_cutoff(asset, now_dt - timedelta(seconds=10))
            p30  = self._price_at_or_before_cutoff(asset, now_dt - timedelta(seconds=30))
            p60  = self._price_at_or_before_cutoff(asset, now_dt - timedelta(seconds=60))
            p15m = self._price_at_or_before_cutoff(asset, now_dt - timedelta(minutes=15))
            p1h  = self._price_at_or_before_cutoff(asset, now_dt - timedelta(hours=1))

            def ratio(n, d): return n/d if d > 0 else None
            def flow_imb(b, s, t): return (b-s)/t if t > 0 else None

            return {
                # trade counts
                "trade_count_10s": int(stats["10s"]["count"]),
                "trade_count_30s": int(stats["30s"]["count"]),
                "trade_count_60s": int(stats["60s"]["count"]),
                "trade_count_15m": int(stats["15m"]["count"]),
                "trade_count_1h":  int(stats["1h"]["count"]),
                # volumes (asset)
                "volume_asset_10s": stats["10s"]["vol_asset"],
                "volume_asset_30s": stats["30s"]["vol_asset"],
                "volume_asset_60s": stats["60s"]["vol_asset"],
                "volume_asset_15m": stats["15m"]["vol_asset"],
                "volume_asset_1h":  stats["1h"]["vol_asset"],
                # volumes (usdt)
                "volume_usdt_10s": stats["10s"]["vol_usdt"],
                "volume_usdt_30s": stats["30s"]["vol_usdt"],
                "volume_usdt_60s": stats["60s"]["vol_usdt"],
                "volume_usdt_15m": stats["15m"]["vol_usdt"],
                "volume_usdt_1h":  stats["1h"]["vol_usdt"],
                # buy volumes (asset)
                "buy_volume_asset_10s": stats["10s"]["buy_asset"],
                "buy_volume_asset_30s": stats["30s"]["buy_asset"],
                "buy_volume_asset_60s": stats["60s"]["buy_asset"],
                "buy_volume_asset_15m": stats["15m"]["buy_asset"],
                "buy_volume_asset_1h":  stats["1h"]["buy_asset"],
                # sell volumes (asset)
                "sell_volume_asset_10s": stats["10s"]["sell_asset"],
                "sell_volume_asset_30s": stats["30s"]["sell_asset"],
                "sell_volume_asset_60s": stats["60s"]["sell_asset"],
                "sell_volume_asset_15m": stats["15m"]["sell_asset"],
                "sell_volume_asset_1h":  stats["1h"]["sell_asset"],
                # buy volumes (usdt)
                "buy_volume_usdt_10s": stats["10s"]["buy_usdt"],
                "buy_volume_usdt_30s": stats["30s"]["buy_usdt"],
                "buy_volume_usdt_60s": stats["60s"]["buy_usdt"],
                "buy_volume_usdt_15m": stats["15m"]["buy_usdt"],
                "buy_volume_usdt_1h":  stats["1h"]["buy_usdt"],
                # sell volumes (usdt)
                "sell_volume_usdt_10s": stats["10s"]["sell_usdt"],
                "sell_volume_usdt_30s": stats["30s"]["sell_usdt"],
                "sell_volume_usdt_60s": stats["60s"]["sell_usdt"],
                "sell_volume_usdt_15m": stats["15m"]["sell_usdt"],
                "sell_volume_usdt_1h":  stats["1h"]["sell_usdt"],
                # ratios & flow
                "buy_ratio_usdt_10s": ratio(stats["10s"]["buy_usdt"], stats["10s"]["vol_usdt"]),
                "buy_ratio_usdt_30s": ratio(stats["30s"]["buy_usdt"], stats["30s"]["vol_usdt"]),
                "buy_ratio_usdt_60s": ratio(stats["60s"]["buy_usdt"], stats["60s"]["vol_usdt"]),
                "buy_ratio_usdt_15m": ratio(stats["15m"]["buy_usdt"], stats["15m"]["vol_usdt"]),
                "buy_ratio_usdt_1h":  ratio(stats["1h"]["buy_usdt"],  stats["1h"]["vol_usdt"]),
                "flow_imbalance_usdt_10s": flow_imb(stats["10s"]["buy_usdt"], stats["10s"]["sell_usdt"], stats["10s"]["vol_usdt"]),
                "flow_imbalance_usdt_30s": flow_imb(stats["30s"]["buy_usdt"], stats["30s"]["sell_usdt"], stats["30s"]["vol_usdt"]),
                "flow_imbalance_usdt_60s": flow_imb(stats["60s"]["buy_usdt"], stats["60s"]["sell_usdt"], stats["60s"]["vol_usdt"]),
                "flow_imbalance_usdt_15m": flow_imb(stats["15m"]["buy_usdt"], stats["15m"]["sell_usdt"], stats["15m"]["vol_usdt"]),
                "flow_imbalance_usdt_1h":  flow_imb(stats["1h"]["buy_usdt"],  stats["1h"]["sell_usdt"],  stats["1h"]["vol_usdt"]),
                # vwap (all windows)
                "vwap_10s": stats["10s"]["vol_usdt"] / stats["10s"]["vol_asset"] if stats["10s"]["vol_asset"] > 0 else None,
                "vwap_30s": stats["30s"]["vol_usdt"] / stats["30s"]["vol_asset"] if stats["30s"]["vol_asset"] > 0 else None,
                "vwap_60s": stats["60s"]["vol_usdt"] / stats["60s"]["vol_asset"] if stats["60s"]["vol_asset"] > 0 else None,
                "vwap_15m": stats["15m"]["vol_usdt"] / stats["15m"]["vol_asset"] if stats["15m"]["vol_asset"] > 0 else None,
                "vwap_1h":  stats["1h"]["vol_usdt"]  / stats["1h"]["vol_asset"]  if stats["1h"]["vol_asset"]  > 0 else None,
                # returns
                "return_10s_bps":  bps_change(current_price, p10),
                "return_30s_bps":  bps_change(current_price, p30),
                "return_60s_bps":  bps_change(current_price, p60),
                "return_15m_bps":  bps_change(current_price, p15m),
                "return_1h_bps":   bps_change(current_price, p1h),
                # health
                "ws_last_trade_utc": iso_utc(state["last_trade_dt"]) if state["last_trade_dt"] else "",
                "ws_lag_ms":   lag_value,
                "ws_status":   self.status,
                "ws_health":   health,
            }