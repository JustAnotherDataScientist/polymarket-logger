"""
event_recording.py — Hooks into existing WS clients to record raw events.

Minimal-invasion approach: these are thin wrappers/mixins that intercept
the raw WS messages BEFORE they're aggregated, and pipe them to the EventStore.

Usage in main.py or logger_core.py:
    from event_store import EventStore
    from event_recording import attach_poly_recorder, attach_binance_recorder
    
    store = EventStore("historical_data")
    
    poly_ws = PolymarketBookWS()
    attach_poly_recorder(poly_ws, store)   # monkey-patches _on_message
    
    multi_ws = MultiAssetBinanceTradeWS(TRACKED_ASSETS)
    attach_binance_recorder(multi_ws, store)  # monkey-patches _on_message
    
    # Everything else unchanged — existing aggregation still works,
    # but now raw events also flow to disk.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from event_store import EventStore
    from ws_clients import MultiAssetBinanceTradeWS, PolymarketBookWS

logger = logging.getLogger(__name__)


def _ts_us_now() -> int:
    return int(time.time() * 1_000_000)


def _ts_us_from_dt(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def _parse_poly_levels(levels_raw: list | None) -> tuple[list[float], list[float]]:
    """Parse Polymarket book levels into separate price/size arrays."""
    if not levels_raw:
        return [], []
    prices, sizes = [], []
    for lv in levels_raw:
        if isinstance(lv, dict):
            p = lv.get("price")
            s = lv.get("size") or lv.get("quantity") or 0
        elif isinstance(lv, (list, tuple)) and len(lv) >= 2:
            p, s = lv[0], lv[1]
        else:
            continue
        try:
            prices.append(float(p))
            sizes.append(float(s or 0))
        except (TypeError, ValueError):
            continue
    return prices, sizes


# ─────────────────────────────────────────────────────────────────────────────
# Polymarket event recorder
# ─────────────────────────────────────────────────────────────────────────────

def attach_poly_recorder(poly_ws: "PolymarketBookWS", store: "EventStore") -> None:
    """
    Monkey-patches PolymarketBookWS._on_message to also record raw events.
    
    JSON is parsed ONCE — the parsed dict is passed to both the recorder
    and the original handler via _process_parsed().
    """
    def _recording_on_message(ws, message: str) -> None:
        recv_us = _ts_us_now()
        try:
            data = json.loads(message)
        except Exception:
            return

        # Record raw event (fast — just appends to in-memory buffer)
        try:
            _record_poly_event_parsed(data, recv_us, store, poly_ws)
        except Exception as e:
            logger.debug("Poly event recording error: %s", e)

        # Run original handler with pre-parsed data (no second json.loads)
        poly_ws._process_parsed(data)

    poly_ws._on_message = _recording_on_message
    logger.info("Polymarket event recording attached")


def _record_poly_event_parsed(
    data: Any, recv_us: int,
    store: "EventStore", poly_ws: "PolymarketBookWS",
) -> None:
    events = data if isinstance(data, list) else [data]
    for event in events:
        if not isinstance(event, dict):
            continue

        event_type_str = (
            event.get("event_type") or event.get("update_type") or ""
        ).strip().lower()

        if event_type_str == "price_change":
            _record_poly_book_update(event, recv_us, store, event_type_code=0)

        elif event_type_str in ("book", "book_snapshot"):
            _record_poly_book_update(event, recv_us, store, event_type_code=1)

        elif event_type_str == "last_trade_price":
            _record_poly_trade(event, recv_us, store)


def _record_poly_book_update(
    event: dict, recv_us: int, store: "EventStore", event_type_code: int,
) -> None:
    """Record a book state update (price_change or book_snapshot)."""
    token_id = (event.get("asset_id") or event.get("token_id") or "").strip()
    if not token_id:
        return

    bids_raw = event.get("bids", [])
    asks_raw = event.get("asks", [])

    bid_prices, bid_sizes = _parse_poly_levels(bids_raw)
    ask_prices, ask_sizes = _parse_poly_levels(asks_raw)

    # If only best_bid/best_ask (no full book), still record as single-level
    if not bid_prices:
        bb = event.get("best_bid")
        if bb is not None:
            try:
                bid_prices = [float(bb)]
                bid_sizes = [0.0]
            except (TypeError, ValueError):
                pass

    if not ask_prices:
        ba = event.get("best_ask")
        if ba is not None:
            try:
                ask_prices = [float(ba)]
                ask_sizes = [0.0]
            except (TypeError, ValueError):
                pass

    # Sort bids descending, asks ascending (canonical order)
    if bid_prices:
        pairs = sorted(zip(bid_prices, bid_sizes), key=lambda x: -x[0])
        bid_prices, bid_sizes = [list(x) for x in zip(*pairs)] if pairs else ([], [])
    if ask_prices:
        pairs = sorted(zip(ask_prices, ask_sizes), key=lambda x: x[0])
        ask_prices, ask_sizes = [list(x) for x in zip(*pairs)] if pairs else ([], [])

    best_bid = bid_prices[0] if bid_prices else None
    best_ask = ask_prices[0] if ask_prices else None

    store.poly_books.append(
        ts_us=recv_us,
        token_id=token_id,
        event_type=event_type_code,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        best_bid=best_bid,
        best_ask=best_ask,
    )


def _record_poly_trade(event: dict, recv_us: int, store: "EventStore") -> None:
    """Record a confirmed Polymarket trade."""
    token_id = (event.get("asset_id") or "").strip()
    tx_hash = (event.get("transaction_hash") or "").strip()

    # Timestamp from the event itself (exchange time)
    # Polymarket sends timestamps in milliseconds (13 digits)
    ts_raw = event.get("timestamp")
    if ts_raw is not None:
        try:
            ts_val = float(ts_raw)
            if ts_val > 1e15:        # already microseconds
                ts_us = int(ts_val)
            elif ts_val > 1e12:      # milliseconds → microseconds
                ts_us = int(ts_val * 1_000)
            else:                    # seconds → microseconds
                ts_us = int(ts_val * 1_000_000)
        except (TypeError, ValueError):
            ts_us = recv_us
    else:
        ts_us = recv_us

    try:
        price = float(event.get("price") or 0)
    except (TypeError, ValueError):
        price = 0.0

    try:
        size = float(event.get("size") or 0)
    except (TypeError, ValueError):
        size = 0.0

    store.poly_trades.append(
        ts_us=ts_us,
        recv_us=recv_us,
        token_id=token_id,
        price=price,
        size=size,
        tx_hash=tx_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Binance event recorder
# ─────────────────────────────────────────────────────────────────────────────

def attach_binance_recorder(
    multi_ws: "MultiAssetBinanceTradeWS", store: "EventStore",
) -> None:
    """
    Same monkey-patch approach for the Binance WS.
    JSON parsed once — shared between recorder and original handler.
    """
    def _recording_on_message(ws, message: str) -> None:
        recv_us = _ts_us_now()
        try:
            payload = json.loads(message)
        except Exception:
            return

        try:
            _record_binance_event_parsed(payload, recv_us, store, multi_ws)
        except Exception as e:
            logger.debug("Binance event recording error: %s", e)

        multi_ws._process_parsed(payload)

    multi_ws._on_message = _recording_on_message
    logger.info("Binance event recording attached")


def _record_binance_event_parsed(
    payload: dict, recv_us: int,
    store: "EventStore", multi_ws: "MultiAssetBinanceTradeWS",
) -> None:
    if not isinstance(payload, dict):
        return

    stream = (payload.get("stream") or "").strip().lower()
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return

    # Depth snapshot
    if "depth" in stream:
        _record_binance_book(data, stream, recv_us, store, multi_ws)
        return

    # Trade
    event_type = (data.get("e") or "").strip().lower()
    if event_type == "trade":
        _record_binance_trade(data, recv_us, store, multi_ws)


def _record_binance_trade(
    data: dict, recv_us: int,
    store: "EventStore", multi_ws: "MultiAssetBinanceTradeWS",
) -> None:
    symbol = (data.get("s") or "").upper()
    if symbol not in multi_ws.asset_by_symbol:
        return

    try:
        ts_ms = int(data.get("T", 0))
        ts_us = ts_ms * 1000
    except (TypeError, ValueError):
        ts_us = recv_us

    try:
        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
    except (TypeError, ValueError):
        return

    is_buy = not bool(data.get("m"))  # m=True means buyer is maker → seller is taker
    trade_id = int(data.get("t", 0))

    store.bn_trades.append(
        ts_us=ts_us,
        recv_us=recv_us,
        symbol=symbol,
        price=price,
        qty=qty,
        is_buy=is_buy,
        trade_id=trade_id,
    )


def _record_binance_book(
    data: dict, stream: str, recv_us: int,
    store: "EventStore", multi_ws: "MultiAssetBinanceTradeWS",
) -> None:
    symbol = (data.get("s") or "").upper()
    if not symbol and stream:
        symbol = stream.split("@")[0].upper()
    if symbol not in multi_ws.asset_by_symbol:
        return

    bids_raw = data.get("bids") or data.get("b", [])
    asks_raw = data.get("asks") or data.get("a", [])

    def _parse(levels):
        prices, sizes = [], []
        for lv in levels:
            try:
                p, q = float(lv[0]), float(lv[1])
                prices.append(p)
                sizes.append(q)
            except (ValueError, IndexError):
                continue
        return prices, sizes

    bid_prices, bid_sizes = _parse(bids_raw)
    ask_prices, ask_sizes = _parse(asks_raw)

    store.bn_books.append(
        ts_us=recv_us,
        symbol=symbol,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )
