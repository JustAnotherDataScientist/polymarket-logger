from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import requests

from config import BINANCE_BASE, CLOB_BASE, GAMMA_BASE, REQUEST_TIMEOUT
from utils import (
    dt_to_ms,
    first_present_number,
    floor_to_5m,
    market_window_from_slug,
    normalize_text,
    parse_maybe_json_list,
    safe_float,
)

logger = logging.getLogger(__name__)


class ApiClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 polymarket-multi-asset-logger",
                "Accept": "application/json",
            }
        )

    def get_json(self, url: str, params: Optional[dict[str, Any]] = None) -> Any:
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError:
            # SSL glitches happen with Polymarket CLOB — reset session and retry once
            logger.debug("SSL error on %s — resetting session and retrying", url)
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 polymarket-multi-asset-logger",
                "Accept": "application/json",
            })
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()

    def get_spot_price(self, symbol: str) -> float:
        url = f"{BINANCE_BASE}/api/v3/ticker/price"
        data = self.get_json(url, params={"symbol": symbol})
        price = safe_float(data.get("price"))
        if price is None:
            raise RuntimeError(f"Could not parse price for {symbol}: {data}")
        return price

    def get_symbol_target_price_at_window_start(
        self,
        symbol: str,
        window_start: datetime,
    ) -> Optional[float]:
        start_ms = dt_to_ms(window_start)
        end_ms = start_ms + 60_000

        try:
            url = f"{BINANCE_BASE}/api/v3/aggTrades"
            data = self.get_json(
                url,
                params={
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            if isinstance(data, list) and data:
                price = safe_float(data[0].get("p"))
                if price is not None:
                    return price
        except Exception as exc:
            logger.debug("aggTrades fallback for %s: %s", symbol, exc)

        try:
            url = f"{BINANCE_BASE}/api/v3/klines"
            data = self.get_json(
                url,
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": start_ms,
                    "limit": 1,
                },
            )
            if isinstance(data, list) and data and isinstance(data[0], list) and len(data[0]) >= 2:
                price = safe_float(data[0][1])
                if price is not None:
                    return price
        except Exception as exc:
            logger.debug("klines fallback for %s: %s", symbol, exc)

        return None

    def get_market_by_slug(self, slug: str) -> Optional[dict[str, Any]]:
        url = f"{GAMMA_BASE}/markets/slug/{slug}"
        try:
            data = self.get_json(url)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                return None
            raise

        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        return None

    def build_candidate_slugs(self, coin: str, interval_min: int,
                               now_dt: datetime) -> list[str]:
        """
        Build candidate slugs for a given coin + interval.
        Tries current window + one window before/after in case of clock drift.
        e.g. coin='btc', interval_min=5 → ['btc-updown-5m-1773300900', ...]
        """
        interval_sec = interval_min * 60
        # Floor to interval boundary
        base_ts = int(now_dt.timestamp()) // interval_sec * interval_sec
        candidates = []
        seen: set[str] = set()
        for offset in [0, -interval_sec, interval_sec, -2 * interval_sec]:
            ts  = base_ts + offset
            slug = f"{coin}-updown-{interval_min}m-{ts}"
            if slug not in seen:
                seen.add(slug)
                candidates.append(slug)
        return candidates

    def find_active_market(self, coin: str, interval_min: int,
                           now_dt: datetime) -> Optional[dict[str, Any]]:
        """Find the currently active Polymarket updown market for coin+interval."""
        best: Optional[dict[str, Any]] = None
        best_start: Optional[datetime] = None

        for slug in self.build_candidate_slugs(coin, interval_min, now_dt):
            market = self.get_market_by_slug(slug)
            if market is None:
                continue
            market_slug = normalize_text(market.get("slug"))
            window_start, window_end = market_window_from_slug(market_slug)
            if window_start is None or window_end is None:
                continue
            if not (window_start <= now_dt < window_end):
                continue
            if best is None or best_start is None or window_start > best_start:
                best       = market
                best_start = window_start
        return best

    def find_next_market(self, coin: str, interval_min: int,
                         current_window_end: datetime) -> Optional[dict[str, Any]]:
        """Pre-fetch the next market starting at current_window_end.
        Slug is deterministic: coin-updown-Xm-{unix_ts}."""
        ts = int(current_window_end.timestamp())
        slug = f"{coin}-updown-{interval_min}m-{ts}"
        return self.get_market_by_slug(slug)

    # Keep legacy method for backward compatibility
    def find_latest_btc_5m_market(self, now_dt: datetime) -> Optional[dict[str, Any]]:
        return self.find_active_market("btc", 5, now_dt)

    def get_up_token_id(self, market: dict[str, Any]) -> Optional[str]:
        outcomes   = parse_maybe_json_list(market.get("outcomes"))
        token_ids  = parse_maybe_json_list(market.get("clobTokenIds"))

        if not isinstance(outcomes, list) or not isinstance(token_ids, list):
            return None
        if len(outcomes) != len(token_ids):
            return None

        for index, outcome in enumerate(outcomes):
            if normalize_text(outcome).lower() == "up":
                token_id = normalize_text(token_ids[index])
                return token_id or None

        return None

    def get_down_token_id(self, market: dict[str, Any]) -> Optional[str]:
        """Returns the DOWN token asset_id for a market."""
        outcomes   = parse_maybe_json_list(market.get("outcomes"))
        token_ids  = parse_maybe_json_list(market.get("clobTokenIds"))

        if not isinstance(outcomes, list) or not isinstance(token_ids, list):
            return None
        if len(outcomes) != len(token_ids):
            return None

        for index, outcome in enumerate(outcomes):
            if normalize_text(outcome).lower() == "down":
                token_id = normalize_text(token_ids[index])
                return token_id or None

        return None

    def get_condition_id(self, market: dict[str, Any]) -> str:
        for key in ["conditionId", "condition_id", "market", "marketId", "id"]:
            text = normalize_text(market.get(key))
            if text.startswith("0x") and len(text) >= 10:
                return text

        event_obj = market.get("event")
        if isinstance(event_obj, dict):
            for key in ["conditionId", "condition_id", "market", "marketId", "id"]:
                text = normalize_text(event_obj.get(key))
                if text.startswith("0x") and len(text) >= 10:
                    return text

        return ""

    def get_order_book(self, token_id: str) -> Optional[dict[str, Any]]:
        url = f"{CLOB_BASE}/book"
        try:
            data = self.get_json(url, params={"token_id": token_id})
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                return None
            raise

        if isinstance(data, dict):
            return data
        return None

    def get_up_book_metrics(self, token_id: str) -> dict[str, Optional[float]]:
        result: dict[str, Optional[float]] = {
            "up_bid": None, "up_ask": None, "up_mid": None,
            "up_spread": None, "up_spread_bps_mid": None,
            "up_best_bid_size": None, "up_best_ask_size": None,
            "up_bid_depth_top5": None, "up_ask_depth_top5": None,
            "up_depth_imbalance_top5": None,
        }

        if not token_id:
            return result

        book = self.get_order_book(token_id)
        if not book:
            return result

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        bid_levels: list[tuple[float, float]] = []
        ask_levels: list[tuple[float, float]] = []

        if isinstance(bids, list):
            for level in bids:
                if not isinstance(level, dict):
                    continue
                price = first_present_number(level, ["price"])
                size  = first_present_number(level, ["size", "quantity"])
                if price is not None:
                    bid_levels.append((price, 0.0 if size is None else size))

        if isinstance(asks, list):
            for level in asks:
                if not isinstance(level, dict):
                    continue
                price = first_present_number(level, ["price"])
                size  = first_present_number(level, ["size", "quantity"])
                if price is not None:
                    ask_levels.append((price, 0.0 if size is None else size))

        bid_levels.sort(key=lambda x: x[0], reverse=True)
        ask_levels.sort(key=lambda x: x[0])

        best_bid      = bid_levels[0][0] if bid_levels else None
        best_ask      = ask_levels[0][0] if ask_levels else None
        best_bid_size = bid_levels[0][1] if bid_levels else None
        best_ask_size = ask_levels[0][1] if ask_levels else None

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_ask is not None:
            mid = best_ask
        elif best_bid is not None:
            mid = best_bid

        spread = spread_bps_mid = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            if mid and mid != 0:
                spread_bps_mid = spread / mid * 10000.0

        bid_depth_top5 = sum(s for _, s in bid_levels[:5]) if bid_levels else None
        ask_depth_top5 = sum(s for _, s in ask_levels[:5]) if ask_levels else None

        depth_imbalance_top5 = None
        if bid_depth_top5 is not None and ask_depth_top5 is not None:
            denom = bid_depth_top5 + ask_depth_top5
            if denom > 0:
                depth_imbalance_top5 = (bid_depth_top5 - ask_depth_top5) / denom

        result["up_bid"]                = best_bid
        result["up_ask"]                = best_ask
        result["up_mid"]                = mid
        result["up_spread"]             = spread
        result["up_spread_bps_mid"]     = spread_bps_mid
        result["up_best_bid_size"]      = best_bid_size
        result["up_best_ask_size"]      = best_ask_size
        result["up_bid_depth_top5"]     = bid_depth_top5
        result["up_ask_depth_top5"]     = ask_depth_top5
        result["up_depth_imbalance_top5"] = depth_imbalance_top5
        return result
