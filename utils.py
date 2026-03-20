from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def floor_to_5m(dt: datetime) -> datetime:
    assert dt.tzinfo is not None, "requires timezone-aware datetime"
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def floor_to_interval_seconds(dt: datetime, interval_seconds: int) -> datetime:
    assert dt.tzinfo is not None, "requires timezone-aware datetime"
    floored_ts = int(dt.timestamp()) // interval_seconds * interval_seconds
    return datetime.fromtimestamp(floored_ts, tz=timezone.utc)


def from_unix_ts(ts_value: Any) -> Optional[datetime]:
    try:
        x = float(ts_value)
    except Exception:
        return None
    if x > 1e12:
        return datetime.fromtimestamp(x / 1000.0, tz=timezone.utc)
    return datetime.fromtimestamp(x, tz=timezone.utc)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_maybe_json_list(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def sanitize_csv_field(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()


def first_present_number(d: dict[str, Any], keys: list[str]) -> Optional[float]:
    for key in keys:
        if key in d:
            val = safe_float(d.get(key))
            if val is not None:
                return val
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fmt_float(value: Optional[float], decimals: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def fmt_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(value)


def fmt_bool(value: bool) -> str:
    return "1" if value else "0"


def bps_change(current: Optional[float], past: Optional[float]) -> Optional[float]:
    if current is None or past is None or past == 0:
        return None
    return ((current / past) - 1.0) * 10000.0


def lag_ms(now_dt: datetime, event_dt: Optional[datetime]) -> Optional[int]:
    if event_dt is None:
        return None
    delta = now_dt - event_dt
    return max(0, int(delta.total_seconds() * 1000))


def ws_health_from_lag(status: str, lag_value_ms: Optional[int],
                        stale_ms: int, dead_ms: int) -> str:
    s = normalize_text(status).lower()
    if s.startswith("error") or s in {"stopped", "closed"}:
        return "dead"
    if lag_value_ms is None:
        return "no_trades_yet"
    if lag_value_ms >= dead_ms:
        return "dead"
    if lag_value_ms >= stale_ms:
        return "stale"
    return "fresh"


def compute_time_window_metrics(now_dt: datetime, window_start_dt: datetime,
                                 window_seconds: int) -> tuple[datetime, datetime, float, float]:
    window_end_dt = window_start_dt + timedelta(seconds=window_seconds)
    remaining_seconds = max(0.0, (window_end_dt - now_dt).total_seconds())
    total_seconds = float(window_seconds)
    elapsed = total_seconds - remaining_seconds
    progress_pct = clamp((elapsed / total_seconds) * 100.0, 0.0, 100.0)
    return window_start_dt, window_end_dt, remaining_seconds, progress_pct


def parse_slug(slug: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse a slug like 'btc-updown-5m-1773300900'.
    Returns (coin, interval_minutes, window_start_unix) or (None, None, None).
    Supports any coin and any interval.
    """
    parts = slug.split("-")
    # Pattern: {coin}-updown-{interval}-{unix_ts}
    # e.g. btc-updown-5m-1773300900   → parts = ['btc','updown','5m','1773300900']
    # e.g. eth-updown-15m-1773300900  → parts = ['eth','updown','15m','1773300900']
    # coin can be multi-part? Unlikely but handle gracefully.
    try:
        updown_idx = parts.index("updown")
    except ValueError:
        return None, None, None
    if updown_idx < 1 or updown_idx + 2 >= len(parts):
        return None, None, None
    coin         = "-".join(parts[:updown_idx])
    interval_str = parts[updown_idx + 1]          # e.g. "5m", "15m", "1h"
    ts_str       = parts[updown_idx + 2]
    try:
        ts = int(ts_str)
    except ValueError:
        return None, None, None
    # Convert interval string to minutes
    interval_min: Optional[int] = None
    if interval_str.endswith("m"):
        try: interval_min = int(interval_str[:-1])
        except ValueError: pass
    elif interval_str.endswith("h"):
        try: interval_min = int(interval_str[:-1]) * 60
        except ValueError: pass
    return coin, interval_min, ts


def market_window_from_slug(market_slug: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse window start/end from any updown slug."""
    coin, interval_min, ts = parse_slug(market_slug)
    if coin is None or interval_min is None or ts is None:
        return None, None
    try:
        window_start = datetime.fromtimestamp(ts, tz=timezone.utc)
        window_end   = window_start + timedelta(minutes=interval_min)
        return window_start, window_end
    except Exception:
        return None, None


def prune_market_target_cache(cache: dict[str, Optional[float]],
                               now_dt: datetime, ttl_seconds: int) -> None:
    keep_min_ts = int((now_dt - timedelta(seconds=ttl_seconds)).timestamp())
    to_delete = []
    for key in list(cache):
        try:
            _, _, ts = parse_slug(key)
            if ts is not None and ts < keep_min_ts:
                to_delete.append(key)
        except Exception:
            pass
    for k in to_delete:
        cache.pop(k, None)


def prune_asset_target_cache(cache: dict[str, Optional[float]],
                              now_dt: datetime, ttl_seconds: int) -> None:
    keep_min_ts = int((now_dt - timedelta(seconds=ttl_seconds)).timestamp())
    to_delete = []
    for key in list(cache):
        try:
            parts = key.split(":")
            if len(parts) == 3 and parts[1] in {"15m", "1h"}:
                ts = int(parts[2])
                if ts < keep_min_ts:
                    to_delete.append(key)
            else:
                logger.warning("prune_asset_target_cache: unexpected key '%s', keeping", key)
        except Exception:
            logger.warning("prune_asset_target_cache: could not parse key '%s', keeping", key)
    for k in to_delete:
        cache.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility & probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def realized_vol_bps_per_second(close_prices: list[float]) -> Optional[float]:
    """
    Realized volatility = std of second-by-second log returns, expressed in bps/s.
    Requires at least 3 data points.

    Returns vol in bps per second. To get vol over N seconds: vol * sqrt(N).
    """
    if len(close_prices) < 3:
        return None
    log_returns = []
    for i in range(1, len(close_prices)):
        prev = close_prices[i - 1]
        curr = close_prices[i]
        if prev > 0 and curr > 0:
            log_returns.append(math.log(curr / prev) * 10000.0)  # in bps
    if len(log_returns) < 2:
        return None
    try:
        return statistics.stdev(log_returns)
    except Exception:
        return None


# Minimum time floor: avoid z-score explosion in last seconds of window
_Z_MIN_SECONDS = 30.0
# Minimum vol floor: avoid division by near-zero at window open (bps/s)
# ~0.5 bps/s = very quiet market (BTC moving ~30 bps/min)
_Z_MIN_VOL_BPS = 0.5

def z_score_binary(distance_to_target_bps: Optional[float],
                   vol_bps_per_second: Optional[float],
                   seconds_remaining: Optional[float]) -> Optional[float]:
    """
    Z-score for a binary prediction: how many standard deviations is the
    current price above/below the target, given remaining time.

    z = distance / (vol_per_second * sqrt(max(seconds_remaining, 30)))

    Floors applied:
    - seconds_remaining floored at 30s to prevent explosion near window close
    - vol_bps_per_second floored at 0.5 bps/s to prevent division by near-zero
      at window open before enough trades have been seen by the WebSocket

    This is the d1 of a Black-Scholes digital call option.
    P(UP) ≈ Φ(z)
    """
    if distance_to_target_bps is None or vol_bps_per_second is None:
        return None
    if seconds_remaining is None or seconds_remaining <= 0:
        # Window over: sign of distance determines outcome
        if distance_to_target_bps > 0:
            return 10.0   # essentially certain UP
        elif distance_to_target_bps < 0:
            return -10.0  # essentially certain DOWN
        return 0.0
    # Apply floors
    t_eff   = max(seconds_remaining, _Z_MIN_SECONDS)
    vol_eff = max(vol_bps_per_second, _Z_MIN_VOL_BPS)
    expected_vol = vol_eff * math.sqrt(t_eff)
    return distance_to_target_bps / expected_vol


def analytical_prob_up(z: Optional[float]) -> Optional[float]:
    """P(UP) = Φ(z) — standard normal CDF."""
    if z is None:
        return None
    # Approximation of Φ using math.erfc for stdlib compatibility
    return 0.5 * math.erfc(-z / math.sqrt(2))


def mid_velocity_bps_per_second(mid_now: Optional[float],
                                 mid_past: Optional[float],
                                 elapsed_seconds: float) -> Optional[float]:
    """Rate of change of Polymarket mid in bps per second."""
    if mid_now is None or mid_past is None or elapsed_seconds <= 0:
        return None
    return (mid_now - mid_past) / elapsed_seconds * 10000.0  # bps/s


def cross_asset_agreement(btc_return: Optional[float],
                           other_returns: list[Optional[float]]) -> Optional[float]:
    """
    Fraction of other assets moving in the same direction as BTC.
    Returns 0.0–1.0 (or None if btc_return is None).
    """
    if btc_return is None:
        return None
    btc_dir = 1 if btc_return > 0 else (-1 if btc_return < 0 else 0)
    if btc_dir == 0:
        return None
    valid = [r for r in other_returns if r is not None]
    if not valid:
        return None
    agreements = sum(1 for r in valid if (r > 0) == (btc_dir > 0))
    return agreements / len(valid)


def cross_asset_momentum(returns: list[Optional[float]]) -> Optional[float]:
    """
    Average normalized return across a list of assets.
    Each return is already in bps — we just average non-None values.
    """
    valid = [r for r in returns if r is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)
