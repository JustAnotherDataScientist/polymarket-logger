"""
paper_trader.py — Paper trading module for BTC C3 strategy.

Runs alongside the logger, uses live data, logs every decision.
Does NOT place real orders — just records what it would have done.

Strategy: BTC C3 (fi + dist>=+5bps)
  - 15m windows only
  - At t+30s: check flow > 0.2, imbalance > 0, distance >= +5bps
  - If all 3 → "BUY UP" at ask price (with fee)
  - At window end → record settlement and virtual PnL

Integration:
  paper = PaperTrader()
  # In main loop, after computing market_snaps:
  paper.tick(market_snaps, btc_tracker.poly_ws, now_dt)

Output: ~/paper_trades.csv with one row per decision point
"""

from __future__ import annotations
import csv
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Polymarket fee model
# ─────────────────────────────────────────────────────────────────────

def taker_fee_rate(price: float) -> float:
    p = max(0.01, min(0.99, price))
    return 0.072 * p * (1 - p)


# ─────────────────────────────────────────────────────────────────────
# Strategy parameters
# ─────────────────────────────────────────────────────────────────────

STRATEGY_NAME = "BTC_C3"
COIN = "btc"
INTERVAL = 15
OBS_SECONDS = 30          # observe for 30s, then decide
FLOW_THRESHOLD = 0.20     # poly flow (buy-sell)/total > 0.20
IMB_THRESHOLD = 0.0       # book imbalance > 0
DIST_MIN_BPS = 5.0        # BN distance >= +5 bps
DIST_MAX_BPS = 999.0      # no upper cap (set to e.g. 15 to test)
SIZE_USDC = 2.0           # paper trade size (for PnL calculation)

CSV_PATH = os.path.expanduser("~/paper_trades.csv")
CSV_FIELDS = [
    "timestamp_utc", "slug", "window_start", "window_end",
    "decision_at_pct", "decision",
    "bn_price", "target_price", "distance_bps",
    "up_ask", "up_mid", "up_bid",
    "depth_imbalance", "poly_flow_30s", "poly_n_trades_30s",
    "entry_price_with_fee", "size_usdc",
    "outcome", "settle_price", "pnl",
    "cum_pnl", "n_trades", "win_rate",
]


class PaperTrader:
    """
    Paper trading engine for BTC C3.

    Call .tick() every second from the main loop.
    It will automatically detect window transitions and make decisions.
    """

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        self._current_slug: Optional[str] = None
        self._decision_made = False
        self._entry_price: Optional[float] = None
        self._entry_row: Optional[dict] = None
        self._cum_pnl = 0.0
        self._n_trades = 0
        self._n_wins = 0

        # Load existing state from CSV
        self._load_state()

        # Ensure CSV exists with header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

        logger.info("PaperTrader[%s] initialized | cum_pnl=$%.1f | trades=%d | csv=%s",
                     STRATEGY_NAME, self._cum_pnl, self._n_trades, self.csv_path)

    def _load_state(self):
        """Resume state from existing CSV if any."""
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("pnl") and row["pnl"] != "":
                        try:
                            self._cum_pnl = float(row["cum_pnl"])
                            self._n_trades = int(row["n_trades"])
                            wr = float(row["win_rate"])
                            self._n_wins = round(self._n_trades * wr / 100)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.warning("PaperTrader: could not load state: %s", e)

    def tick(self, market_snaps: dict, btc_15m_poly_ws: Any, now_dt: datetime) -> None:
        """
        Called every tick (~1s) from the main loop.

        Args:
            market_snaps: dict of market snapshots from logger_core
            btc_15m_poly_ws: the PolymarketBookWS instance for btc-15m
            now_dt: current UTC datetime
        """
        snap = market_snaps.get("btc-15m")
        if snap is None or not snap.get("market_found"):
            return

        slug = snap.get("slug") or ""
        progress = snap.get("window_progress") or 0
        remaining = snap.get("seconds_remaining") or 0

        # ── New window detected ──────────────────────────────────
        if slug != self._current_slug:
            # Settle previous window if we had an open position
            if self._entry_price is not None and self._entry_row is not None:
                self._settle_previous(market_snaps)

            self._current_slug = slug
            self._decision_made = False
            self._entry_price = None
            self._entry_row = None

        # ── Decision point: ~30s elapsed (progress ~33%) ─────────
        if not self._decision_made and progress is not None:
            elapsed_s = (1.0 - remaining / (INTERVAL * 60)) * INTERVAL * 60 if remaining else 0
            if elapsed_s >= OBS_SECONDS:
                self._make_decision(snap, btc_15m_poly_ws, now_dt)
                self._decision_made = True

        # ── Settlement: window ending (progress > 99%) ───────────
        if self._entry_price is not None and remaining is not None and remaining < 2:
            outcome = snap.get("outcome")
            if outcome is not None and outcome in (0, 1, "0", "1"):
                self._settle(snap, now_dt)

    def _compute_poly_flow(self, poly_ws: Any, now_dt: datetime) -> tuple[float, int]:
        """
        Compute poly flow from trade_events over the last 30s.
        Returns (flow_ratio, n_trades).
        Flow = (buy_volume - sell_volume) / total_volume.
        Buy if trade price >= mid, sell otherwise.
        """
        with poly_ws._lock:
            mid = None
            if poly_ws._bids and poly_ws._asks:
                best_bid = poly_ws._bids[0][0] if poly_ws._bids else 0
                best_ask = poly_ws._asks[0][0] if poly_ws._asks else 0
                if best_bid > 0 and best_ask > 0:
                    mid = (best_bid + best_ask) / 2

            if mid is None or mid <= 0:
                return 0.0, 0

            cutoff = now_dt - timedelta(seconds=30)
            buy_vol = 0.0
            sell_vol = 0.0
            n = 0

            for item in poly_ws.trade_events:
                trade_dt = item[0]
                size = item[1]
                asset_id = item[3]
                price = item[4] if len(item) > 4 else 0.0

                # Skip price_change events that have no real trade price
                # (only last_trade_price events carry an actual price > 0)
                if price <= 0:
                    continue

                if trade_dt < cutoff:
                    continue

                # Only count UP token trades
                if poly_ws.current_asset_id and asset_id and asset_id != poly_ws.current_asset_id:
                    continue

                n += 1
                if price >= mid:
                    buy_vol += size
                else:
                    sell_vol += size

            # Count skipped events for diagnostics
            n_skipped_price0 = sum(
                1 for item in poly_ws.trade_events
                if item[0] >= cutoff and (item[4] if len(item) > 4 else 0.0) <= 0
            )

            total = buy_vol + sell_vol
            flow = (buy_vol - sell_vol) / total if total > 0 else 0.0
            logger.info(
                "PAPER[%s] flow_diag: mid=%.4f | deque=%d | real_trades=%d "
                "skipped_p0=%d | buy=%.1f sell=%.1f → flow=%+.3f",
                STRATEGY_NAME, mid, len(poly_ws.trade_events), n,
                n_skipped_price0, buy_vol, sell_vol, flow,
            )
            return flow, n

    def _make_decision(self, snap: dict, poly_ws: Any, now_dt: datetime) -> None:
        """Evaluate the 3 conditions and log the decision."""
        distance = snap.get("distance_bps")
        imbalance = snap.get("up_depth_imbalance")
        up_ask = snap.get("up_ask")
        up_mid = snap.get("up_mid")
        up_bid = snap.get("up_bid")
        bn_price = snap.get("coin_price")
        target = snap.get("target_price")

        # Compute flow
        flow, n_trades = self._compute_poly_flow(poly_ws, now_dt)

        # Check conditions
        cond_dist = distance is not None and DIST_MIN_BPS <= distance <= DIST_MAX_BPS
        cond_flow = flow >= FLOW_THRESHOLD
        cond_imb = imbalance is not None and imbalance > IMB_THRESHOLD

        decision = "BUY_UP" if (cond_dist and cond_flow and cond_imb) else "SKIP"

        # Entry price with fee
        entry_with_fee = None
        if decision == "BUY_UP" and up_ask is not None and up_ask > 0:
            entry_with_fee = up_ask * (1 + taker_fee_rate(up_ask))

        elapsed_pct = snap.get("window_progress")
        if elapsed_pct is None:
            remaining = snap.get("seconds_remaining") or 0
            elapsed_pct = (1 - remaining / (INTERVAL * 60)) * 100

        row = {
            "timestamp_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "slug": snap.get("slug", ""),
            "window_start": snap.get("window_start", ""),
            "window_end": snap.get("window_end", ""),
            "decision_at_pct": f"{elapsed_pct:.1f}",
            "decision": decision,
            "bn_price": f"{bn_price:.2f}" if bn_price else "",
            "target_price": f"{target:.2f}" if target else "",
            "distance_bps": f"{distance:+.1f}" if distance is not None else "",
            "up_ask": f"{up_ask:.4f}" if up_ask else "",
            "up_mid": f"{up_mid:.4f}" if up_mid else "",
            "up_bid": f"{up_bid:.4f}" if up_bid else "",
            "depth_imbalance": f"{imbalance:+.3f}" if imbalance is not None else "",
            "poly_flow_30s": f"{flow:+.3f}",
            "poly_n_trades_30s": str(n_trades),
            "entry_price_with_fee": f"{entry_with_fee:.4f}" if entry_with_fee else "",
            "size_usdc": f"{SIZE_USDC:.2f}" if decision == "BUY_UP" else "",
            "outcome": "",
            "settle_price": "",
            "pnl": "",
            "cum_pnl": "",
            "n_trades": "",
            "win_rate": "",
        }

        if decision == "BUY_UP" and entry_with_fee:
            self._entry_price = entry_with_fee
            self._entry_row = row
            logger.info(
                "PAPER[%s] BUY_UP slug=%s entry=%.4f dist=%+.1f flow=%+.2f imb=%+.3f",
                STRATEGY_NAME, snap.get("slug", "")[-20:],
                entry_with_fee, distance or 0, flow, imbalance or 0,
            )
        else:
            # Log SKIP immediately (no settlement to wait for)
            self._append_csv(row)
            reason = []
            if not cond_dist:
                reason.append(f"dist={distance:+.1f}" if distance is not None else "dist=None")
            if not cond_flow:
                reason.append(f"flow={flow:+.2f}")
            if not cond_imb:
                reason.append(f"imb={imbalance:+.3f}" if imbalance is not None else "imb=None")
            logger.info(
                "PAPER[%s] SKIP slug=%s reason=[%s]",
                STRATEGY_NAME, snap.get("slug", "")[-20:], ", ".join(reason),
            )

    def _settle(self, snap: dict, now_dt: datetime) -> None:
        """Record settlement outcome for an open position."""
        if self._entry_price is None or self._entry_row is None:
            return

        outcome = snap.get("outcome")
        if outcome is None:
            return

        outcome_int = int(outcome)
        settle = 1.0 if outcome_int == 1 else 0.0  # UP wins → settle=1
        pnl = (settle / self._entry_price - 1.0) * SIZE_USDC

        self._cum_pnl += pnl
        self._n_trades += 1
        if pnl > 0:
            self._n_wins += 1
        wr = self._n_wins / self._n_trades * 100 if self._n_trades > 0 else 0

        row = self._entry_row
        row["outcome"] = str(outcome_int)
        row["settle_price"] = f"{settle:.1f}"
        row["pnl"] = f"{pnl:+.2f}"
        row["cum_pnl"] = f"{self._cum_pnl:+.2f}"
        row["n_trades"] = str(self._n_trades)
        row["win_rate"] = f"{wr:.1f}"

        self._append_csv(row)

        win_str = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "PAPER[%s] %s pnl=$%+.2f cum=$%+.1f wr=%.0f%% (%d trades) slug=%s",
            STRATEGY_NAME, win_str, pnl, self._cum_pnl, wr,
            self._n_trades, snap.get("slug", "")[-20:],
        )

        self._entry_price = None
        self._entry_row = None

    def _settle_previous(self, market_snaps: dict) -> None:
        """Try to settle previous window — outcome might not be available yet."""
        # If the previous window's outcome wasn't captured, log it as MISSED
        if self._entry_row is not None:
            row = self._entry_row
            row["outcome"] = "MISSED"
            row["pnl"] = ""
            self._append_csv(row)
            logger.warning("PAPER[%s] MISSED settlement for slug=%s",
                           STRATEGY_NAME, row.get("slug", ""))
        self._entry_price = None
        self._entry_row = None

    def _append_csv(self, row: dict) -> None:
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)
        except Exception as e:
            logger.error("PaperTrader CSV write error: %s", e)
