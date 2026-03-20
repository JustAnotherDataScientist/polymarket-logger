"""
book_replay.py — Reconstruct Polymarket order book at any point in time
                  and simulate order execution with precise slippage.

This is the core of the backtesting engine. It replays recorded events
in timestamp order to reconstruct the exact book state, then provides
execution simulation for both taker and maker orders.

Usage:
    from book_replay import BookReplayEngine
    
    engine = BookReplayEngine("historical_data/events")
    
    # Load a day of data for a specific token
    engine.load("2026-03-20", token_id="0xabc...")
    
    # Iterate through time, get book state at each point
    for snap in engine.iter_snapshots(step_us=250_000):  # 250ms steps
        book = snap.book
        ts   = snap.ts_us
        
        # Your strategy logic here
        if should_buy(book):
            fill = snap.simulate_taker_buy(size_usdc=100.0)
            # fill.avg_price, fill.total_cost, fill.levels_consumed
        
        if should_post_limit(book):
            order_id = snap.post_limit_buy(price=0.52, size_usdc=200.0)
        
        # Check if any pending limit orders were filled
        for filled in snap.check_fills():
            # filled.order_id, filled.fill_price, filled.fill_size, filled.fill_ts
            pass

Design choices:
  - Book state is reconstructed from poly_books events (every price_change
    and book_snapshot). Between events, the book is assumed unchanged.
  - Taker fills: walk the book levels, consuming liquidity at each price.
    This gives exact average fill price including slippage.
  - Maker fills: conservative model — your order is LAST in queue at its
    price level. A fill happens only when cumulative trade volume at your
    price level exceeds the size that was ahead of you in the queue.
  - Fees: Polymarket taker fee (currently 0 for most markets, but
    configurable) applied to fills.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

TAKER_FEE_BPS = 0.0    # Polymarket currently charges 0 taker fee on most markets
MAKER_FEE_BPS = 0.0    # adjust if Polymarket changes fee structure


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BookLevel:
    price: float
    size: float       # USDC


@dataclass
class BookState:
    """Snapshot of the order book at a point in time."""
    ts_us: int
    token_id: str
    bids: list[BookLevel]          # sorted price descending (best first)
    asks: list[BookLevel]          # sorted price ascending (best first)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        if self.mid and self.best_bid is not None and self.best_ask is not None:
            return (self.best_ask - self.best_bid) / self.mid * 10_000
        return None

    def bid_depth(self, n_levels: int = 5) -> float:
        return sum(lv.size for lv in self.bids[:n_levels])

    def ask_depth(self, n_levels: int = 5) -> float:
        return sum(lv.size for lv in self.asks[:n_levels])

    def bid_depth_at_price(self, price: float) -> float:
        """Total size on the bid side at exactly this price."""
        return sum(lv.size for lv in self.bids if abs(lv.price - price) < 1e-8)

    def ask_depth_at_price(self, price: float) -> float:
        """Total size on the ask side at exactly this price."""
        return sum(lv.size for lv in self.asks if abs(lv.price - price) < 1e-8)


@dataclass
class TakerFill:
    """Result of simulating a taker order."""
    side: str               # "buy" or "sell"
    requested_size: float   # USDC requested
    filled_size: float      # USDC actually filled (may be less if book too thin)
    avg_price: float        # volume-weighted average fill price
    total_cost: float       # total USDC spent (buy) or received (sell)
    fee: float              # fee in USDC
    levels_consumed: int    # how many book levels were touched
    worst_price: float      # price of the last fill level
    fully_filled: bool


@dataclass
class PendingLimitOrder:
    order_id: int
    side: str               # "buy" or "sell"
    price: float
    size: float             # USDC remaining
    original_size: float
    post_ts_us: int
    queue_behind: float     # USDC of orders ahead of us at this level
    filled_size: float = 0.0
    filled: bool = False
    fill_ts_us: Optional[int] = None


@dataclass
class LimitFillEvent:
    order_id: int
    fill_price: float
    fill_size: float
    fill_ts_us: int
    remaining: float


# ─────────────────────────────────────────────────────────────────────────────
# Execution simulator
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionSimulator:
    """
    Simulates order execution against the book state.
    Tracks pending limit orders and checks for fills on each trade event.
    """

    def __init__(self) -> None:
        self._next_order_id = 1
        self.pending_orders: list[PendingLimitOrder] = []

    def simulate_taker_buy(self, book: BookState, size_usdc: float) -> TakerFill:
        """
        Simulate buying `size_usdc` worth of tokens by lifting asks.
        
        Walks the ask side of the book, consuming liquidity at each level
        until the requested size is filled or the book is exhausted.
        
        Returns exact average fill price including slippage.
        """
        remaining = size_usdc
        total_cost = 0.0
        levels_consumed = 0
        worst_price = 0.0

        for level in book.asks:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.size)
            total_cost += fill_at_level  # on Polymarket, size IS the USDC cost
            remaining -= fill_at_level
            worst_price = level.price
            levels_consumed += 1

        filled = size_usdc - remaining
        fee = filled * TAKER_FEE_BPS / 10_000
        avg_price = worst_price if levels_consumed <= 1 else (
            # Weighted average across levels
            sum(
                min(remaining_at_step, lv.size) * lv.price
                for remaining_at_step, lv in self._walk_levels(book.asks, size_usdc)
            ) / filled if filled > 0 else 0.0
        )

        # Simpler, correct calculation:
        avg_price = self._calc_avg_price(book.asks, size_usdc)

        return TakerFill(
            side="buy",
            requested_size=size_usdc,
            filled_size=filled,
            avg_price=avg_price,
            total_cost=filled + fee,
            fee=fee,
            levels_consumed=levels_consumed,
            worst_price=worst_price,
            fully_filled=(remaining <= 0),
        )

    def simulate_taker_sell(self, book: BookState, size_usdc: float) -> TakerFill:
        """
        Simulate selling `size_usdc` worth of tokens by hitting bids.
        Walks the bid side, same logic as taker_buy but reversed.
        """
        remaining = size_usdc
        levels_consumed = 0
        worst_price = 0.0

        for level in book.bids:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.size)
            remaining -= fill_at_level
            worst_price = level.price
            levels_consumed += 1

        filled = size_usdc - remaining
        fee = filled * TAKER_FEE_BPS / 10_000
        avg_price = self._calc_avg_price(book.bids, size_usdc)

        return TakerFill(
            side="sell",
            requested_size=size_usdc,
            filled_size=filled,
            avg_price=avg_price,
            total_cost=filled - fee,
            fee=fee,
            levels_consumed=levels_consumed,
            worst_price=worst_price,
            fully_filled=(remaining <= 0),
        )

    def post_limit_buy(
        self, book: BookState, price: float, size_usdc: float, ts_us: int,
    ) -> PendingLimitOrder:
        """
        Post a limit buy order at `price`. 
        
        Conservative assumption: our order goes to the BACK of the queue
        at this price level. We record how much size is ahead of us.
        """
        queue_behind = book.bid_depth_at_price(price)

        order = PendingLimitOrder(
            order_id=self._next_order_id,
            side="buy",
            price=price,
            size=size_usdc,
            original_size=size_usdc,
            post_ts_us=ts_us,
            queue_behind=queue_behind,
        )
        self._next_order_id += 1
        self.pending_orders.append(order)
        return order

    def post_limit_sell(
        self, book: BookState, price: float, size_usdc: float, ts_us: int,
    ) -> PendingLimitOrder:
        """Post a limit sell order. Same queue model as limit buy."""
        queue_behind = book.ask_depth_at_price(price)

        order = PendingLimitOrder(
            order_id=self._next_order_id,
            side="sell",
            price=price,
            size=size_usdc,
            original_size=size_usdc,
            post_ts_us=ts_us,
            queue_behind=queue_behind,
        )
        self._next_order_id += 1
        self.pending_orders.append(order)
        return order

    def cancel_order(self, order_id: int) -> bool:
        for i, o in enumerate(self.pending_orders):
            if o.order_id == order_id and not o.filled:
                self.pending_orders.pop(i)
                return True
        return False

    def process_trade(
        self, trade_price: float, trade_size: float, trade_ts_us: int,
    ) -> list[LimitFillEvent]:
        """
        Called for every recorded trade. Checks if any pending limit orders
        would have been filled by this trade.
        
        Fill model (conservative):
          - For buy limit at price P: filled when a trade occurs at price ≤ P
            AND cumulative volume at that level exceeds our queue position.
          - Queue position starts at the depth that was ahead of us when we
            posted, and decreases as trades consume that level.
        """
        fills: list[LimitFillEvent] = []
        still_pending = []

        for order in self.pending_orders:
            if order.filled:
                continue

            triggered = (
                (order.side == "buy" and trade_price <= order.price) or
                (order.side == "sell" and trade_price >= order.price)
            )

            if not triggered:
                still_pending.append(order)
                continue

            # Reduce queue position by trade size
            order.queue_behind = max(0.0, order.queue_behind - trade_size)

            if order.queue_behind <= 0:
                # We're at the front — our order gets filled
                fill_size = min(order.size, trade_size)
                order.filled_size += fill_size
                order.size -= fill_size

                fills.append(LimitFillEvent(
                    order_id=order.order_id,
                    fill_price=order.price,
                    fill_size=fill_size,
                    fill_ts_us=trade_ts_us,
                    remaining=order.size,
                ))

                if order.size <= 0.01:  # fully filled (with float tolerance)
                    order.filled = True
                    order.fill_ts_us = trade_ts_us
                else:
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        return fills

    def _calc_avg_price(self, levels: list[BookLevel], size: float) -> float:
        """Walk levels and compute VWAP fill price."""
        remaining = size
        weighted_sum = 0.0
        total_filled = 0.0
        for lv in levels:
            if remaining <= 0:
                break
            fill = min(remaining, lv.size)
            weighted_sum += fill * lv.price
            total_filled += fill
            remaining -= fill
        return weighted_sum / total_filled if total_filled > 0 else 0.0

    @staticmethod
    def _walk_levels(levels, size):
        remaining = size
        for lv in levels:
            if remaining <= 0:
                break
            yield remaining, lv
            remaining -= min(remaining, lv.size)


# ─────────────────────────────────────────────────────────────────────────────
# Book replay engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplaySnapshot:
    """One point in time during replay. Provides book state + execution API."""
    ts_us: int
    book: BookState
    _sim: ExecutionSimulator
    # Binance context at this time (for feature computation)
    bn_price: Optional[float] = None
    bn_book: Optional[dict] = None

    def simulate_taker_buy(self, size_usdc: float) -> TakerFill:
        return self._sim.simulate_taker_buy(self.book, size_usdc)

    def simulate_taker_sell(self, size_usdc: float) -> TakerFill:
        return self._sim.simulate_taker_sell(self.book, size_usdc)

    def post_limit_buy(self, price: float, size_usdc: float) -> PendingLimitOrder:
        return self._sim.post_limit_buy(self.book, price, size_usdc, self.ts_us)

    def post_limit_sell(self, price: float, size_usdc: float) -> PendingLimitOrder:
        return self._sim.post_limit_sell(self.book, price, size_usdc, self.ts_us)

    def cancel_order(self, order_id: int) -> bool:
        return self._sim.cancel_order(order_id)

    @property
    def pending_orders(self) -> list[PendingLimitOrder]:
        return self._sim.pending_orders


class BookReplayEngine:
    """
    Loads recorded events and replays them to reconstruct book state
    at arbitrary points in time.
    
    Usage:
        engine = BookReplayEngine("historical_data/events")
        engine.load("2026-03-20", token_id="0xabc123...")
        
        for snap in engine.iter_snapshots(step_us=250_000):
            # snap.book has the full order book at this timestamp
            # snap.simulate_taker_buy(100) to test fills
            pass
    """

    def __init__(self, events_dir: str) -> None:
        self.events_dir = Path(events_dir)

    def load(self, date_str: str, token_id: str) -> None:
        """Load all events for a given date and token."""
        self.token_id = token_id

        # Load poly book events
        books_dir = self.events_dir / "poly_books"
        book_files = sorted(
            list(books_dir.glob(f"{date_str}_*.parquet")) +
            list(books_dir.glob(f"{date_str}.parquet"))
        )
        if book_files:
            import pyarrow.compute as pc
            tables = [pq.read_table(f) for f in book_files]
            all_books = _concat_and_sort(tables, "ts_us")
            # Filter to our token
            mask = pc.equal(all_books.column("token_id"), token_id)
            self._book_events = all_books.filter(mask)
        else:
            self._book_events = None

        # Load poly trades
        trades_dir = self.events_dir / "poly_trades"
        trade_files = sorted(
            list(trades_dir.glob(f"{date_str}_*.parquet")) +
            list(trades_dir.glob(f"{date_str}.parquet"))
        )
        if trade_files:
            import pyarrow.compute as pc
            tables = [pq.read_table(f) for f in trade_files]
            all_trades = _concat_and_sort(tables, "ts_us")
            mask = pc.equal(all_trades.column("token_id"), token_id)
            self._trade_events = all_trades.filter(mask)
        else:
            self._trade_events = None

        # Load Binance trades (all symbols — for feature computation)
        bn_trades_dir = self.events_dir / "bn_trades"
        bn_files = sorted(
            list(bn_trades_dir.glob(f"{date_str}_*.parquet")) +
            list(bn_trades_dir.glob(f"{date_str}.parquet"))
        )
        if bn_files:
            tables = [pq.read_table(f) for f in bn_files]
            self._bn_trades = _concat_and_sort(tables, "ts_us")
        else:
            self._bn_trades = None

        n_books = self._book_events.num_rows if self._book_events else 0
        n_trades = self._trade_events.num_rows if self._trade_events else 0
        n_bn = self._bn_trades.num_rows if self._bn_trades else 0
        logger.info(
            "Loaded %s/%s: %d book events, %d poly trades, %d bn trades",
            date_str, token_id[:12], n_books, n_trades, n_bn,
        )

    def iter_snapshots(
        self,
        step_us: int = 250_000,            # 250ms default (4Hz, like your current logger)
        start_us: Optional[int] = None,
        end_us: Optional[int] = None,
    ) -> Iterator[ReplaySnapshot]:
        """
        Yield book snapshots at regular intervals.
        
        Between snapshots, all trade events are processed through the
        execution simulator (for limit order fill tracking).
        
        Args:
            step_us: microseconds between snapshots (250_000 = 250ms)
            start_us: start timestamp (default: first event)
            end_us: end timestamp (default: last event)
        """
        if self._book_events is None or self._book_events.num_rows == 0:
            return

        # Convert to Python for iteration (Arrow columns → lists)
        book_ts = self._book_events.column("ts_us").to_pylist()
        book_bid_p = self._book_events.column("bid_prices").to_pylist()
        book_bid_s = self._book_events.column("bid_sizes").to_pylist()
        book_ask_p = self._book_events.column("ask_prices").to_pylist()
        book_ask_s = self._book_events.column("ask_sizes").to_pylist()

        trade_ts = []
        trade_prices = []
        trade_sizes = []
        if self._trade_events and self._trade_events.num_rows > 0:
            trade_ts = self._trade_events.column("ts_us").to_pylist()
            trade_prices = self._trade_events.column("price").to_pylist()
            trade_sizes = self._trade_events.column("size").to_pylist()

        if start_us is None:
            start_us = book_ts[0]
        if end_us is None:
            end_us = book_ts[-1]

        sim = ExecutionSimulator()
        current_book = BookState(
            ts_us=start_us, token_id=self.token_id, bids=[], asks=[],
        )

        book_idx = 0
        trade_idx = 0
        cursor = start_us

        while cursor <= end_us:
            # Advance book state to current time
            while book_idx < len(book_ts) and book_ts[book_idx] <= cursor:
                bp = book_bid_p[book_idx] or []
                bs = book_bid_s[book_idx] or []
                ap = book_ask_p[book_idx] or []
                ask_s = book_ask_s[book_idx] or []

                current_book = BookState(
                    ts_us=book_ts[book_idx],
                    token_id=self.token_id,
                    bids=[BookLevel(p, s) for p, s in zip(bp, bs) if s > 0],
                    asks=[BookLevel(p, s) for p, s in zip(ap, ask_s) if s > 0],
                )
                book_idx += 1

            # Process trades up to current time (for limit fill tracking)
            while trade_idx < len(trade_ts) and trade_ts[trade_idx] <= cursor:
                sim.process_trade(
                    trade_prices[trade_idx],
                    trade_sizes[trade_idx],
                    trade_ts[trade_idx],
                )
                trade_idx += 1

            yield ReplaySnapshot(
                ts_us=cursor,
                book=current_book,
                _sim=sim,
            )

            cursor += step_us

    def time_range(self) -> tuple[Optional[int], Optional[int]]:
        """Return (first_ts_us, last_ts_us) of loaded data."""
        if self._book_events is None or self._book_events.num_rows == 0:
            return None, None
        ts = self._book_events.column("ts_us")
        import pyarrow.compute as pc
        return pc.min(ts).as_py(), pc.max(ts).as_py()


def _concat_and_sort(tables: list, sort_col: str):
    import pyarrow as pa
    import pyarrow.compute as pc
    merged = pa.concat_tables(tables)
    indices = pc.sort_indices(merged, sort_keys=[(sort_col, "ascending")])
    return merged.take(indices)


# ─────────────────────────────────────────────────────────────────────────────
# Window-aware backtest — handles settlement at expiry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketWindow:
    """One 5min or 15min prediction market window."""
    slug: str
    coin: str
    interval_min: int
    up_token_id: str
    down_token_id: str
    window_start_us: int
    window_end_us: int
    target_price: float
    close_price: float        # 0 if unknown
    outcome: int              # 1=UP, 0=DOWN, -1=unknown

    @property
    def resolved(self) -> bool:
        return self.outcome in (0, 1)


@dataclass
class Position:
    """Tracks a position in UP or DOWN tokens."""
    token: str                # "up" or "down"
    size_usdc: float          # total USDC invested
    avg_entry_price: float    # volume-weighted entry price
    n_fills: int = 0


@dataclass
class Trade:
    """Record of a completed trade."""
    ts_us: int
    token: str                # "up" or "down"
    side: str                 # "buy" or "sell" or "settlement"
    price: float
    size_usdc: float
    pnl: float                # realized PnL (0 for buys)
    slippage_bps: float       # vs mid at time of fill
    window_slug: str = ""


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    trades: list[Trade]
    total_pnl: float
    n_windows: int
    n_trades: int
    n_settlements: int
    win_rate: float           # % of profitable trades
    avg_pnl_per_trade: float


class WindowBacktest:
    """
    Full backtest engine that understands Polymarket updown market lifecycle.
    
    For each market window:
      1. Loads the book events for that window's UP token
      2. Replays tick by tick, calling your strategy at each step
      3. At window_end, settles any remaining position at outcome value
         (1.0 if UP won, 0.0 if DOWN won)
    
    Your strategy is a callable:
        def strategy(ctx: StrategyContext) -> Optional[Action]
    
    Usage:
        def my_strategy(ctx):
            if ctx.book.mid < 0.45 and ctx.position is None:
                return Action("buy_up", size_usdc=50)
            if ctx.book.mid > 0.60 and ctx.position is not None:
                return Action("sell_up", size_usdc=ctx.position.size_usdc)
            return None
        
        bt = WindowBacktest("historical_data/events")
        result = bt.run("2026-03-20", coin="btc", interval_min=5, strategy=my_strategy)
        print(f"PnL: {result.total_pnl:.2f} over {result.n_windows} windows")
    """

    def __init__(self, events_dir: str) -> None:
        self.events_dir = Path(events_dir)

    def load_windows(self, date_str: str, coin: str = None,
                     interval_min: int = None) -> list[MarketWindow]:
        """Load resolved market windows from poly_markets table."""
        markets_dir = self.events_dir / "poly_markets"
        files = sorted(
            list(markets_dir.glob(f"{date_str}_*.parquet")) +
            list(markets_dir.glob(f"{date_str}.parquet"))
        )
        if not files:
            logger.warning("No poly_markets data for %s", date_str)
            return []

        import pyarrow.compute as pc
        tables = [pq.read_table(f) for f in files]
        all_markets = _concat_and_sort(tables, "ts_us")
        df = all_markets.to_pandas()

        # Filter
        if coin:
            df = df[df["coin"] == coin]
        if interval_min:
            df = df[df["interval_min"] == interval_min]

        # Group by slug — each slug has a "switch" row (outcome=-1)
        # and optionally a "close" row (outcome=0 or 1)
        windows = {}
        for _, row in df.iterrows():
            slug = row["slug"]
            if slug not in windows:
                windows[slug] = MarketWindow(
                    slug=slug,
                    coin=row["coin"],
                    interval_min=int(row["interval_min"]),
                    up_token_id=row["up_token_id"],
                    down_token_id=row["down_token_id"],
                    window_start_us=int(row["window_start_us"]),
                    window_end_us=int(row["window_end_us"]),
                    target_price=float(row["target_price"]),
                    close_price=0.0,
                    outcome=-1,
                )
            # Update with outcome if this is the close row
            if int(row.get("outcome", -1)) in (0, 1):
                windows[slug].outcome = int(row["outcome"])
                windows[slug].close_price = float(row.get("close_price", 0))

        # Only return resolved windows, sorted by start time
        resolved = [w for w in windows.values() if w.resolved]
        resolved.sort(key=lambda w: w.window_start_us)
        logger.info("Loaded %d resolved windows (%d total) for %s %s %sm",
                     len(resolved), len(windows), date_str, coin or "all",
                     interval_min or "all")
        return resolved

    def run(self, date_str: str, coin: str, interval_min: int,
            strategy, step_us: int = 250_000) -> BacktestResult:
        """
        Run backtest over all resolved windows for a given day/coin/interval.
        
        Args:
            strategy: callable(StrategyContext) → Optional[Action]
            step_us: tick interval in microseconds (250_000 = 250ms)
        """
        windows = self.load_windows(date_str, coin, interval_min)
        if not windows:
            logger.warning("No resolved windows found")
            return BacktestResult([], 0, 0, 0, 0, 0, 0)

        all_trades: list[Trade] = []
        engine = BookReplayEngine(str(self.events_dir))

        for window in windows:
            window_trades = self._run_window(
                engine, window, strategy, step_us,
            )
            all_trades.extend(window_trades)

        total_pnl = sum(t.pnl for t in all_trades)
        n_settlements = sum(1 for t in all_trades if t.side == "settlement")
        profitable = [t for t in all_trades if t.pnl > 0]

        return BacktestResult(
            trades=all_trades,
            total_pnl=total_pnl,
            n_windows=len(windows),
            n_trades=len(all_trades),
            n_settlements=n_settlements,
            win_rate=len(profitable) / len(all_trades) if all_trades else 0,
            avg_pnl_per_trade=total_pnl / len(all_trades) if all_trades else 0,
        )

    def _run_window(self, engine: BookReplayEngine, window: MarketWindow,
                    strategy, step_us: int) -> list[Trade]:
        """Run strategy on a single market window."""
        trades: list[Trade] = []
        position: Optional[Position] = None

        # Load book data for this window's UP token
        try:
            date_str = str(window.window_start_us)[:10]  # fallback
            # Extract date from slug or timestamp
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(window.window_start_us / 1e6, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            engine.load(date_str, token_id=window.up_token_id)
        except Exception as e:
            logger.warning("Failed to load data for %s: %s", window.slug, e)
            return trades

        for snap in engine.iter_snapshots(
            step_us=step_us,
            start_us=window.window_start_us,
            end_us=window.window_end_us,
        ):
            book = snap.book
            if book.best_bid is None or book.best_ask is None:
                continue

            # Build context for strategy
            time_remaining_s = (window.window_end_us - snap.ts_us) / 1e6
            ctx = StrategyContext(
                ts_us=snap.ts_us,
                book=book,
                position=position,
                window=window,
                time_remaining_s=time_remaining_s,
                snap=snap,
            )

            action = strategy(ctx)
            if action is None:
                continue

            mid = book.mid or 0

            if action.action == "buy_up" and action.size_usdc > 0:
                fill = snap.simulate_taker_buy(action.size_usdc)
                if fill.filled_size > 0:
                    if position is None:
                        position = Position("up", fill.filled_size, fill.avg_price, 1)
                    else:
                        # Average in
                        total = position.size_usdc + fill.filled_size
                        position.avg_entry_price = (
                            position.avg_entry_price * position.size_usdc +
                            fill.avg_price * fill.filled_size
                        ) / total
                        position.size_usdc = total
                        position.n_fills += 1

                    slippage = (fill.avg_price - mid) / mid * 10000 if mid else 0
                    trades.append(Trade(
                        ts_us=snap.ts_us, token="up", side="buy",
                        price=fill.avg_price, size_usdc=fill.filled_size,
                        pnl=0, slippage_bps=slippage, window_slug=window.slug,
                    ))

            elif action.action == "sell_up" and position is not None:
                sell_size = min(action.size_usdc, position.size_usdc)
                fill = snap.simulate_taker_sell(sell_size)
                if fill.filled_size > 0:
                    pnl = (fill.avg_price - position.avg_entry_price) * fill.filled_size
                    slippage = (mid - fill.avg_price) / mid * 10000 if mid else 0
                    trades.append(Trade(
                        ts_us=snap.ts_us, token="up", side="sell",
                        price=fill.avg_price, size_usdc=fill.filled_size,
                        pnl=pnl, slippage_bps=slippage, window_slug=window.slug,
                    ))
                    position.size_usdc -= fill.filled_size
                    if position.size_usdc < 0.01:
                        position = None

        # ── Settlement at expiry ──────────────────────────────────────────
        if position is not None:
            settlement_price = 1.0 if window.outcome == 1 else 0.0
            pnl = (settlement_price - position.avg_entry_price) * position.size_usdc
            trades.append(Trade(
                ts_us=window.window_end_us, token="up", side="settlement",
                price=settlement_price, size_usdc=position.size_usdc,
                pnl=pnl, slippage_bps=0, window_slug=window.slug,
            ))
            logger.info(
                "Settlement [%s] outcome=%s | entry=%.4f size=%.1f → PnL=%.2f",
                window.slug, "UP" if window.outcome == 1 else "DOWN",
                position.avg_entry_price, position.size_usdc, pnl,
            )
            position = None

        return trades


@dataclass
class StrategyContext:
    """Everything your strategy sees at each tick."""
    ts_us: int
    book: BookState               # full order book
    position: Optional[Position]  # current position (None if flat)
    window: MarketWindow          # market metadata (target, outcome, times)
    time_remaining_s: float       # seconds until window close
    snap: ReplaySnapshot          # for advanced use (limit orders, etc.)


@dataclass
class Action:
    """What the strategy wants to do."""
    action: str          # "buy_up", "sell_up", "buy_down", "sell_down"
    size_usdc: float     # how much to trade


# ─────────────────────────────────────────────────────────────────────────────
# Example strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_buy_and_hold(ctx: StrategyContext) -> Optional[Action]:
    """
    Simplest strategy: buy UP at start of window, hold until settlement.
    If outcome=UP → profit = (1.0 - entry_price) × size.
    If outcome=DOWN → loss = entry_price × size.
    """
    if ctx.position is None and ctx.time_remaining_s > (ctx.window.interval_min * 60 - 5):
        return Action("buy_up", size_usdc=50.0)
    return None


def strategy_edge_scalp(ctx: StrategyContext) -> Optional[Action]:
    """
    Buy when mid is cheap relative to a naive fair value.
    Sell when the edge is gone, OR hold to settlement.
    
    Replace naive_fair_value with your actual model.
    """
    book = ctx.book
    mid = book.mid
    if mid is None:
        return None

    # Placeholder — your model would compute this from Binance price,
    # vol, time remaining, etc.
    naive_fair_value = 0.55

    edge_bps = (naive_fair_value - mid) * 10_000

    # Buy when cheap
    if edge_bps > 100 and ctx.position is None:
        return Action("buy_up", size_usdc=50.0)

    # Take profit if mid moved in our favor
    if ctx.position is not None and mid > ctx.position.avg_entry_price + 0.05:
        return Action("sell_up", size_usdc=ctx.position.size_usdc)

    # Otherwise hold to settlement
    return None

