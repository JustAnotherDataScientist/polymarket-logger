"""
event_store.py — Event-sourced storage for backtest-grade data.

Architecture:
  4 independent Parquet streams, each with its own buffer → chunk → merge cycle.
  
  Tables:
    poly_books  — every Polymarket book update (price_change + book_snapshot)
    poly_trades — every Polymarket confirmed trade (last_trade_price)
    bn_trades   — every Binance trade
    bn_books    — every Binance @depth20@100ms snapshot

  Book levels stored as Arrow list<float32> columns — native Parquet nested types,
  NOT JSON strings. This gives:
    - 5-10x better compression than JSON (delta encoding on sorted prices)
    - Zero deserialization cost at read time (Polars/DuckDB read natively)
    - Direct numpy array access via .to_numpy()

  RAM: O(buffer_size) per writer ≈ 20-40 MB total across all 4 tables.
  Disk: ~800 MB/day compressed (vs ~2 GB with 101-col derived snapshots).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

COMPRESSION = "zstd"           # better ratio than snappy for this data profile
ROW_GROUP_SIZE = 10_000
MERGE_HOUR_UTC = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_us(dt: datetime) -> int:
    """Datetime → microsecond Unix timestamp (int64)."""
    return int(dt.timestamp() * 1_000_000)


def _utc_date_str(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas — one per event table
# ─────────────────────────────────────────────────────────────────────────────

# Polymarket book updates: full book state after each event
POLY_BOOKS_SCHEMA = pa.schema([
    ("ts_us",        pa.int64()),           # recv timestamp, microseconds
    ("token_id",     pa.string()),          # UP or DOWN token
    ("event_type",   pa.uint8()),           # 0=price_change 1=book_snapshot
    ("bid_prices",   pa.list_(pa.float32())),
    ("bid_sizes",    pa.list_(pa.float32())),
    ("ask_prices",   pa.list_(pa.float32())),
    ("ask_sizes",    pa.list_(pa.float32())),
    ("best_bid",     pa.float32()),         # redundant but fast to query
    ("best_ask",     pa.float32()),
])

# Polymarket confirmed trades
POLY_TRADES_SCHEMA = pa.schema([
    ("ts_us",        pa.int64()),           # trade timestamp from event
    ("recv_us",      pa.int64()),           # local recv timestamp
    ("token_id",     pa.string()),
    ("price",        pa.float32()),
    ("size",         pa.float32()),         # USDC
    ("tx_hash",      pa.string()),
])

# Binance trades
BN_TRADES_SCHEMA = pa.schema([
    ("ts_us",        pa.int64()),           # exchange timestamp
    ("recv_us",      pa.int64()),           # local recv timestamp
    ("symbol",       pa.dictionary(pa.int8(), pa.string())),  # dict-encoded
    ("price",        pa.float64()),
    ("qty",          pa.float64()),
    ("is_buy",       pa.bool_()),
    ("trade_id",     pa.int64()),
])

# Binance depth snapshots (@depth20@100ms)
BN_BOOKS_SCHEMA = pa.schema([
    ("ts_us",        pa.int64()),
    ("symbol",       pa.dictionary(pa.int8(), pa.string())),
    ("bid_prices",   pa.list_(pa.float64())),
    ("bid_sizes",    pa.list_(pa.float64())),
    ("ask_prices",   pa.list_(pa.float64())),
    ("ask_sizes",    pa.list_(pa.float64())),
])

# Market metadata — logged on market switch + window close (outcome)
POLY_MARKETS_SCHEMA = pa.schema([
    ("ts_us",           pa.int64()),
    ("slug",            pa.string()),
    ("coin",            pa.string()),
    ("interval_min",    pa.int16()),
    ("up_token_id",     pa.string()),
    ("down_token_id",   pa.string()),
    ("window_start_us", pa.int64()),
    ("window_end_us",   pa.int64()),
    ("target_price",    pa.float64()),
    ("close_price",     pa.float64()),    # filled at window close, 0 at switch
    ("outcome",         pa.int8()),       # 1=UP won, 0=DOWN won, -1=unknown
])


# ─────────────────────────────────────────────────────────────────────────────
# EventWriter — one per table, buffers in memory, flushes to Parquet chunks
# ─────────────────────────────────────────────────────────────────────────────

class EventWriter:
    """
    Append-only writer for a single event table.
    
    Buffers rows in Python lists (minimal overhead), flushes to a Parquet
    chunk when the buffer exceeds `flush_rows` or `flush_seconds`.
    
    No WAL file needed — the in-memory buffer IS the WAL.
    On crash, you lose at most `flush_seconds` of data (acceptable for
    a logger that runs 24/7 — the next chunk picks up immediately).
    
    Thread-safe: multiple WS threads can call append() concurrently.
    """

    def __init__(
        self,
        name: str,
        schema: pa.Schema,
        output_dir: Path,
        flush_rows: int = 50_000,
        flush_seconds: int = 60,
    ) -> None:
        self.name = name
        self.schema = schema
        self.output_dir = output_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._flush_rows = flush_rows
        self._flush_seconds = flush_seconds

        self._lock = threading.Lock()
        # One list per column — columnar layout even in the buffer
        self._buffers: dict[str, list] = {f.name: [] for f in schema}
        self._buf_len = 0
        self._current_date: Optional[str] = None
        self._chunk_idx = 0

        self._stats_rows_written = 0
        self._stats_bytes_written = 0

        # Background IO worker (single thread replaces ephemeral threads)
        self._stop = threading.Event()
        self._write_q: "queue.Queue[tuple]" = queue.Queue()
        self._io_thread = threading.Thread(
            target=self._io_loop, daemon=True, name=f"io-{name}"
        )
        self._io_thread.start()

        # Background flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name=f"flush-{name}"
        )
        self._flush_thread.start()

        # Daily merge thread
        self._last_merge_date: Optional[str] = None
        self._merge_thread = threading.Thread(
            target=self._merge_loop, daemon=True, name=f"merge-{name}"
        )
        self._merge_thread.start()

        logger.info("EventWriter[%s] → %s", name, self.output_dir)

    def append(self, **kwargs: Any) -> None:
        """
        Append a single event. Column names must match the schema.
        
        Example:
            writer.append(ts_us=16..., token_id="0x...", price=0.55, ...)
        """
        with self._lock:
            date_str = _utc_date_str()
            if self._current_date and self._current_date != date_str:
                self._do_flush(self._current_date)
                self._current_date = date_str
                self._chunk_idx = len(list(
                    self.output_dir.glob(f"{date_str}_*.parquet")
                ))

            if self._current_date is None:
                self._current_date = date_str
                self._chunk_idx = len(list(
                    self.output_dir.glob(f"{date_str}_*.parquet")
                ))

            for col_name in self._buffers:
                self._buffers[col_name].append(kwargs.get(col_name))
            self._buf_len += 1

            if self._buf_len >= self._flush_rows:
                self._do_flush(self._current_date)

    def append_batch(self, columns: dict[str, list]) -> None:
        """
        Append multiple rows at once. Each value is a list of column values.
        More efficient than calling append() in a loop.
        """
        if not columns:
            return
        n = len(next(iter(columns.values())))
        with self._lock:
            date_str = _utc_date_str()
            if self._current_date and self._current_date != date_str:
                self._do_flush(self._current_date)
                self._current_date = date_str
                self._chunk_idx = len(list(
                    self.output_dir.glob(f"{date_str}_*.parquet")
                ))
            if self._current_date is None:
                self._current_date = date_str
                self._chunk_idx = len(list(
                    self.output_dir.glob(f"{date_str}_*.parquet")
                ))
            for col_name, values in columns.items():
                self._buffers[col_name].extend(values)
            self._buf_len += n
            if self._buf_len >= self._flush_rows:
                self._do_flush(self._current_date)

    def _do_flush(self, date_str: str) -> None:
        """Flush buffer to a Parquet chunk. Caller must hold self._lock."""
        if self._buf_len == 0:
            return

        # Snapshot and reset buffer
        columns = {k: list(v) for k, v in self._buffers.items()}
        n_rows = self._buf_len
        for v in self._buffers.values():
            v.clear()
        self._buf_len = 0
        chunk_idx = self._chunk_idx
        self._chunk_idx += 1

        # Queue the IO work to the persistent worker thread
        self._write_q.put((date_str, chunk_idx, columns, n_rows))

    def _io_loop(self) -> None:
        """Single persistent thread that handles all Parquet writes."""
        while not self._stop.is_set():
            try:
                item = self._write_q.get(timeout=1.0)
            except queue.Empty:
                continue
            date_str, chunk_idx, columns, n_rows = item
            chunk_path = self.output_dir / f"{date_str}_{chunk_idx:04d}.parquet"
            tmp = chunk_path.with_suffix(".tmp")
            try:
                arrays = []
                for field in self.schema:
                    col_data = columns[field.name]
                    arrays.append(pa.array(col_data, type=field.type))
                table = pa.table(
                    {f.name: a for f, a in zip(self.schema, arrays)},
                    schema=self.schema,
                )
                pq.write_table(
                    table, str(tmp),
                    compression=COMPRESSION,
                    row_group_size=ROW_GROUP_SIZE,
                )
                os.replace(str(tmp), str(chunk_path))
                size = chunk_path.stat().st_size
                self._stats_rows_written += n_rows
                self._stats_bytes_written += size
                logger.debug(
                    "EventWriter[%s] chunk %s: %d rows, %.1f KB",
                    self.name, chunk_path.name, n_rows, size / 1024,
                )
            except Exception as e:
                logger.error("EventWriter[%s] flush failed: %s", self.name, e)
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            finally:
                self._write_q.task_done()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._flush_seconds)
            with self._lock:
                if self._current_date and self._buf_len > 0:
                    self._do_flush(self._current_date)

    def _merge_loop(self) -> None:
        """Merge yesterday's chunks into a single Parquet at 02:00 UTC."""
        while not self._stop.is_set():
            now = _utc_now()
            next_merge = now.replace(
                hour=MERGE_HOUR_UTC, minute=0, second=30, microsecond=0
            )
            if next_merge <= now:
                next_merge += timedelta(days=1)
            self._stop.wait(timeout=(next_merge - now).total_seconds())
            if self._stop.is_set():
                break

            yesterday = _utc_date_str(_utc_now() - timedelta(days=1))
            if yesterday != self._last_merge_date:
                self._do_merge(yesterday)
                self._last_merge_date = yesterday

    def _do_merge(self, date_str: str) -> None:
        """Merge all chunks for a given date into one Parquet file."""
        chunks = sorted(self.output_dir.glob(f"{date_str}_*.parquet"))
        if not chunks:
            return
        final = self.output_dir / f"{date_str}.parquet"
        logger.info(
            "EventWriter[%s] merging %d chunks → %s",
            self.name, len(chunks), final.name,
        )
        try:
            tables = []
            for chunk in chunks:
                try:
                    tables.append(pq.read_table(chunk, schema=self.schema))
                except Exception as e:
                    logger.warning("Chunk unreadable, skipped: %s (%s)", chunk.name, e)

            if not tables:
                return

            merged = pa.concat_tables(tables)
            # Sort by timestamp
            ts_col = "ts_us" if "ts_us" in merged.column_names else merged.column_names[0]
            indices = pa.compute.sort_indices(merged, sort_keys=[(ts_col, "ascending")])
            merged = merged.take(indices)

            tmp = final.with_suffix(".tmp")
            pq.write_table(
                merged, str(tmp),
                compression=COMPRESSION,
                row_group_size=ROW_GROUP_SIZE,
            )
            os.replace(str(tmp), str(final))

            for chunk in chunks:
                try:
                    chunk.unlink()
                except Exception:
                    pass

            logger.info(
                "EventWriter[%s] merged: %d rows, %.1f MB",
                self.name, merged.num_rows, final.stat().st_size / 1e6,
            )
        except Exception as e:
            logger.error("EventWriter[%s] merge failed: %s", self.name, e)

    def flush(self) -> None:
        """Force immediate flush (for shutdown / tests)."""
        with self._lock:
            if self._current_date and self._buf_len > 0:
                self._do_flush(self._current_date)

    def close(self) -> None:
        self.flush()
        # Wait for IO worker to drain the queue
        self._write_q.join()
        self._stop.set()

    def stats(self) -> dict:
        chunks = list(self.output_dir.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in chunks)
        return {
            "name": self.name,
            "chunks": len(chunks),
            "total_mb": round(total_bytes / 1e6, 1),
            "buf_rows": self._buf_len,
            "total_rows_written": self._stats_rows_written,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EventStore — public API, manages all 4 (or 5) writers
# ─────────────────────────────────────────────────────────────────────────────

class EventStore:
    """
    Manages all event writers. Single entry point for the logger.
    
    Usage:
        store = EventStore("historical_data")
        
        # From Polymarket WS callback:
        store.poly_books.append(ts_us=..., token_id=..., ...)
        store.poly_trades.append(ts_us=..., token_id=..., ...)
        
        # From Binance WS callback:
        store.bn_trades.append(ts_us=..., symbol=..., ...)
        store.bn_books.append(ts_us=..., symbol=..., ...)
        
        # Market switch:
        store.poly_markets.append(ts_us=..., slug=..., ...)
    """

    def __init__(self, output_dir: str = "historical_data") -> None:
        base = Path(output_dir) / "events"

        self.poly_books = EventWriter(
            "poly_books", POLY_BOOKS_SCHEMA, base,
            flush_rows=20_000, flush_seconds=60,
        )
        self.poly_trades = EventWriter(
            "poly_trades", POLY_TRADES_SCHEMA, base,
            flush_rows=5_000, flush_seconds=60,
        )
        self.bn_trades = EventWriter(
            "bn_trades", BN_TRADES_SCHEMA, base,
            flush_rows=100_000, flush_seconds=30,  # higher volume
        )
        self.bn_books = EventWriter(
            "bn_books", BN_BOOKS_SCHEMA, base,
            flush_rows=50_000, flush_seconds=30,
        )
        self.poly_markets = EventWriter(
            "poly_markets", POLY_MARKETS_SCHEMA, base,
            flush_rows=1_000, flush_seconds=300,  # low volume, flush rarely
        )

    def flush_all(self) -> None:
        for w in (self.poly_books, self.poly_trades,
                  self.bn_trades, self.bn_books, self.poly_markets):
            w.flush()

    def close_all(self) -> None:
        for w in (self.poly_books, self.poly_trades,
                  self.bn_trades, self.bn_books, self.poly_markets):
            w.close()

    def stats(self) -> dict:
        return {
            w.name: w.stats()
            for w in (self.poly_books, self.poly_trades,
                      self.bn_trades, self.bn_books, self.poly_markets)
        }
