"""
main.py — PM·BN Watcher entry point.

Runs the live logger with event-sourced recording for backtesting.
Two storage paths run in parallel:
  1. Legacy derived snapshots (Parquet, 101 columns, for dashboard/live monitoring)
  2. Event store (raw trades + books, for precise backtest replay)
"""
from __future__ import annotations

import sys
import logging

from logger_setup import setup_logging
from dashboard import DashboardServer
from event_store import EventStore
from logger_core import run_logger


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)
    log.info("PM·BN Watcher starting up…")

    event_store = EventStore("historical_data")
    log.info("Event store ready: %s", event_store.stats())

    dashboard = DashboardServer()
    dashboard.start()

    try:
        run_logger(dashboard=dashboard, event_store=event_store)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — exiting")
    except Exception as exc:
        log.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        event_store.close_all()
        log.info("Event store closed. Stats: %s", event_store.stats())


if __name__ == "__main__":
    main()
