from __future__ import annotations

import logging
import logging.handlers
import sys

from config import LOG_BACKUP_COUNT, LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES


def setup_logging() -> logging.Logger:
    """
    Call once at startup (in main.py).
    Returns the root logger configured with:
      - colored StreamHandler (stdout)
      - RotatingFileHandler (if LOG_FILE is set)
    All modules just do: logger = logging.getLogger(__name__)
    """
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-import
    if root.handlers:
        return root

    # ── stdout handler ──────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(_ColorFormatter(fmt, datefmt=datefmt))
    root.addHandler(stream_handler)

    # ── rotating file handler ───────────────────────────
    if LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    return root


class _ColorFormatter(logging.Formatter):
    """ANSI color codes for terminal output."""

    _COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)
