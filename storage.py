"""
storage.py — Stockage Parquet sans lecture du fichier existant.

Architecture v2 :
  - Chaque compaction WAL → écrit un chunk Parquet INDÉPENDANT
    (ex: 2026-03-17_001.parquet, 2026-03-17_002.parquet, ...)
  - Jamais de pd.read_parquet() pendant le logging → RAM constante ~50 MB
  - Merge quotidien à 02h00 UTC → produit 2026-03-17.parquet final
  - Au démarrage : récupère les WAL orphelins, détecte les trous

RAM utilisée : O(1) — indépendant de la durée de logging
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from csv_schema import EXPECTED_HEADER, N_COLS

logger = logging.getLogger(__name__)

PARQUET_FLUSH_SECONDS  = 120    # compaction WAL → chunk toutes les 2 min
MERGE_HOUR_UTC         = 2      # merge des chunks à 02h00 UTC
COMPRESSION            = "snappy"
ROW_GROUP_SIZE         = 5000

_DEDUP_COLS = ["timestamp_utc", "coin", "interval_min", "slug"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_date_str(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Chunk writer : écrit un seul bloc WAL en parquet sans relire l'existant
# ─────────────────────────────────────────────────────────────────────────────

def _write_chunk(filepath: Path, rows: list[list[Any]]) -> None:
    """
    Écrit rows dans filepath (un chunk parquet indépendant).
    N'ouvre JAMAIS un parquet existant — RAM = O(rows) uniquement.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows, columns=EXPECTED_HEADER)
    df = (
        df.drop_duplicates(subset=_DEDUP_COLS, keep="last")
          .sort_values("timestamp_utc")
          .reset_index(drop=True)
    )
    tmp = filepath.with_suffix(".tmp")
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, str(tmp), compression=COMPRESSION,
                   row_group_size=ROW_GROUP_SIZE)
    os.replace(str(tmp), str(filepath))
    logger.debug("Chunk écrit : %s (%d lignes)", filepath.name, len(df))


def _merge_chunks(date_str: str, output_dir: Path) -> bool:
    """
    Merge tous les chunks du jour en un seul parquet final.
    Appelé une fois par jour (02h00 UTC) ou à la demande.
    Seul endroit où on lit plusieurs parquets — mais c'est J-1 donc ok.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks = sorted(output_dir.glob(f"{date_str}_*.parquet"))
    final  = output_dir / f"{date_str}.parquet"

    if not chunks:
        return False

    logger.info("Merge %s : %d chunks → %s", date_str, len(chunks), final.name)
    dfs = []
    for chunk in chunks:
        try:
            dfs.append(pd.read_parquet(chunk))
        except Exception as e:
            logger.warning("Chunk illisible ignoré : %s (%s)", chunk.name, e)

    if not dfs:
        return False

    merged = (
        pd.concat(dfs, ignore_index=True)
          .drop_duplicates(subset=_DEDUP_COLS, keep="last")
          .sort_values("timestamp_utc")
          .reset_index(drop=True)
    )
    tmp = final.with_suffix(".tmp")
    tbl = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(tbl, str(tmp), compression=COMPRESSION,
                   row_group_size=ROW_GROUP_SIZE)
    os.replace(str(tmp), str(final))

    # Supprimer les chunks mergés
    for chunk in chunks:
        try:
            chunk.unlink()
        except Exception:
            pass

    logger.info("Merge %s terminé : %d lignes dans %s (%.1f MB)",
                date_str, len(merged), final.name,
                final.stat().st_size / 1e6)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Journal WAL — identique à v1
# ─────────────────────────────────────────────────────────────────────────────

class _Journal:
    """Fichier journal append-only : une ligne CSV = un tick."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh   = open(path, "a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)

    def write(self, row: list[Any]) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()

    def read_all(self) -> list[list[str]]:
        rows = []
        try:
            import csv as _csv
            with open(self._path, "r", encoding="utf-8", newline="") as f:
                reader = _csv.reader(f)
                for lineno, parts in enumerate(reader, 1):
                    if not parts:
                        continue
                    if len(parts) == N_COLS:
                        rows.append(parts)
                    else:
                        logger.warning(
                            "WAL ligne %d ignorée (%d cols != %d)",
                            lineno, len(parts), N_COLS,
                        )
        except Exception as e:
            logger.error("WAL read_all: %s", e)
        return rows

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass

    def close_and_delete(self) -> None:
        self.close()
        with self._lock:
            try:
                self._path.unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# StorageManager — API publique
# ─────────────────────────────────────────────────────────────────────────────

class ParquetStorage:
    """
    Stockage Parquet RAM-constant.

    Flux : tick → WAL (append) → chunk toutes les 2min → merge quotidien 02h UTC
    RAM peak : ~50 MB (2 min de données × 8 marchés × 4Hz)
    """

    def __init__(self, output_dir: str,
                 flush_every_seconds: int = PARQUET_FLUSH_SECONDS) -> None:
        self.output_dir   = Path(output_dir) / "live"
        self._flush_every = flush_every_seconds
        self._lock        = threading.Lock()
        self._enabled     = self._check_imports()

        self._current_date: Optional[str]     = None
        self._journal:      Optional[_Journal] = None
        self._chunk_idx:    int               = 0   # compteur de chunks du jour
        self._last_merge_date: Optional[str]  = None

        if self._enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._recover_on_startup()
            # Thread compaction périodique
            threading.Thread(
                target=self._compact_loop, daemon=True, name="parquet-compact"
            ).start()
            # Thread merge quotidien
            threading.Thread(
                target=self._merge_loop, daemon=True, name="parquet-merge"
            ).start()

    def _check_imports(self) -> bool:
        try:
            import pyarrow  # noqa
            import pandas   # noqa
            logger.info("Parquet storage v2 (RAM-constant) → %s", self.output_dir)
            return True
        except ImportError as e:
            logger.error("pyarrow/pandas manquant — Parquet désactivé: %s", e)
            return False

    # ── Startup ───────────────────────────────────────────────────────────────

    def _recover_on_startup(self) -> None:
        """Compacte les WAL orphelins (crash précédent) → chunks."""
        wals = sorted(self.output_dir.glob("*.wal"))
        if wals:
            logger.info("Récupération : %d WAL orphelin(s)", len(wals))
        for wal_path in wals:
            date_str = wal_path.stem
            j = _Journal(wal_path)
            rows = j.read_all()
            if rows:
                chunk_path = self._next_chunk_path(date_str)
                try:
                    _write_chunk(chunk_path, rows)
                    j.close_and_delete()
                    logger.info("  Récupéré %s : %d lignes → %s",
                                date_str, len(rows), chunk_path.name)
                except Exception as e:
                    logger.error("  Récupération %s échouée : %s", date_str, e)
                    j.close()
            else:
                j.close_and_delete()

        # Nettoie les .tmp orphelins
        for tmp in self.output_dir.glob("*.tmp"):
            logger.warning("Suppression .tmp orphelin : %s", tmp.name)
            try:
                tmp.unlink()
            except Exception:
                pass

        self._check_gap()

    def _check_gap(self) -> None:
        last_ts = self.get_last_stored_timestamp()
        if last_ts is None:
            logger.info("Aucune donnée existante — démarrage à zéro")
            return
        gap = (_utc_now() - last_ts).total_seconds()
        if gap > 120:
            logger.warning("⚠ TROU : %.0f secondes manquantes (dernier tick : %s)",
                           gap, last_ts.strftime("%Y-%m-%d %H:%M:%S UTC"))
        else:
            logger.info("Reprise propre — dernier tick il y a %.0f secondes", gap)

    # ── Écriture ──────────────────────────────────────────────────────────────

    def append(self, row: list[Any]) -> None:
        if not self._enabled:
            return
        date_str = _utc_date_str()
        with self._lock:
            # Changement de jour → compacte l'ancien WAL
            if self._current_date and self._current_date != date_str:
                self._do_compact(self._current_date)
                self._journal      = None
                self._current_date = None
                self._chunk_idx    = 0

            if self._journal is None:
                wal = self.output_dir / f"{date_str}.wal"
                self._journal      = _Journal(wal)
                self._current_date = date_str
                # Calculer le prochain index de chunk
                existing = sorted(self.output_dir.glob(f"{date_str}_*.parquet"))
                self._chunk_idx = len(existing)

            self._journal.write(row)

    # ── Compaction WAL → chunk ────────────────────────────────────────────────

    def _next_chunk_path(self, date_str: str) -> Path:
        existing = sorted(self.output_dir.glob(f"{date_str}_*.parquet"))
        idx = len(existing)
        return self.output_dir / f"{date_str}_{idx:04d}.parquet"

    def _do_compact(self, date_str: str) -> None:
        """WAL → chunk indépendant. RAM = O(2min de données) seulement."""
        if self._journal is None:
            return
        rows = self._journal.read_all()
        if not rows:
            return
        chunk_path = self._next_chunk_path(date_str)
        try:
            _write_chunk(chunk_path, rows)
            self._journal.close_and_delete()
            self._journal   = None
            # Recréer le journal pour les prochains ticks
            wal = self.output_dir / f"{date_str}.wal"
            self._journal = _Journal(wal)
            self._chunk_idx += 1
        except Exception as e:
            logger.error("Compaction échouée : %s — WAL conservé", e)

    def _compact_loop(self) -> None:
        while True:
            time.sleep(self._flush_every)
            with self._lock:
                if self._current_date and self._journal:
                    self._do_compact(self._current_date)

    # ── Merge quotidien (02h UTC) ─────────────────────────────────────────────

    def _merge_loop(self) -> None:
        """Merge les chunks J-1 en un seul parquet à 02h00 UTC."""
        from datetime import timedelta
        while True:
            now = _utc_now()
            # Calculer le prochain 02h00 UTC
            next_merge = now.replace(hour=MERGE_HOUR_UTC, minute=0,
                                     second=30, microsecond=0)
            if next_merge <= now:
                next_merge += timedelta(days=1)
            time.sleep((next_merge - now).total_seconds())

            # Merger J-1 (le jour d'hier est forcément terminé)
            yesterday = _utc_date_str(
                _utc_now() - timedelta(days=1)
            )
            if yesterday != self._last_merge_date:
                try:
                    merged = _merge_chunks(yesterday, self.output_dir)
                    if merged:
                        self._last_merge_date = yesterday
                except Exception as e:
                    logger.error("Merge %s échoué : %s", yesterday, e)

    def flush(self) -> None:
        """Force une compaction immédiate (pour les tests / shutdown)."""
        with self._lock:
            if self._current_date and self._journal:
                self._do_compact(self._current_date)

    def close(self) -> None:
        self.flush()
        with self._lock:
            if self._journal:
                try:
                    self._journal.close()
                except Exception:
                    pass

    # ── Stats & lecture ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        chunks   = list(self.output_dir.glob("*.parquet")) + \
                   list(self.output_dir.glob("*_*.parquet"))
        wal      = list(self.output_dir.glob("*.wal"))
        total_mb = sum(f.stat().st_size for f in chunks) / 1e6
        return {
            "parquet_files":  len(chunks),
            "wal_files":      len(wal),
            "total_mb":       round(total_mb, 1),
            "current_date":   self._current_date,
        }

    def get_last_stored_timestamp(self) -> Optional[datetime]:
        try:
            import pandas as pd
            # Chercher dans les chunks du jour d'abord, puis les parquets finaux
            files = sorted(
                list(self.output_dir.glob("*_*.parquet")) +
                list(self.output_dir.glob("[0-9][0-9][0-9][0-9]-*.parquet"))
            )
            if not files:
                return None
            df = pd.read_parquet(files[-1], columns=["timestamp_utc"])
            if df.empty:
                return None
            ts = pd.to_datetime(df["timestamp_utc"].max(), utc=True)
            return ts.to_pydatetime()
        except Exception:
            return None


class StorageManager:
    """API publique utilisée par logger_core."""

    def __init__(self) -> None:
        from config import PARQUET_OUTPUT_DIR
        self._parquet = ParquetStorage(PARQUET_OUTPUT_DIR)
        self._cached_stats: dict = {}
        self._stats_mono = 0.0

    def append(self, row: list[Any]) -> None:
        self._parquet.append(row)

    def stats(self) -> dict:
        import time
        now = time.monotonic()
        if now - self._stats_mono < 30:
            return self._cached_stats
        self._cached_stats = self._parquet.stats()
        self._stats_mono = now
        return self._cached_stats

    def flush(self) -> None:
        self._parquet.flush()

    def close(self) -> None:
        self._parquet.close()
