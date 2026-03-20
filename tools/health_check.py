"""
health_check.py — Surveillance système du logger live.

Vérifie toutes les 15 minutes (via cron) :
  - Service pm-watcher actif
  - Disque : alerte si > 80%, critique si > 90%
  - RAM : alerte si > 80%
  - WAL actif et récent (données en cours d'écriture)
  - Colonnes critiques remplies dans les dernières données
  - Taux de ticks par marché (détecte les marchés silencieux)

Sortie : log structuré dans logs/health.log
Exit code : 0 = OK, 1 = WARNING, 2 = CRITICAL
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "historical_data" / "live"
LOG_DIR   = BASE_DIR / "logs"
LOG_FILE  = LOG_DIR / "health.log"

DISK_WARN_PCT   = 80
DISK_CRIT_PCT   = 90
RAM_WARN_PCT    = 80
WAL_MAX_AGE_S   = 600   # alerte si WAL pas modifié depuis 10min
TICK_MIN_RATE   = 2.0   # ticks/min minimum par marché attendu (sur 4Hz = 240/min)

CRITICAL_COLS = [
    "up_bid", "up_ask", "up_mid",
    "up_book_bids", "up_book_asks",
    "coin_price", "bn_best_bid", "bn_best_ask",
]

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────
status_code = 0  # 0=OK 1=WARN 2=CRIT

def ok(msg):
    log.info("OK    %s", msg)
    print(f"  ✅ {msg}")

def warn(msg):
    global status_code
    status_code = max(status_code, 1)
    log.warning("WARN  %s", msg)
    print(f"  ⚠️  {msg}")

def crit(msg):
    global status_code
    status_code = max(status_code, 2)
    log.error("CRIT  %s", msg)
    print(f"  🚨 {msg}")

def section(title):
    log.info("── %s ──", title)
    print(f"\n── {title} ──")

# ── Checks ───────────────────────────────────────────────────────────────────

def check_service():
    section("Service pm-watcher")
    r = subprocess.run(
        ["systemctl", "is-active", "pm-watcher"],
        capture_output=True, text=True
    )
    if r.stdout.strip() == "active":
        # Récupérer le PID et uptime
        r2 = subprocess.run(
            ["systemctl", "show", "pm-watcher", "--property=MainPID,ActiveEnterTimestamp"],
            capture_output=True, text=True
        )
        ok(f"pm-watcher actif  {r2.stdout.strip().replace(chr(10), ' | ')}")
    else:
        crit(f"pm-watcher NON ACTIF (état: {r.stdout.strip()})")


def check_disk():
    section("Disque")
    total, used, free = shutil.disk_usage("/")
    pct = used / total * 100
    free_gb = free / 1e9
    total_gb = total / 1e9
    msg = f"{pct:.1f}% utilisé ({free_gb:.1f} GB libre / {total_gb:.0f} GB)"
    if pct >= DISK_CRIT_PCT:
        crit(f"Disque CRITIQUE : {msg}")
    elif pct >= DISK_WARN_PCT:
        warn(f"Disque élevé : {msg}")
    else:
        ok(f"Disque OK : {msg}")

    # Taille des données
    if DATA_DIR.exists():
        data_size = sum(f.stat().st_size for f in DATA_DIR.glob("**/*") if f.is_file())
        parquets  = list(DATA_DIR.glob("*.parquet"))
        chunks    = list(DATA_DIR.glob("*_*.parquet"))
        wals      = list(DATA_DIR.glob("*.wal"))
        ok(f"Données : {data_size/1e6:.0f} MB | {len(parquets)} parquets | "
           f"{len(chunks)} chunks | {len(wals)} WAL")


def check_ram():
    section("RAM")
    with open("/proc/meminfo") as f:
        info = {}
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = int(v.split()[0])  # kB
    total = info["MemTotal"]
    avail = info["MemAvailable"]
    used  = total - avail
    pct   = used / total * 100
    msg   = f"{pct:.1f}% utilisé ({avail/1e6:.1f} GB dispo / {total/1e6:.1f} GB)"
    if pct >= RAM_WARN_PCT:
        warn(f"RAM élevée : {msg}")
    else:
        ok(f"RAM OK : {msg}")


def check_wal():
    section("WAL (écriture active)")
    now = datetime.now(timezone.utc)
    wals = sorted(DATA_DIR.glob("*.wal")) if DATA_DIR.exists() else []

    if not wals:
        warn("Aucun WAL actif — logger écrit-il correctement ?")
        return

    for wal in wals:
        stat   = wal.stat()
        age_s  = now.timestamp() - stat.st_mtime
        size_kb = stat.st_size / 1024
        lines  = sum(1 for _ in open(wal, errors="ignore"))
        if age_s > WAL_MAX_AGE_S:
            warn(f"WAL {wal.name} : dernier write il y a {age_s:.0f}s "
                 f"(max={WAL_MAX_AGE_S}s) — {lines} lignes, {size_kb:.0f} KB")
        else:
            ok(f"WAL {wal.name} : actif il y a {age_s:.0f}s — "
               f"{lines} lignes, {size_kb:.0f} KB")


def check_data_quality():
    section("Qualité données (derniers parquets)")
    import pandas as pd

    # Charger les fichiers les plus récents
    all_files = sorted(
        list(DATA_DIR.glob("*.parquet")) +
        list(DATA_DIR.glob("*_*.parquet"))
    ) if DATA_DIR.exists() else []

    if not all_files:
        warn("Aucun parquet disponible — données pas encore compactées")
        return

    # Prendre les 2 derniers fichiers
    files = all_files[-2:]
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            warn(f"Parquet illisible : {f.name} ({e})")

    if not dfs:
        crit("Impossible de lire les parquets récents")
        return

    df = pd.concat(dfs, ignore_index=True)
    n_rows = len(df)
    ok(f"{n_rows:,} lignes chargées depuis {len(files)} fichier(s)")

    # Check colonnes critiques
    for col in CRITICAL_COLS:
        if col not in df.columns:
            crit(f"Colonne manquante : {col}")
            continue
        if col in ("up_book_bids", "up_book_asks"):
            # Colonne JSON : vérifier non-vide
            non_empty = (df[col].fillna("") != "").mean() * 100
            if non_empty < 90:
                warn(f"{col} : {non_empty:.1f}% non-vide (attendu >90%)")
            else:
                ok(f"{col} : {non_empty:.1f}% non-vide")
            # Vérifier que le JSON est parsable + profondeur
            sample = df[col][df[col].fillna("") != ""].head(10)
            bad = 0
            depths = []
            for v in sample:
                try:
                    parsed = json.loads(v)
                    depths.append(len(parsed))
                except Exception:
                    bad += 1
            if bad > 0:
                crit(f"{col} : {bad} valeurs JSON non parsables")
            elif depths:
                ok(f"{col} : {min(depths)}-{max(depths)} niveaux/tick")
        else:
            fill = pd.to_numeric(df[col], errors="coerce").notna().mean() * 100
            if fill < 85:
                warn(f"{col} : {fill:.1f}% rempli (attendu >85%)")
            else:
                ok(f"{col} : {fill:.1f}% rempli")

    # Check taux de ticks par marché
    if "coin" in df.columns and "interval_min" in df.columns and "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        now_dt = datetime.now(timezone.utc)
        recent = df[df["timestamp_utc"] > now_dt - timedelta(minutes=10)]
        if len(recent) == 0:
            warn("Aucun tick dans les 10 dernières minutes")
        else:
            for (coin, iv), grp in recent.groupby(["coin", "interval_min"]):
                rate = len(grp) / 10  # ticks/min
                key  = f"{coin}-{iv}m"
                if rate < TICK_MIN_RATE:
                    warn(f"{key} : {rate:.1f} ticks/min (attendu ≥{TICK_MIN_RATE})")
                else:
                    ok(f"{key} : {rate:.1f} ticks/min")

    # Check bid < ask
    if "up_bid" in df.columns and "up_ask" in df.columns:
        bid = pd.to_numeric(df["up_bid"], errors="coerce")
        ask = pd.to_numeric(df["up_ask"], errors="coerce")
        viols = (bid >= ask).sum()
        if viols > 0:
            crit(f"up_bid >= up_ask sur {viols} lignes")
        else:
            ok(f"up_bid < up_ask sur toutes les lignes")


def check_logs():
    section("Logs système")
    log_file = BASE_DIR / "logger.log"
    if log_file.exists():
        size_mb = log_file.stat().st_size / 1e6
        ok(f"logger.log : {size_mb:.1f} MB")
        if size_mb > 50:
            warn(f"logger.log volumineux ({size_mb:.1f} MB) — rotation bientôt")
    else:
        warn("logger.log absent")

    # Dernières erreurs dans journald
    r = subprocess.run(
        ["journalctl", "-u", "pm-watcher", "-n", "100",
         "--no-pager", "--since", "1 hour ago"],
        capture_output=True, text=True
    )
    errors = [l for l in r.stdout.splitlines()
              if any(w in l.lower() for w in ["error", "exception", "traceback", "critical"])]
    if errors:
        warn(f"{len(errors)} erreur(s) dans les logs de la dernière heure")
        for e in errors[:3]:
            log.warning("  LOG: %s", e.strip())
    else:
        ok("Aucune erreur dans les logs (1h)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info("═══ Health check %s ═══", now)
    print(f"\n{'═'*50}")
    print(f"  Health Check — {now}")
    print(f"{'═'*50}")

    check_service()
    check_disk()
    check_ram()
    check_wal()
    check_data_quality()
    check_logs()

    print(f"\n{'═'*50}")
    labels = {0: "✅ TOUT OK", 1: "⚠️  WARNINGS", 2: "🚨 CRITIQUE"}
    msg = labels.get(status_code, "❓ INCONNU")
    print(f"  {msg}  (exit code {status_code})")
    log.info("═══ Résultat : %s (code %d) ═══", msg, status_code)
    print(f"{'═'*50}\n")
    sys.exit(status_code)


if __name__ == "__main__":
    main()
