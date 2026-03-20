#!/usr/bin/env python3
"""
wal_watchdog.py - Redémarre pm-watcher si le WAL est trop vieux.
Cron : */5 * * * * python3 /home/ubuntu/pm_watcher/wal_watchdog.py
"""
import subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

WAL_MAX_AGE_S = 300
DATA_DIR = Path("/home/ubuntu/pm_watcher/historical_data/live")
LOG_FILE = Path("/home/ubuntu/pm_watcher/logs/watchdog.log")

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts} {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def main():
    now = datetime.now(timezone.utc).timestamp()
    wals = sorted(DATA_DIR.glob("*.wal")) if DATA_DIR.exists() else []
    if not wals:
        log("WARNING: Aucun WAL trouve")
        return
    latest = max(wals, key=lambda p: p.stat().st_mtime)
    age_s = now - latest.stat().st_mtime
    if age_s < WAL_MAX_AGE_S:
        return
    log(f"CRITICAL: WAL {latest.name} pas ecrit depuis {age_s:.0f}s")
    log("ACTION: Redemarrage pm-watcher...")
    try:
        r = subprocess.run(["sudo", "systemctl", "restart", "pm-watcher"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            log("OK: pm-watcher redemarre")
        else:
            log(f"ERROR: {r.stderr.strip()}")
    except Exception as e:
        log(f"EXCEPTION: {e}")

if __name__ == "__main__":
    main()
