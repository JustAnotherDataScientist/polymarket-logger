#!/usr/bin/env bash
# deploy/deploy.sh — Update from git and restart.
#
# Usage:
#   cd ~/polymarket-logger && bash deploy/deploy.sh
#   cd ~/polymarket-logger && bash deploy/deploy.sh --flush-data
set -euo pipefail

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; W='\033[1m'; N='\033[0m'
ok()   { echo -e "${G}  ✓ $*${N}"; }
warn() { echo -e "${Y}  ⚠ $*${N}"; }
err()  { echo -e "${R}  ✗ $*${N}"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"
FLUSH="${1:-}"

echo -e "\n${W}=== Deploy ===${N}  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# 1. Git pull
echo -e "\n[1/4] Git pull"
cd "$PROJECT_DIR"
BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
git pull 2>/dev/null && ok "$(git log -1 --format='%h %s')" || warn "Not a git repo — skipping pull"

# 2. Stop
echo -e "\n[2/4] Stop service"
sudo systemctl stop pm-watcher 2>/dev/null || true
sleep 2
ok "Stopped"

# 3. Flush data if requested
echo -e "\n[3/4] Data"
if [[ "$FLUSH" == "--flush-data" ]]; then
    rm -f "$PROJECT_DIR"/historical_data/live/*.parquet \
          "$PROJECT_DIR"/historical_data/live/*.wal \
          "$PROJECT_DIR"/historical_data/live/*.tmp 2>/dev/null || true
    warn "Live data flushed"
else
    COUNT=$(ls "$PROJECT_DIR"/historical_data/live/*.parquet 2>/dev/null | wc -l || echo 0)
    ok "Data preserved ($COUNT files)"
fi

# 4. Update deps + restart
echo -e "\n[4/4] Restart"
"$PROJECT_DIR/venv/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

sudo systemctl start pm-watcher
sleep 4

if systemctl is-active --quiet pm-watcher; then
    ok "Service active"
    sudo journalctl -u pm-watcher -n 5 --no-pager
else
    sudo journalctl -u pm-watcher -n 20 --no-pager
    err "Service failed"
fi
