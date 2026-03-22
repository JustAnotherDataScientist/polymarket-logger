@echo off
REM sync.bat — Flush local data + download fresh from VM
REM Usage: sync.bat [IP]
REM   sync.bat              (uses default IP)
REM   sync.bat 3.250.28.192 (override IP)

SET SSH_KEY=C:\Users\Raphael\PythonProjects\.ssh\mykeys.pem
SET LOCAL=C:\Users\Raphael\PythonProjects\polymarket-logger
SET REMOTE=/home/ubuntu/polymarket-logger

IF "%1"=="" (SET VM_IP=3.250.28.192) ELSE (SET VM_IP=%1)

echo.
echo ═══════════════════════════════════════════
echo   Sync from VM (%VM_IP%)
echo ═══════════════════════════════════════════
echo.

REM 1. Flush local data
echo [1/3] Flushing local data...
if exist "%LOCAL%\historical_data" rmdir /s /q "%LOCAL%\historical_data"
echo   Done.

REM 2. Create tar on VM
echo [2/3] Packaging data on VM...
ssh -i %SSH_KEY% ubuntu@%VM_IP% "cd %REMOTE% && tar czf /tmp/data.tar.gz historical_data/events historical_data/live 2>/dev/null"

REM 3. Download + extract
echo [3/3] Downloading...
scp -i %SSH_KEY% ubuntu@%VM_IP%:/tmp/data.tar.gz "%LOCAL%\data.tar.gz"
cd /d "%LOCAL%"
tar xzf data.tar.gz
del data.tar.gz

echo.
echo ═══════════════════════════════════════════
echo   Done. Data at: %LOCAL%\historical_data
echo ═══════════════════════════════════════════
echo.
dir /b historical_data\events\
echo.
