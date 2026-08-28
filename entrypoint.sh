#!/bin/bash
# DiveGuard entrypoint — phased graceful shutdown within the 90s
# stop_grace_period. Budget (audit-corrected from 110.5s overrun):
#   Phase 1  5s  input pause (uvicorn stops accepting, SIGTERM forwarded)
#   Phase 2 55s  ring buffer drain to WAL (event-driven, exits early)
#   Phase 3  5s  checkpoint/fsync (measured 22ms; 5s is a guard, not a sleep)
#   Phase 4 25s  process exit margin
# All phases are implemented inside blueos_extension.DiveGuardService.shutdown;
# this script's job is correct signal forwarding + exec so PID 1 is python.
set -euo pipefail

log() { echo "{\"ts\":\"$(date -u +%FT%TZ)\",\"lvl\":\"INFO\",\"mod\":\"entrypoint\",\"msg\":\"$1\"}"; }

export DIVEGUARD_WAL_DIR="${DIVEGUARD_WAL_DIR:-/data/wal}"
mkdir -p "$DIVEGUARD_WAL_DIR"

log "starting DiveGuard extension (port ${DIVEGUARD_PORT:-8734})"

# exec => python receives SIGTERM directly from Docker; uvicorn triggers
# FastAPI shutdown hook -> DiveGuardService.shutdown() phased drain.
exec python -m uvicorn blueos_extension:app \
    --host 0.0.0.0 \
    --port "${DIVEGUARD_PORT:-8734}" \
    --timeout-graceful-shutdown 85 \
    --log-level "${DIVEGUARD_LOG_LEVEL:-info}"
