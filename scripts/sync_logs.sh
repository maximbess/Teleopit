#!/usr/bin/env bash
# Pull logs/rsl_rl from the training machine into this repo.
# The remote is read-only: scp only writes to the local destination.
# Usage:
#   scripts/sync_logs.sh user@host:/path/to/Teleopit --port 12345
#   scripts/sync_logs.sh user@host:/path/to/Teleopit --port 12345 --watch 30
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@host:/path/to/Teleopit --port PORT [--watch SECONDS]" >&2
  exit 2
fi

REMOTE="${1%/}"
shift
PORT=""
WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:-}"; shift 2 ;;
    --watch) WATCH="${2:-30}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PORT" ]]; then
  echo "missing --port" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/logs/rsl_rl"

# Connection multiplexing asks for the password on the first sync and reuses
# that authenticated connection. scp is used because the server has no rsync.
CONTROL_PATH="/tmp/teleopit-sync-${UID}-%C"

sync_once() {
  scp -r -p -P "${PORT}" \
    -o ControlMaster=auto \
    -o ControlPersist=600 \
    -o "ControlPath=${CONTROL_PATH}" \
    "${REMOTE}/logs/rsl_rl/." \
    "${ROOT}/logs/rsl_rl/"
}

if [[ "$WATCH" == 0 ]]; then
  sync_once
else
  while true; do
    sync_once || true
    sleep "$WATCH"
  done
fi