#!/usr/bin/env bash
# Pull events.out and one chosen checkpoint from a remote logs/rsl_rl run.
# A run that has no checkpoint yet still downloads its event files.
# The remote is read-only: scp only writes to the local destination.
# Usage:
#   scripts/sync_logs.sh user@host:/path/to/Teleopit --port 12345
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@host:/path/to/Teleopit --port PORT" >&2
  exit 2
fi

REMOTE="${1%/}"
shift
if [[ "$REMOTE" != *:* ]]; then
  echo "remote must be user@host:/path/to/Teleopit" >&2
  exit 2
fi
REMOTE_HOST="${REMOTE%%:*}"
REMOTE_ROOT="${REMOTE#*:}"
PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:-}"; shift 2 ;;
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
SSH_OPTIONS=(
  -p "${PORT}"
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=${CONTROL_PATH}"
)

RUN_OUTPUT="$(
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "find '${REMOTE_ROOT}/logs/rsl_rl' -mindepth 2 -maxdepth 2 -type d -print | sort"
)"

RUNS=()
while IFS= read -r remote_run; do
  [[ -n "$remote_run" ]] || continue
  RUNS[${#RUNS[@]}]="${remote_run#"${REMOTE_ROOT}/logs/rsl_rl/"}"
done <<< "$RUN_OUTPUT"

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "no runs found under ${REMOTE_ROOT}/logs/rsl_rl" >&2
  exit 1
fi

echo "Available remote runs:"
for ((i = 0; i < ${#RUNS[@]}; i++)); do
  printf "  %2d) %s\n" "$((i + 1))" "${RUNS[$i]}"
done

read -r -p "Choose a run [1-${#RUNS[@]}]: " CHOICE
if [[ ! "$CHOICE" =~ ^[0-9]+$ ]] \
  || ((CHOICE < 1 || CHOICE > ${#RUNS[@]})); then
  echo "invalid selection: ${CHOICE}" >&2
  exit 2
fi

RUN_REL="${RUNS[$((CHOICE - 1))]}"
REMOTE_RUN="${REMOTE_ROOT}/logs/rsl_rl/${RUN_REL}"

MODEL_OUTPUT="$(
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "find '${REMOTE_RUN}' -maxdepth 1 -type f -name 'model_*.pt' -print | sort"
)"

MODELS=()
while IFS= read -r remote_model; do
  [[ -n "$remote_model" ]] || continue
  MODELS[${#MODELS[@]}]="$(basename "${remote_model}")"
done < <(printf '%s\n' "$MODEL_OUTPUT" | sort -V)

MODEL=""
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "no model_*.pt checkpoints in logs/rsl_rl/${RUN_REL}; pulling events only"
else
  echo "Available checkpoints in logs/rsl_rl/${RUN_REL}:"
  for ((i = 0; i < ${#MODELS[@]}; i++)); do
    printf "  %2d) %s\n" "$((i + 1))" "${MODELS[$i]}"
  done

  read -r -p "Choose a checkpoint [1-${#MODELS[@]}]: " MODEL_CHOICE
  if [[ ! "$MODEL_CHOICE" =~ ^[0-9]+$ ]] \
      || ((MODEL_CHOICE < 1 || MODEL_CHOICE > ${#MODELS[@]})); then
    echo "invalid selection: ${MODEL_CHOICE}" >&2
    exit 2
  fi

  MODEL="${MODELS[$((MODEL_CHOICE - 1))]}"
fi

EVENT_OUTPUT="$(
  ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
    "find '${REMOTE_RUN}' -maxdepth 1 -type f -name 'events.out*' -print | sort"
)"

EVENTS=()
while IFS= read -r remote_event; do
  [[ -n "$remote_event" ]] || continue
  EVENTS[${#EVENTS[@]}]="$(basename "${remote_event}")"
done <<< "$EVENT_OUTPUT"

if [[ ${#EVENTS[@]} -eq 0 ]]; then
  echo "no events.out files in logs/rsl_rl/${RUN_REL}" >&2
  exit 1
fi

LOCAL_RUN="${ROOT}/logs/rsl_rl/${RUN_REL}"
mkdir -p "${LOCAL_RUN}"

SCP_SOURCES=()
for event in "${EVENTS[@]}"; do
  SCP_SOURCES+=("${REMOTE_HOST}:${REMOTE_RUN}/${event}")
done
if [[ -n "$MODEL" ]]; then
  SCP_SOURCES+=("${REMOTE_HOST}:${REMOTE_RUN}/${MODEL}")
  echo "Downloading logs/rsl_rl/${RUN_REL}/{${EVENTS[*]},${MODEL}}"
else
  echo "Downloading logs/rsl_rl/${RUN_REL}/{${EVENTS[*]}}"
fi
scp -p -P "${PORT}" \
  -o ControlMaster=auto \
  -o ControlPersist=600 \
  -o "ControlPath=${CONTROL_PATH}" \
  "${SCP_SOURCES[@]}" \
  "${LOCAL_RUN}/"