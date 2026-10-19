#!/usr/bin/env bash
# List local TensorBoard runs and open a chosen subset.
# Usage:
#   scripts/tensorboard.sh
#   scripts/tensorboard.sh --port 6007
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_ROOT="${ROOT}/logs/rsl_rl"

if [[ ! -d "$LOG_ROOT" ]]; then
  echo "no logs found at ${LOG_ROOT}" >&2
  exit 1
fi

RUN_OUTPUT="$(
  find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type d -print | sort
)"

RUNS=()
while IFS= read -r run; do
  [[ -n "$run" ]] || continue
  if ! find "$run" -maxdepth 1 -type f -name 'events.out*' -print -quit | grep -q .; then
    continue
  fi
  RUNS[${#RUNS[@]}]="$run"
done <<< "$RUN_OUTPUT"

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "no TensorBoard runs found under ${LOG_ROOT}" >&2
  exit 1
fi

echo "Available runs:"
for ((i = 0; i < ${#RUNS[@]}; i++)); do
  relative="${RUNS[$i]#"${LOG_ROOT}/"}"
  printf "  %2d) %s\n" "$((i + 1))" "$relative"
done

read -r -p "Choose runs (e.g. 1,2,5): " CHOICE

IFS=',' read -r -a INDEXES <<< "$CHOICE"
if [[ ${#INDEXES[@]} -eq 0 || -z "${CHOICE//[[:space:]]/}" ]]; then
  echo "no runs selected" >&2
  exit 2
fi

SELECTED=()
SPEC=""
for raw in "${INDEXES[@]}"; do
  idx="${raw//[[:space:]]/}"
  if [[ -z "$idx" ]]; then
    echo "invalid selection: ${CHOICE}" >&2
    exit 2
  fi
  if [[ ! "$idx" =~ ^[0-9]+$ ]] \
    || ((idx < 1 || idx > ${#RUNS[@]})); then
    echo "invalid selection: ${idx}" >&2
    exit 2
  fi
  run="${RUNS[$((idx - 1))]}"
  name="${run#"${LOG_ROOT}/"}"
  SELECTED+=("$name")
  if [[ -n "$SPEC" ]]; then
    SPEC+=","
  fi
  SPEC+="${name}:${run}"
done

echo "Launching TensorBoard for:"
printf '  %s\n' "${SELECTED[@]}"

if command -v tensorboard >/dev/null 2>&1; then
  exec tensorboard --logdir_spec "$SPEC" "$@"
fi

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -m tensorboard.main --logdir_spec "$SPEC" "$@"
