#!/usr/bin/env bash
# Select a local ladder checkpoint and run it in mjlab's Viser viewer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT}/logs/rsl_rl/g1_ladder_rl"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
  echo "no local ladder logs found at ${CHECKPOINT_ROOT}" >&2
  echo "run scripts/sync_logs.sh first" >&2
  exit 1
fi

CHECKPOINT_OUTPUT="$(
  find "$CHECKPOINT_ROOT" -type f -name 'model_*.pt' -print | sort
)"

CHECKPOINTS=()
while IFS= read -r checkpoint; do
  [[ -n "$checkpoint" ]] || continue
  CHECKPOINTS[${#CHECKPOINTS[@]}]="$checkpoint"
done <<< "$CHECKPOINT_OUTPUT"

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "no model_*.pt checkpoints found under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

echo "Available local ladder checkpoints:"
for ((i = 0; i < ${#CHECKPOINTS[@]}; i++)); do
  relative="${CHECKPOINTS[$i]#"${ROOT}/"}"
  printf "  %2d) %s\n" "$((i + 1))" "$relative"
done

read -r -p "Choose a checkpoint [1-${#CHECKPOINTS[@]}]: " CHOICE
if [[ ! "$CHOICE" =~ ^[0-9]+$ ]] \
  || ((CHOICE < 1 || CHOICE > ${#CHECKPOINTS[@]})); then
  echo "invalid selection: ${CHOICE}" >&2
  exit 2
fi

CHECKPOINT="${CHECKPOINTS[$((CHOICE - 1))]}"
PYTHON_BIN="${PYTHON:-python}"

echo "Launching Viser at http://localhost:8012"
cd "$ROOT"
exec "$PYTHON_BIN" train_mimic/scripts/play.py \
  --task G1-Ladder-Climb-RL \
  --checkpoint "$CHECKPOINT" \
  --viewer viser \
  "$@"
