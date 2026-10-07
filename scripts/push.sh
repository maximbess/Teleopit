#!/usr/bin/env bash
# Push the current branch to the devbox checkout over SSH.
# A fresh SSH session is opened first. The remote repo is set to
# receive.denyCurrentBranch=updateInstead before the push, so a
# fast-forward updates the branch that is checked out there.
# Usage:
#   scripts/push.sh
#   scripts/push.sh root@host:/path/to/Teleopit --port 34211
set -euo pipefail

REMOTE="root@172.16.78.10:/vladyslavdenysiuk/Teleopit"
PORT="34211"
if [[ $# -gt 0 && "$1" != --* ]]; then
  REMOTE="${1%/}"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$REMOTE" != *:* ]]; then
  echo "remote must be user@host:/path/to/Teleopit" >&2
  exit 2
fi
if [[ -z "$PORT" ]]; then
  echo "missing --port" >&2
  exit 2
fi

REMOTE_HOST="${REMOTE%%:*}"
REMOTE_ROOT="${REMOTE#*:}"
if [[ "${REMOTE_ROOT}" != /* ]]; then
  echo "remote path must be absolute: ${REMOTE_ROOT}" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" == "HEAD" ]]; then
  echo "detached HEAD; check out a branch before pushing" >&2
  exit 2
fi

CONTROL_PATH="/tmp/teleopit-push-${UID}-%C"
SSH_OPTIONS=(
  -p "${PORT}"
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=${CONTROL_PATH}"
)

# Drop a previous multiplexed session so this run authenticates again.
ssh "${SSH_OPTIONS[@]}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true

echo "Opening a fresh connection to ${REMOTE_HOST} port ${PORT}"
ssh "${SSH_OPTIONS[@]}" -fN "${REMOTE_HOST}"

echo "Setting receive.denyCurrentBranch=updateInstead in ${REMOTE_ROOT}"
ssh "${SSH_OPTIONS[@]}" "${REMOTE_HOST}" \
  "git -C '${REMOTE_ROOT}' config receive.denyCurrentBranch updateInstead"

echo "Pushing ${BRANCH} to ${REMOTE_HOST}:${REMOTE_ROOT}"
git -c core.sshCommand="ssh ${SSH_OPTIONS[*]}" \
  push "ssh://${REMOTE_HOST}:${PORT}${REMOTE_ROOT}" "HEAD:${BRANCH}"
