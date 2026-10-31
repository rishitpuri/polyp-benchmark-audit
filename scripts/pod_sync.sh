#!/usr/bin/env bash
# Pull results off the pod. RunPod pods behind NAT cannot push to a laptop, so
# the laptop must poll.
#
#   bash scripts/pod_sync.sh <host> <port> [once|watch] [ssh_key]
#
# `watch` polls every 3 minutes until interrupted. Run it for the whole job.
#
# This exists because a pod created with volumeInGb=0 has no persistent
# storage: /workspace is the container overlay, and terminating the pod
# destroys every result on it. A previous phase-1 run was lost exactly this
# way. Checkpoints (*.pth) are deliberately NOT pulled -- they are large and
# reproducible; the per-image CSVs and run summaries are what the analysis
# needs.
set -euo pipefail

HOST="${1:?usage: pod_sync.sh <host> <port> [once|watch] [ssh_key]}"
PORT="${2:?usage: pod_sync.sh <host> <port> [once|watch] [ssh_key]}"
MODE="${3:-once}"
KEY="${4:-$HOME/.ssh/id_ed25519}"
REMOTE="/workspace/attempt2"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL=180

pull () {
  mkdir -p "${LOCAL}/outputs/paper2" "${LOCAL}/runs_paper2"
  local opts=(-rlptz --no-o --no-g
    -e "ssh -i ${KEY} -p ${PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20")
  rsync "${opts[@]}" \
    "root@${HOST}:${REMOTE}/outputs/paper2/" "${LOCAL}/outputs/paper2/" || return 1
  rsync "${opts[@]}" --exclude '*.pth' \
    "root@${HOST}:${REMOTE}/runs_paper2/" "${LOCAL}/runs_paper2/" || return 1
  local csvs summaries
  csvs=$(find "${LOCAL}/outputs/paper2" -name 'perimage_*.csv' 2>/dev/null | wc -l | tr -d ' ')
  summaries=$(find "${LOCAL}/runs_paper2" -name 'summary.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%H:%M:%S)  pulled: ${csvs} per-image CSVs, ${summaries} run summaries"
}

if [ "$MODE" = "watch" ]; then
  echo "polling ${HOST}:${PORT} every ${INTERVAL}s -- Ctrl-C to stop"
  while true; do
    pull || echo "$(date +%H:%M:%S)  pull failed (pod down or restarting); will retry"
    sleep "$INTERVAL"
  done
else
  pull
fi
