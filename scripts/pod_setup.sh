#!/usr/bin/env bash
# One-shot pod bootstrap: upload the project, build the environment, smoke test.
#
#   bash scripts/pod_setup.sh <host> <port> [ssh_key]
#
# Example:
#   bash scripts/pod_setup.sh 213.173.109.89 18497
#
# The RunPod PyTorch image marks its system interpreter as externally managed
# (PEP 668), so a plain `pip install` is refused. A venv with
# --system-site-packages is used instead: the image's torch build is inherited
# untouched, which matters because reinstalling torch pulls a build that can
# mismatch the host driver.
set -euo pipefail

HOST="${1:?usage: pod_setup.sh <host> <port> [ssh_key]}"
PORT="${2:?usage: pod_setup.sh <host> <port> [ssh_key]}"
KEY="${3:-$HOME/.ssh/id_ed25519}"
REMOTE="/workspace/attempt2"
SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25)

echo "== uploading project to ${HOST}:${PORT}${REMOTE}"
# --no-o --no-g: RunPod network volumes are MooseFS mounts that refuse chown,
# and plain `rsync -a` fails with exit 23 on every file as a result.
rsync -rlptz --no-o --no-g --stats \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude '__pycache__' --exclude 'credentials.json' \
  --exclude 'data/provenance' --exclude 'data/cvc-clinicDB-original' \
  --exclude 'paper1_results' --exclude 'runs_paper2' --exclude 'features' \
  --exclude '*.npy' --exclude '.git' --exclude '.DS_Store' \
  "$(dirname "$0")/.."/ "root@${HOST}:${REMOTE}/" | tail -4

echo ""
echo "== building environment"
ssh "${SSH_OPTS[@]}" "root@${HOST}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/attempt2
if [ ! -x /workspace/venv/bin/python ]; then
  python -m venv --system-site-packages /workspace/venv
fi
/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -q -r requirements-pod.txt
/workspace/venv/bin/python - <<'PY'
import torch, segmentation_models_pytorch as smp, albumentations, cv2
assert torch.cuda.is_available(), "CUDA not available"
print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"smp {smp.__version__}  albumentations {albumentations.__version__}  cv2 {cv2.__version__}")
PY
echo "-- split files present:"
ls outputs/splits/training/*.csv | wc -l
echo "-- disk:"
df -h / | tail -1
REMOTE_SCRIPT

echo ""
echo "== smoke test (2 epochs, discarded)"
ssh "${SSH_OPTS[@]}" "root@${HOST}" \
  'cd /workspace/attempt2 && /workspace/venv/bin/python -m polypseg.train \
     --model unet --seed 42 --tag smoke \
     --data-root data/external/cvc-clinicdb \
     --splits outputs/splits/training/clinicdb_seqdisjoint_fold0.csv \
     --out-dir /tmp/smoke --epochs 2 --num-workers 8 2>&1 | tail -3 && rm -rf /tmp/smoke'

cat <<EOF

== ready. Next, in two separate terminals:

   # 1. start the run (detached, survives disconnect)
   ssh -i ${KEY} -p ${PORT} root@${HOST} \\
     'cd ${REMOTE} && PYTHON=/workspace/venv/bin/python nohup bash run_paper2.sh 1 \\
        > outputs/paper2/run_phase1.log 2>&1 &'

   # 2. start the result puller -- DO NOT SKIP THIS
   bash scripts/pod_sync.sh ${HOST} ${PORT} watch

EOF
