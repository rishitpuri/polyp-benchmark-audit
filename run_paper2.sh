#!/usr/bin/env bash
# Paper 2 GPU experiments. Run from the attempt2 project root.
#
#   bash run_paper2.sh 1      # experiment 1 only (CVC-ClinicDB, 50 runs)
#   bash run_paper2.sh 2      # experiment 2 only (Kvasir-SEG, 50 runs)
#   bash run_paper2.sh        # both
#
# Experiment 1 is the headline: the same five architectures trained on
# CVC-ClinicDB under a random frame split (the published protocol, in which
# 100% of test sequences also appear in training) and under a split that
# keeps the 29 official source sequences disjoint (0%). The gap is the
# inflation attributable to leakage, to be read against sigma_run = 0.0053.
#
# Experiment 2 asks the same question of Kvasir-SEG, whose folds in paper 1
# were image-level. Both arms here share one fold construction so that the
# only difference is whether near-duplicates are held together.
#
# Resumable: any run with a summary.json and any evaluation with an existing
# CSV is skipped, so an interrupted job restarts without losing work.
set -euo pipefail

PHASE="${1:-all}"
# Override on a pod that keeps torch in the system interpreter, e.g.
#   PYTHON=/workspace/venv/bin/python bash run_paper2.sh 1
PYTHON="${PYTHON:-python}"
MODELS=("unet" "unetpp" "segformer" "segformer_b2" "unet_pvt")
FOLDS=(0 1 2 3 4)
SEED=42
SIZE=352
BATCH=16
EPOCHS=60
WORKERS=8

RUNS_DIR="runs_paper2"
OUT_DIR="outputs/paper2"
SPLIT_DIR="outputs/splits/training"
mkdir -p "$RUNS_DIR" "$OUT_DIR"

# arm_name : data_root : external sets to also evaluate on
# CVC-300 is deliberately absent: it is a strict subset of CVC-ColonDB.
run_arm () {
  local arm="$1" data_root="$2" externals="$3" indomain="$4"
  echo ""
  echo "######## arm: ${arm} (data root ${data_root}, in-domain ${indomain}) ########"
  for model in "${MODELS[@]}"; do
    for k in "${FOLDS[@]}"; do
      local split="${SPLIT_DIR}/${arm}_fold${k}.csv"
      [ -f "$split" ] || { echo "!! missing split ${split}"; continue; }
      # this is exactly the directory name train.py derives from --tag
      local run="${model}_seed${SEED}_${arm}_fold${k}"

      if [ -f "${RUNS_DIR}/${run}/summary.json" ]; then
        echo "== skip ${run} (already complete)"
      else
        echo "== train ${run}"
        "$PYTHON" -m polypseg.train \
          --model "$model" --seed "$SEED" --tag "${arm}_fold${k}" \
          --data-root "$data_root" --splits "$split" --out-dir "$RUNS_DIR" \
          --size "$SIZE" --batch-size "$BATCH" --epochs "$EPOCHS" \
          --num-workers "$WORKERS"
      fi

      local ckpt="${RUNS_DIR}/${run}/best.pth"
      [ -f "$ckpt" ] || { echo "!! missing ${ckpt}"; continue; }

      if [ ! -f "${OUT_DIR}/perimage_${run}.csv" ]; then
        echo "== eval ${run}"
        # shellcheck disable=SC2086
        POLYPSEG_INDOMAIN_NAME="$indomain" "$PYTHON" -m polypseg.evaluate \
          --checkpoint "$ckpt" --model "$model" --size "$SIZE" \
          --data-root "$data_root" --splits "$split" --split test \
          --out "${OUT_DIR}/perimage_${run}.csv" \
          --external ${externals}
      fi
    done
  done
}

# External sets carry distinct names so they cannot collide with the
# relabelled in-domain rows. CVC-300 is deliberately absent throughout.
CLINICDB_EXT="kvasir-seg-ext=data/Kvasir-SEG/ cvc-colondb=data/external/cvc-colondb/ etis-larib=data/external/etis-larib/"
KVASIR_EXT="cvc-clinicdb-ext=data/external/cvc-clinicdb/ cvc-colondb=data/external/cvc-colondb/ etis-larib=data/external/etis-larib/"

if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
  echo "===== EXPERIMENT 1: CVC-ClinicDB, random vs sequence-disjoint ====="
  run_arm "clinicdb_random"      "data/external/cvc-clinicdb" "$CLINICDB_EXT" "cvc-clinicdb"
  run_arm "clinicdb_seqdisjoint" "data/external/cvc-clinicdb" "$CLINICDB_EXT" "cvc-clinicdb"
fi

if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
  echo "===== EXPERIMENT 2: Kvasir-SEG, image-level vs grouped ====="
  run_arm "kvasir_imagelevel"      "data/Kvasir-SEG" "$KVASIR_EXT" "kvasir-seg"
  run_arm "kvasir_grouped_tau0.90" "data/Kvasir-SEG" "$KVASIR_EXT" "kvasir-seg"
fi

echo ""
echo "== complete: $(ls ${OUT_DIR}/perimage_*.csv 2>/dev/null | wc -l) per-image result files"
