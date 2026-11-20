#!/usr/bin/env bash
# Fine-tune the three core audit models on PlantDoc (official split), two seeds.
#
# This is the positive-result experiment for the PAA revision.  It takes the
# PlantVillage-pretrained audit checkpoints, fine-tunes on the 1,614 training
# images of the 20 mapped PlantDoc classes, holds out 15% of *train* for early
# stopping, and evaluates once on the official 169-image test split.
#
# Usage:
#   ./scripts/run_finetune_matrix.sh
#   PYTHON=.venv/bin/python ./scripts/run_finetune_matrix.sh
#
# Outputs:
#   results/metrics/finetune_<model>__s<seed>.json
#   results/predictions/finetune_<model>__s<seed>.csv
#   results/checkpoints/finetune_<model>__s<seed>_best.pth
#   results/logs/finetune_<model>__s<seed>.log

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-25}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
mkdir -p results/logs

run_one() {
    local model="$1"
    local seed="$2"
    local checkpoint="$3"
    local tag="finetune_${model}__s${seed}"
    local log="results/logs/${tag}.log"

    if [[ ! -f "$checkpoint" ]]; then
        echo "missing checkpoint: $checkpoint" >&2
        exit 1
    fi

    echo ">>> ${tag}"
    "$PYTHON" scripts/audit_finetune.py \
        --model "$model" \
        --checkpoint "$checkpoint" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --run-tag "$tag" 2>&1 | tee "$log"
}

run_one mobilenetv3_large 42 \
    "results/checkpoints/MobileNetV3_Large_SE_r4__mobilenetv3_large__imagenet__s42_epoch_23.pth"
run_one mobilenetv3_large 43 \
    "results/checkpoints/MobileNetV3_Large_SE_r4__mobilenetv3_large__imagenet__s43_epoch_34.pth"

run_one proposed_large_32 42 \
    "results/checkpoints/MobileNetV3_Large_CBAM_r32__proposed_large_32__imagenet__s42_epoch_26.pth"
run_one proposed_large_32 43 \
    "results/checkpoints/MobileNetV3_Large_CBAM_r32__proposed_large_32__imagenet__s43_epoch_20.pth"

run_one mobilenetv3_large_none 42 \
    "results/checkpoints/MobileNetV3_Large_NoAttn__mobilenetv3_large_none__imagenet__s42_epoch_15.pth"
run_one mobilenetv3_large_none 43 \
    "results/checkpoints/MobileNetV3_Large_NoAttn__mobilenetv3_large_none__imagenet__s43_epoch_26.pth"

echo "All fine-tune runs finished."
