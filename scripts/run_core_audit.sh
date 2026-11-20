#!/usr/bin/env bash
# Core Path B audit, in dependency order.
#
# Usage:
#   SEEDS="42" ./scripts/run_core_audit.sh
#   SEEDS="42 43 44" ./scripts/run_core_audit.sh   # full seed replication
#
# Run this on the GPU machine after:
#   python -m unittest discover -s tests -v
#   RUN_PRETRAINED_TESTS=1 python -m unittest tests.test_torch_integration -v

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHON="${PYTHON:-.venv/bin/python}"
export DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-50}"
PRETRAINED_EPOCHS="${PRETRAINED_EPOCHS:-40}"

mkdir -p results/logs
LOG="results/logs/core_audit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

CORE_MODELS="mobilenetv3_large proposed_large_32 mobilenetv3_large_none"
ATTENTION_MODELS="mobilenetv3_large_se_r16 mobilenetv3_large_se_r32 proposed_large_16 proposed_large_32 mobilenetv3_small_se_r16 mobilenetv3_small_se_r32 proposed_small_16 proposed_small_32"

echo "=== core audit ==="
echo "seeds: $SEEDS | epochs: $EPOCHS | pretrained epochs: $PRETRAINED_EPOCHS"
echo "log: $LOG"

echo "=== E1a: scratch, core models ==="
MODELS="$CORE_MODELS" SEEDS="$SEEDS" EPOCHS="$EPOCHS" MODE=scratch ./scripts/run_path_b.sh

echo "=== E1b: ImageNet backbone, core models ==="
MODELS="$CORE_MODELS" SEEDS="$SEEDS" EPOCHS="$PRETRAINED_EPOCHS" MODE=pretrained ./scripts/run_path_b.sh

echo "=== E2: no-background ablation, core models ==="
MODELS="$CORE_MODELS" SEEDS="$SEEDS" EPOCHS="$EPOCHS" MODE=nobg ./scripts/run_path_b.sh

echo "=== E3: border-only probe ==="
MODELS="mobilenetv3_small mobilenetv3_large" SEEDS="$SEEDS" EPOCHS=30 MODE=border ./scripts/run_path_b.sh

echo "=== E5 inputs: equal-r attention variants (scratch) ==="
MODELS="$ATTENTION_MODELS" SEEDS="$SEEDS" EPOCHS="$EPOCHS" MODE=scratch ./scripts/run_path_b.sh

echo "=== E4: leaf-crop intervention ==="
MODELS="$CORE_MODELS" SEEDS="$SEEDS" MODE=crop ./scripts/run_path_b.sh

echo "=== aggregate and equal-r comparisons ==="
MODELS="$CORE_MODELS" SEEDS="$SEEDS" MODE=aggregate ./scripts/run_path_b.sh

echo "=== done ==="
echo "metrics: results/metrics/*.json"
echo "summary: results/metrics_summary.csv"
echo "log: $LOG"
