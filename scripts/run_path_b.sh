#!/usr/bin/env bash
# Path B experiment runner.
#
# Examples:
#   MODE=scratch EPOCHS=50 SEEDS="42 43 44" ./scripts/run_path_b.sh
#   MODE=pretrained EPOCHS=40 SEEDS="42 43 44" ./scripts/run_path_b.sh
#   MODE=nobg EPOCHS=50 SEEDS="42 43 44" ./scripts/run_path_b.sh
#   MODE=border EPOCHS=30 SEEDS="42 43 44" ./scripts/run_path_b.sh
#   MODE=crop ./scripts/run_path_b.sh
#   MODE=aggregate ./scripts/run_path_b.sh
#   DRY_RUN=1 MODE=scratch ./scripts/run_path_b.sh
#
# Environment:
#   MODE        scratch | pretrained | nobg | border | crop | aggregate | all
#   EPOCHS      training epochs (default 50; border uses 30)
#   SEEDS       space-separated seeds (default "42 43 44")
#   DEVICE      cuda | cpu | auto (default auto)
#   DRY_RUN     1 to print commands without running them
#   PYTHON      python executable (default python)

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-scratch}"
EPOCHS="${EPOCHS:-50}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

# The seven representative models used in the revised manuscript.
DEFAULT_MODELS=(
  mobilenetv3_large          # MobileNetV3 Large, SE r=4
  proposed_large_32          # MobileNetV3 Large, CBAM r=32
  mobilenetv3_large_none     # MobileNetV3 Large, no attention
  mobilenetv3_small          # MobileNetV3 Small, SE r=4
  mobilenetv2
  shufflenetv2
  efficientnet_b0
)

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${MODELS})
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    printf '\n>>> %s\n' "$*"
    "$@"
  fi
}

train_scratch() {
  local model="$1" seed="$2"
  run "$PYTHON" scripts/train.py \
    --model "$model" --epochs "$EPOCHS" --seed "$seed" \
    --pretrained none --device "$DEVICE" --eval-plantdoc \
    --run-tag "${model}__scratch__s${seed}"
}

train_pretrained() {
  local model="$1" seed="$2"
  run "$PYTHON" scripts/train.py \
    --model "$model" --epochs "$EPOCHS" --seed "$seed" \
    --pretrained imagenet --lr 1e-4 --device "$DEVICE" --eval-plantdoc \
    --run-tag "${model}__imagenet__s${seed}"
}

train_nobg() {
  local model="$1" seed="$2"
  run "$PYTHON" scripts/train.py \
    --model "$model" --epochs "$EPOCHS" --seed "$seed" \
    --pretrained none --device "$DEVICE" --eval-plantdoc \
    --exclude-class Background_without_leaves \
    --run-tag "${model}__nobg__s${seed}"
}

border_probe() {
  local model="$1" seed="$2"
  run "$PYTHON" scripts/audit_border.py \
    --model "$model" --epochs 30 --seed "$seed" --device "$DEVICE" \
    --eval-plantdoc --run-tag "border_${model}__s${seed}"
}

crop_probe() {
  local condition="$1" model="$2" seed="$3"
  local tag="${model}__${condition}__s${seed}"
  local checkpoint
  checkpoint="$(ls -t results/checkpoints/*"${tag}"*.pth 2>/dev/null | head -n 1 || true)"
  if [[ -z "$checkpoint" ]]; then
    echo "[skip] no checkpoint for ${tag}"
    return 0
  fi
  local bn_args=()
  if [[ "$condition" == "imagenet" ]]; then
    bn_args=(--bn-eps 0.001 --bn-momentum 0.01)
  fi
  run "$PYTHON" scripts/audit_crop.py \
    --model "$model" --weight "$checkpoint" --device "$DEVICE" \
    --out "results/predictions/crop_${tag}.csv" \
    --metrics-json "results/metrics/crop_${tag}.json" \
    "${bn_args[@]}"
}

case "$MODE" in
  scratch)
    for seed in $SEEDS; do for model in "${MODELS[@]}"; do train_scratch "$model" "$seed"; done; done
    ;;
  pretrained)
    for seed in $SEEDS; do for model in "${MODELS[@]}"; do train_pretrained "$model" "$seed"; done; done
    ;;
  nobg)
    for seed in $SEEDS; do for model in "${MODELS[@]}"; do train_nobg "$model" "$seed"; done; done
    ;;
  border)
    for seed in $SEEDS; do
      for model in mobilenetv3_small mobilenetv3_large; do border_probe "$model" "$seed"; done
    done
    ;;
  crop)
    for seed in $SEEDS; do
      for model in "${MODELS[@]}"; do
        for condition in scratch imagenet; do crop_probe "$condition" "$model" "$seed"; done
      done
    done
    ;;
  aggregate)
    run "$PYTHON" scripts/aggregate_seeds.py aggregate \
      --metrics-glob "results/metrics/*.json" --out results/metrics_summary.csv
    for seed in $SEEDS; do
      for condition in scratch imagenet; do
        pred_a="results/predictions/mobilenetv3_large_se_r16__${condition}__s${seed}_plantdoc.csv"
        pred_b="results/predictions/proposed_large_16__${condition}__s${seed}_plantdoc.csv"
        if [[ -f "$pred_a" && -f "$pred_b" ]]; then
          run "$PYTHON" scripts/aggregate_seeds.py compare \
            --a "$pred_a" --b "$pred_b" \
            --name-a "Large SE r=16" --name-b "Large CBAM r=16" --margin 1.0 \
            --out "results/metrics/compare_SE16_vs_CBAM16_${condition}_s${seed}.json"
        else
          echo "[skip] missing equal-r prediction CSVs for ${condition}, seed ${seed}"
        fi
      done
    done
    ;;
  all)
    MODE=scratch "$0"
    MODE=pretrained "$0"
    MODE=nobg "$0"
    MODE=border "$0"
    MODE=crop "$0"
    MODE=aggregate "$0"
    ;;
  *)
    echo "unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
