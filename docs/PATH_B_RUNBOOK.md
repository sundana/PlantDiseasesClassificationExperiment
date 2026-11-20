# Path B runbook

This patch turns the existing thesis repository into the experiment set described
in `REVISION_PATH_B.md`. It is backward compatible: the original commands still
work, and no existing checkpoint is overwritten unless you run without `--run-tag`.

## 1. What was added

| File | Purpose |
|---|---|
| `src/constants.py` | One label space and PlantDoc mapping for all scripts |
| `src/audit.py` | Metrics, border mask, VOC parsing, leaf cropping, checkpoint loading, McNemar, bootstrap, TOST |
| `src/models/factory.py` | Model factory with optional ImageNet backbone initialisation |
| `src/data_setup.py` | `exclude_classes` argument, applied before the stratified split |
| `src/utils.py` | Optional `model.run_tag` in checkpoint names |
| `scripts/train.py` | `--pretrained`, `--lr`, `--exclude-class`, `--run-tag`, `--eval-plantdoc`, `metrics.json` |
| `scripts/audit_border.py` | E3 border-only probe |
| `scripts/audit_crop.py` | E4 leaf-crop intervention |
| `scripts/aggregate_seeds.py` | Mean ± SD across seeds; paired comparison with TOST |
| `scripts/run_path_b.sh` | Runs the full matrix |
| `tests/test_audit_smoke.py` | CPU tests for metrics, VOC, cropping, statistics |
| `tests/test_torch_integration.py` | Model factory and class-filter tests (skipped without torch) |
| `compute_metrics.py` | Copied from the manuscript folder so the metric test can run |

## 2. Preconditions

**Data layout**

The image side supports both the flat merged layout and the original split
layout:

```
data/plantvillage/<29 class directories>/
data/plantdoc/<20 class directories>/              # merged, flat, as before
data/plantdoc/train/<20 class directories>/       # or the original split layout
data/plantdoc/test/<20 class directories>/
```

For E4, point `--annotations` at the root of the PlantDoc object-detection
release (the `TRAIN/` and `TEST/` folders containing the XMLs). The index is
split-aware, so identical filenames in train and test do not collide, and
`foo.JPG.jpg` correctly matches `foo.JPG.xml`. A flat annotations directory is
also accepted.

`--annotations` accepts one or more roots; the script indexes every `*.xml`
recursively and tolerates `_coll` suffixes produced by `merge_plantdoc.py`.
In the public release, image filenames are unique within each split, so
split-aware lookup is sufficient; check `annotation_coverage` in the metrics
JSON and point `--annotations` at the `TRAIN/`/`TEST/` root if it is low.

**Environment**

The exact environment and dataset commands used on this machine are in
[`DATA_SETUP.md`](DATA_SETUP.md). Short version:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-pathb.txt
```

**ImageNet weights** (one-time, needs network):

```bash
python - <<'PY'
from torchvision import models
models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
PY
```

If the GPU node is offline, run this on a machine with network access and copy
`$TORCH_HOME` to the same path on the node.

## 3. Smoke tests

```bash
python -m unittest discover -s tests -v
```

Expected: 12 tests pass, 1 skipped (the `compute_metrics` comparison now runs
because the file is included). Then, once on the GPU node:

```bash
RUN_PRETRAINED_TESTS=1 python -m unittest tests.test_torch_integration -v
```

This downloads the weights and checks that the custom MobileNetV3 backbones load
them with no unmapped tensor. Do not launch the full matrix before this passes.

## 4. Run matrix

The runner uses the seven representative models from the manuscript:

```
mobilenetv3_large        Large SE r=4
proposed_large_32        Large CBAM r=32
mobilenetv3_large_none   Large, no attention
mobilenetv3_small        Small SE r=4
mobilenetv2
shufflenetv2
efficientnet_b0
```

### E1a — scratch baseline (3 seeds)

```bash
MODE=scratch EPOCHS=50 SEEDS="42 43 44" ./scripts/run_path_b.sh
```

### E1b — ImageNet-pretrained backbone (3 seeds)

```bash
MODE=pretrained EPOCHS=40 SEEDS="42 43 44" ./scripts/run_path_b.sh
```

Pretrained runs use `--lr 1e-4`. Attention modules and the classifier head are
randomly initialised; only the backbone is pretrained. This is deliberate and
must be stated in the paper (see §6).

### E2 — no-background-class ablation (3 seeds)

```bash
MODE=nobg EPOCHS=50 SEEDS="42 43 44" ./scripts/run_path_b.sh
```

This trains 28-class models and evaluates them on the same 1,783 field images.
`train.py` maps predictions through the 28-class label list, so the field metrics
remain comparable with the 29-class runs.

### E3 — border-only probe (2 models × 3 seeds)

```bash
MODE=border EPOCHS=30 SEEDS="42 43 44" ./scripts/run_path_b.sh
```

Report accuracy against chance (3.45% for 29 classes), not against 0%.

### E4 — leaf-crop intervention

```bash
MODE=crop SEEDS="42 43 44" ./scripts/run_path_b.sh
```

This runs inference only, on both scratch and pretrained checkpoints. If the
annotations are not found, the script reports 0% coverage and every crop equals
the full image; check that line before interpreting the result.

### E5 — aggregation and SE/CBAM comparison

```bash
MODE=aggregate SEEDS="42 43 44" ./scripts/run_path_b.sh
```

Outputs:

- `results/metrics_summary.csv` — mean ± SD per model, initialisation and class count.
- `results/metrics/compare_SE4_vs_CBAM32_s*.json` — paired difference, 95% CI,
  exact McNemar, TOST decision at ±1 pp.

### E6 (optional) — fine-tune a pretrained checkpoint

The existing adaptation script already accepts a PlantVillage checkpoint:

```bash
python scripts/train_domain_adapt.py \
  --model mobilenetv3_large --method finetune \
  --pretrained_ckpt results/checkpoints/<pretrained-checkpoint>.pth \
  --epochs 50 --lr 0.001 --seed 42
```

Run this for one or two models only. It gives a pretrained fine-tuned row that
answers the reviewer question "would pretrained fine-tuning close the gap
faster?" without redoing the full adaptation grid.

## 5. Outputs and where they go in the paper

```
results/
  checkpoints/<display>__<run-tag>_epoch_<N>.pth
  metrics/<run-tag>.json
  predictions/<run-tag>_plantdoc.csv
  predictions/crop_<run-tag>.csv
  metrics_summary.csv
```

| Manuscript item | Source |
|---|---|
| Table 2 (lab accuracy + complexity) | `lab.accuracy` in `metrics/*.json`; complexity from `scripts/measure_complexity.py` |
| Table 3 (field accuracy, macro P/R/F1) | `field.*` in `metrics/*.json` |
| Fig. 4 (lab vs field dumbbell) | `lab.accuracy` and `field.accuracy` |
| Table 4 / Fig. 5a (background-class ablation) | tags containing `__nobg__` vs `__scratch__` |
| Fig. 5b (border probe) | `metrics/border_*.json`, `lab.accuracy` vs `chance_accuracy` |
| Fig. 5c (crop intervention) | `annotated_subset.delta_accuracy_pp` and McNemar in `metrics/crop_*.json` |
| Table 5 (adaptation) | existing fine-tuning results; rerun only if the protocol changes |

Use `metrics_summary.csv` for the text and table cells. Do not copy numbers from
the console output into the manuscript.

## 6. Caveats that must appear in the methods

1. **The custom backbone is MobileNetV3-style, not a byte-for-byte copy of
   torchvision.** Two differences matter for the pretrained condition:
   - the custom block always has a 1x1 expand convolution, whereas torchvision
     omits it when the expanded width equals the input width (Large block 0,
     Small block 0); those convs are initialised to the identity;
   - the custom block adds a projection shortcut for channel-changing stride-1
     blocks (Large block 10, Small block 6), whereas torchvision uses no
     residual there; those shortcuts are zero-initialised;
   - the penultimate Large block originally used `672` expansion; it was
     corrected to `960` to match MobileNetV3 and to let the ImageNet backbone
     load completely. This changes the Large parameter count, so regenerate the
     complexity table before submission.

   With these two adjustments, a pretrained no-attention model reproduces the
   torchvision convolutional backbone at initialisation (its SE modules are
   intentionally absent). Record this in the methods and keep the `model_info`
   block from `metrics.json` as evidence.

2. **Pretrained attention modules are random.** For the custom SE/CBAM/none
   variants, only the feature extractor is initialised from ImageNet. This keeps
   SE and CBAM on equal footing. The torchvision MobileNetV3 Large/Small models
   are fully pretrained and are reported as separate reference rows, not as the
   attention comparison.

3. **Batch-norm settings are matched to torchvision** (`eps=0.001`,
   `momentum=0.01`) in pretrained runs, because the copied running statistics
   were estimated with those settings. When evaluating a pretrained checkpoint
   with `audit_crop.py`, pass `--bn-eps 0.001 --bn-momentum 0.01`; the runner
   does this automatically for the `imagenet` condition. `train.py` builds the
   model with the correct settings already.

4. **Two pretrained tracks.** If space allows, report (a) torchvision reference
   models and (b) custom variants with a pretrained backbone. Do not pool them in
   one table without labelling the difference.

5. **PlantDoc annotations.** The crop intervention is only as good as the box
   coverage. Report `annotation_coverage` from the metrics JSON.

6. **No-background checkpoints have 28 outputs.** `audit_crop.py` detects this and
   uses the 28-class label order; do not pass them to `validate_plantdoc.py`,
   which assumes 29.

## 7. Known issue in the current manuscript (fix before submission)

`Algorithm 1` and `Fig. 4` describe attention as applied to the expanded feature
map **before** the pointwise projection. The code applies it **after** the
projection and batch normalisation, on the block's output channels:

```
out = bn3(conv3(out))
out = attention(out)          # C_out channels, after projection
out = out + shortcut(x)
```

This does not affect the measured numbers, but it changes the parameter-count
argument (`C` is the block output width, not the expansion width) and it is a
reproducibility problem because the code is linked in the paper. Either:

- **preferred (no new runs):** move the attention lines after the projection in
  Algorithm 1, state that `C` is the block output width, and check that Fig. 4
  matches; or
- re-run all models with attention before the projection (not recommended under
  the current time constraint).

Also verify that the text "inserted after the depthwise convolution" in
Section 3.5 is changed accordingly.

## 8. Time estimates (single A100)

| Block | Runs | Estimate |
|---|---|---|
| E1a scratch, 7 models × 3 seeds | 21 | 7–9 h |
| E1b pretrained, 7 × 3 | 21 | 7–9 h |
| E2 no-background, 7 × 3 | 21 | 7–9 h |
| E3 border, 2 × 3 | 6 | 1 h |
| E4 crop, inference | 14 × 2 passes | < 30 min |
| E5 aggregation | — | minutes |

A 12-day sprint can drop to 5 models and 2 seeds, but keep E4.

## 9. PAA revision: target-domain fine-tune

`scripts/audit_finetune.py` fine-tunes a PlantVillage checkpoint on the official
PlantDoc training split (1,614 images, 20 mapped classes), holds out 15% of
train for early stopping, and evaluates once on the official 169-image test
split. `scripts/run_finetune_matrix.sh` runs the three core backbones for seeds
42 and 43.

```bash
./scripts/run_finetune_matrix.sh
```

Outputs:

- `results/metrics/finetune_<model>__s<seed>.json`
- `results/predictions/finetune_<model>__s<seed>.csv`
- `results/checkpoints/finetune_<model>__s<seed>_best.pth`
- `results/logs/finetune_<model>__s<seed>.log`

The script sets the torchvision BatchNorm settings (`eps=0.001`,
`momentum=0.01`) and reuses the classifier rows of the 20 mapped classes from
the 29-class checkpoint. Do not evaluate on the training split: the fine-tune
uses the official test split. The manuscript uses seeds 42 and 43; seed 44 is a
robustness check and changes the means.

## 10. PAA table regeneration

`scripts/report_paa_tables.py` reads the prediction files and reports both the
all-image and official-test-split metrics, plus the four attention pairs and the
fine-tune runs:

```bash
.venv/bin/python scripts/report_paa_tables.py
```

Output: `results/paa_tables.json`. The figure script
`figures/gen_fig_paa.py` reads that file and writes the PAA figures to
`figures/` and `paper/`.
