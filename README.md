# MobileNetV3 attention variants and the Path B audit

This repository contains the code for a controlled audit of lightweight
plant-disease classifiers. It has two layers:

1. **The original model code** — MobileNetV3 Large/Small with Squeeze-and-
   Excitation (SE), Convolutional Block Attention Module (CBAM) or no
   attention, plus MobileNetV2, ShuffleNetV2 and EfficientNet-B0 baselines.
2. **The Path B audit** — a reproducible set of interventions that decompose
   the laboratory-to-field gap: ImageNet-pretrained control, background-class
   ablation, border-only probe, paired leaf-crop intervention, equal-complexity
   attention comparison, and target-domain fine-tuning.

The associated manuscript is *When the Background Becomes the Diagnosis: A
Controlled Audit of Shortcut Learning in Mobile Plant Disease Classifiers*.
The audit uses two seeds (42 and 43); seed 44 is available for robustness
checks but is **not** part of the manuscript numbers.

## 1. Environment

Python 3.12 and a CUDA-capable GPU are recommended. The reference environment
is `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` on an RTX 5060 Ti (16 GB).

```bash
cd PlantDiseasesClassificationExperiment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-pathb.txt
```

Activate the environment once per shell, or pass the interpreter explicitly:

```bash
source .venv/bin/activate
# or:
export PYTHON=.venv/bin/python
```

The runner uses `PYTHON` (default `python`), so setting it avoids
`python: command not found` on machines where the venv is not activated.

## 2. Data

The datasets are public. The exact download and layout commands are in
[`docs/DATA_SETUP.md`](docs/DATA_SETUP.md). In short:

- **PlantVillage**: HuggingFace `Project-AgML/plant_village_classification`
  (39 classes). `scripts/prepare_plantvillage.py` extracts the 29 classes used
  in the audit (41,879 images, including 1,143 `Background_without_leaves`).
- **PlantDoc**: `pratikkayal/PlantDoc-Dataset` (images) and
  `pratikkayal/PlantDoc-Object-Detection-Dataset` (Pascal VOC boxes). The audit
  uses the official split: 1,614 train / 169 test images in the 20 mapped
  classes. `data/plantdoc/{train,test}` and `data/plantdoc_annotations` are
  symlinks created by the data-setup script.

`src/config.py` resolves `DATA_DIR` inside this directory. Keep the symlinks
in place; do not copy the datasets into the repository.

## 3. Reproduce the manuscript

Run from the repository root. The commands below reproduce the two-seed audit
in the manuscript. `MODELS` is restricted to the three core Large variants so
that the output matches the manuscript tables; omit it to run the seven-model
comparison used in the earlier thesis version.

```bash
export PYTHON=.venv/bin/python
export MODELS="mobilenetv3_large proposed_large_32 mobilenetv3_large_none"

# E1a: scratch baseline (50 epochs)
MODE=scratch EPOCHS=50 SEEDS="42 43" ./scripts/run_path_b.sh

# E1b: ImageNet-pretrained backbone (40 epochs)
MODE=pretrained EPOCHS=40 SEEDS="42 43" ./scripts/run_path_b.sh

# E2: 28-class background ablation (50 epochs)
MODE=nobg EPOCHS=50 SEEDS="42 43" ./scripts/run_path_b.sh

# E3: border-only probe (30 epochs; Large and Small SE)
MODE=border EPOCHS=30 SEEDS="42 43" ./scripts/run_path_b.sh

# E4: paired leaf-crop intervention (uses the trained checkpoints)
MODE=crop SEEDS="42 43" ./scripts/run_path_b.sh

# E5: equal-complexity SE vs CBAM pairs (seed 42)
# The compare JSONs are produced by scripts/aggregate_seeds.py compare; see
# docs/PATH_B_RUNBOOK.md for the four commands.

# Aggregate two-seed means and SDs
MODE=aggregate SEEDS="42 43" ./scripts/run_path_b.sh
```

Every training run writes `results/metrics/<run-tag>.json` and a PlantDoc
prediction CSV under `results/predictions/`. The aggregate step writes
`results/metrics_summary.csv`; the manuscript tables are derived from these
files.

### Positive-result experiment: fine-tune on PlantDoc

The PAA revision adds a target-domain fine-tune: the PlantVillage-pretrained
checkpoint is fine-tuned on the 1,614 PlantDoc training images (20 mapped
classes), 15% of train is held out for early stopping, and the official
169-image test split is used once for evaluation.

```bash
./scripts/run_finetune_matrix.sh
```

This runs the three core models for seeds 42 and 43 and writes:

- `results/metrics/finetune_<model>__s<seed>.json`
- `results/predictions/finetune_<model>__s<seed>.csv`
- `results/checkpoints/finetune_<model>__s<seed>_best.pth`
- `results/logs/finetune_<model>__s<seed>.log`

A single run can also be launched directly:

```bash
.venv/bin/python scripts/audit_finetune.py \
    --model mobilenetv3_large \
    --checkpoint results/checkpoints/MobileNetV3_Large_SE_r4__mobilenetv3_large__imagenet__s42_epoch_23.pth \
    --seed 42 --epochs 25 --lr 1e-4
```

## 4. What each script does

| Script | Purpose |
|---|---|
| `scripts/train.py` | Train scratch/pretrained/no-background models on PlantVillage, evaluate zero-shot on PlantDoc |
| `scripts/audit_border.py` | E3 border-only probe |
| `scripts/audit_crop.py` | E4 paired full-image vs leaf-crop evaluation |
| `scripts/audit_finetune.py` | Fine-tune on PlantDoc train and evaluate on the official test split |
| `scripts/aggregate_seeds.py` | Mean ± SD across seeds; paired McNemar/bootstrap/TOST comparisons |
| `scripts/run_path_b.sh` | Audit matrix runner |
| `scripts/run_finetune_matrix.sh` | Fine-tune matrix runner |
| `figures/gen_fig_paa.py` | PAA test-split and adaptation figures |
| `figures/gen_fig_paa_protocol.py` | Shortcut Reliance Audit protocol overview (Figure 1) |
| `src/audit.py` | Metrics, border mask, VOC parsing, cropping, statistics |
| `src/models/factory.py` | Model factory, ImageNet backbone loading, BN settings |
| `tests/` | CPU smoke tests and optional torch integration tests |

## 5. Sanity checks

```bash
.venv/bin/python -m unittest discover -s tests
RUN_PRETRAINED_TESTS=1 .venv/bin/python -m unittest tests.test_torch_integration -v
```

The first command runs the CPU tests; the second downloads ImageNet weights and
checks that the custom MobileNetV3 backbones load every feature tensor.

## 6. Notes for reproduction

- **Two seeds, not three.** The manuscript uses seeds 42 and 43. Adding seed 44
  changes the means and will not match the paper.
- **BatchNorm settings.** Pretrained checkpoints were trained with
  `--bn-eps 0.001 --bn-momentum 0.01`. The crop and fine-tune scripts set these
  explicitly; using the defaults changes the features and the reported numbers.
- **Evaluation split.** Zero-shot field accuracy in the audit is reported on
  all 1,783 annotated PlantDoc images; the official 169-image test split is
  reported as a sensitivity check. The fine-tune experiment uses the 169-image
  test split. State which split a number refers to.
- **Stale summary.** `results/metrics_summary.csv` is regenerated by
  `MODE=aggregate SEEDS="42 43"`. Do not quote a summary with `n_runs=1`.
- **Code comments** are in English; the README is the entry point for a fresh
  clone.

## 7. License and citation

The code is released for research use. If you use it, cite the manuscript above
and the PlantVillage and PlantDoc datasets. A `CITATION.cff` will be added at
acceptance.
