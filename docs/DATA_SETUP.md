# Data and environment setup

Reproduces the environment and datasets used for the Path B audit on this
machine (RTX 5060 Ti, 16 GB, driver 595.99, CUDA 13.2).

## Environment (uv)

```bash
cd PlantDiseasesClassificationExperiment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
    numpy scikit-learn scipy matplotlib seaborn pandas tqdm psutil \
    ptflops tensorboard openpyxl python-dotenv requests datasets
```

`torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` support the RTX 50-series
`sm_120` compute capability. Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## PlantVillage

Source: HuggingFace `Project-AgML/plant_village_classification`
(39 classes, 55,448 images; includes `Background_without_leaves`, 1,143 images).

```bash
# download the two parquet shards (~862 MB)
mkdir -p data/raw/plantvillage_hf
curl -L -o data/raw/plantvillage_hf/train-00000.parquet \
  https://huggingface.co/datasets/Project-AgML/plant_village_classification/resolve/main/data/train-00000-of-00002.parquet
curl -L -o data/raw/plantvillage_hf/train-00001.parquet \
  https://huggingface.co/datasets/Project-AgML/plant_village_classification/resolve/main/data/train-00001-of-00002.parquet

# extract the 29 classes used in the manuscript
.venv/bin/python scripts/prepare_plantvillage.py \
  --parquet-glob "data/raw/plantvillage_hf/*.parquet" \
  --out data/plantvillage
```

Result: 41,879 images across 29 classes. The manuscript's 46,564-image table was
produced by augmenting classes below 1,000 images to 1,000; to reproduce that
exactly, add `--min-per-class 1000`. The audit uses the unaugmented 41,879-image
version, which avoids synthetic duplicates.

## PlantDoc

Source: `pratikkayal/PlantDoc-Dataset` (images, split layout) and
`pratikkayal/PlantDoc-Object-Detection-Dataset` (Pascal VOC annotations).

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git \
  data/raw/plantdoc_classification
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git \
  data/raw/plantdoc_annotations
cd data/raw/plantdoc_annotations
git sparse-checkout set --no-cone 'TRAIN/*.xml' 'TEST/*.xml'
cd -
```

Link into the layout expected by `src/config.py` (`data/` is git-ignored):

```bash
mkdir -p data/plantdoc
ln -sfn ../raw/plantdoc_classification/train data/plantdoc/train
ln -sfn ../raw/plantdoc_classification/test  data/plantdoc/test
ln -sfn raw/plantdoc_annotations              data/plantdoc_annotations
```

Verified on this machine:

- 1,783 images in the 20 mapped classes (matches the manuscript).
- 1,614 train / 169 test.
- 1,783/1,783 images have an annotation.
- Annotation filenames are unique within a split; the split-aware lookup handles
  the few cross-split collisions and the `foo.JPG.jpg` → `foo.JPG.xml` pattern.

## Sanity checks

```bash
.venv/bin/python -m unittest discover -s tests -v
RUN_PRETRAINED_TESTS=1 .venv/bin/python -m unittest tests.test_torch_integration -v
```

Expected: all tests pass; the pretrained test downloads ImageNet weights and
checks that the custom MobileNetV3 backbones load every feature tensor.

## Architecture note

The custom Large backbone originally had `672` expansion in the penultimate
block; MobileNetV3 specifies `960`. The block was corrected to `960` so that the
ImageNet backbone loads completely. This changes the Large parameter count and
the complexity table; regenerate `Table 2` with `scripts/measure_complexity.py`.
Small is unchanged.
