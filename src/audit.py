"""Shared utilities for the Path B audit experiments.

Everything in this module is deliberately independent of the training loop so
that the probes (border-only classifier, leaf-crop intervention) can reuse the
same label space, metrics and checkpoint handling as the main experiments.

The metric convention matches ``compute_metrics.py`` at the repository root:
per-class precision/recall/F1 from the confusion matrix, macro-averaged over
the 20 matched classes, with predictions on unmapped PlantVillage classes
counted as false negatives only.
"""

from __future__ import annotations

import csv
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.constants import (
    BACKGROUND_CLASS,
    EVAL_CLASSES,
    PLANTDOC_PV_MAPPING,
    PV_CLASSES,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    eval_classes: Sequence[str] = EVAL_CLASSES,
) -> Dict:
    """Accuracy plus macro-averaged precision, recall and F1.

    Mirrors ``compute_metrics.macro_metrics`` so that the audit tables are
    directly comparable with the numbers already reported in the manuscript.

    A prediction on a class outside ``eval_classes`` is a false negative for
    the true class but not a false positive for any evaluated class. This is
    the convention used for the zero-shot PlantDoc evaluation, where the model
    can predict one of the nine unmapped PlantVillage classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    eval_classes = list(eval_classes)
    index = {c: i for i, c in enumerate(eval_classes)}
    k = len(eval_classes)
    tp = np.zeros(k, dtype=np.int64)
    fp = np.zeros(k, dtype=np.int64)
    fn = np.zeros(k, dtype=np.int64)
    n_correct = n_unmapped = n_background = 0

    for truth, pred in zip(y_true, y_pred):
        if truth not in index:
            raise ValueError(
                f"true label {truth!r} is not one of the {k} evaluated classes; "
                "filter the samples before calling classification_metrics()"
            )
        if pred == truth:
            tp[index[truth]] += 1
            n_correct += 1
        else:
            fn[index[truth]] += 1
            if pred in index:
                fp[index[pred]] += 1
        if pred not in index:
            n_unmapped += 1
        if pred == BACKGROUND_CLASS:
            n_background += 1

    precision = np.divide(
        tp, tp + fp, out=np.zeros(k, dtype=float), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros(k, dtype=float), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(k, dtype=float),
        where=(precision + recall) > 0,
    )
    n = len(y_true)
    per_class = {
        c: {
            "precision": float(100 * precision[i]),
            "recall": float(100 * recall[i]),
            "f1": float(100 * f1[i]),
            "support": int(tp[i] + fn[i]),
        }
        for i, c in enumerate(eval_classes)
    }
    return {
        "n": int(n),
        "accuracy": float(100 * n_correct / n),
        "precision": float(100 * precision.mean()),
        "recall": float(100 * recall.mean()),
        "f1": float(100 * f1.mean()),
        "background_rate": float(100 * n_background / n),
        "unmapped_rate": float(100 * n_unmapped / n),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Image-space probes
# ---------------------------------------------------------------------------


class BorderMask:
    """Keep the outer frame of an image and fill the centre.

    Used by the border-only probe (E3): if a model trained on this view still
    classifies PlantVillage above chance, the background carries label
    information (capture bias). The fill colour is a fixed neutral grey rather
    than the PlantVillage background colour, so the probe does not simply learn
    the dataset's background tone.
    """

    def __init__(self, fraction: float = 0.2, fill: Tuple[int, int, int] = (124, 116, 104)):
        if not 0.0 <= fraction < 0.5:
            raise ValueError("fraction must be in [0, 0.5)")
        self.fraction = float(fraction)
        self.fill = tuple(int(v) for v in fill)

    def __call__(self, image):
        if self.fraction == 0.0:
            return image
        from PIL import ImageDraw

        width, height = image.size
        dx = int(round(self.fraction * width))
        dy = int(round(self.fraction * height))
        if width - 2 * dx < 4 or height - 2 * dy < 4:
            return image
        out = image.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([dx, dy, width - dx, height - dy], fill=self.fill)
        return out

    def __repr__(self):  # helps when the transform is written to metrics.json
        return f"BorderMask(fraction={self.fraction}, fill={self.fill})"


def crop_to_boxes(image, boxes: Iterable[Tuple[float, float, float, float]], margin: float = 0.05):
    """Crop an image to the union of bounding boxes, with a relative margin.

    Returns ``(cropped_image, used_box)``. If no valid box is available the
    original image is returned with ``used_box=None``; callers must record
    whether the intervention actually applied.
    """
    width, height = image.size
    valid = []
    for xmin, ymin, xmax, ymax in boxes:
        xmin = max(0.0, min(float(xmin), width - 1))
        ymin = max(0.0, min(float(ymin), height - 1))
        xmax = max(0.0, min(float(xmax), width))
        ymax = max(0.0, min(float(ymax), height))
        if xmax - xmin >= 2 and ymax - ymin >= 2:
            valid.append((xmin, ymin, xmax, ymax))
    if not valid:
        return image, None

    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    pad_x = margin * (x1 - x0)
    pad_y = margin * (y1 - y0)
    x0 = max(0, int(math.floor(x0 - pad_x)))
    y0 = max(0, int(math.floor(y0 - pad_y)))
    x1 = min(width, int(math.ceil(x1 + pad_x)))
    y1 = min(height, int(math.ceil(y1 + pad_y)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return image, None
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# PlantDoc annotations (Pascal VOC)
# ---------------------------------------------------------------------------


def _split_from_annotation_root(root: str, annotation_root: str) -> Optional[str]:
    """Infer ``train``/``test`` from the first component under the annotation root.

    Handles both ``annotations/TRAIN/*.xml`` and
    ``annotations/TRAIN/<class>/*.xml``. Returns ``None`` for a flat directory.
    """
    rel = os.path.relpath(root, annotation_root)
    if rel == ".":
        base = os.path.basename(os.path.abspath(annotation_root)).lower()
        return base if base in ("train", "test") else None
    first = rel.split(os.sep)[0].lower()
    return first if first in ("train", "test") else None


def index_voc_annotations(annotation_dirs: Sequence[str]) -> Dict[str, str]:
    """Map annotation stems to paths, with optional split-aware keys.

    Two keys are stored per annotation when a split can be inferred:
    ``"<stem>"`` (backward compatible) and ``"<split>/<stem>"``. The split key
    prevents a ``train`` image from being paired with a ``test`` annotation of
    the same filename, which is common in the PlantDoc object-detection release.
    """
    index: Dict[str, str] = {}
    for directory in annotation_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            split = _split_from_annotation_root(root, directory)
            for name in files:
                if not name.lower().endswith(".xml"):
                    continue
                stem = os.path.splitext(name)[0]
                path = os.path.join(root, name)
                index.setdefault(stem, path)
                if split:
                    index.setdefault(f"{split}/{stem}", path)
    return index


def parse_voc_boxes(path: str) -> List[Tuple[float, float, float, float]]:
    """Parse ``<object><bndbox>`` entries from a Pascal VOC XML file."""
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall(".//object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


_COLL_SUFFIX = re.compile(r"_coll\d*$")


def lookup_annotation(stem: str, index: Dict[str, str], split: Optional[str] = None) -> Optional[str]:
    """Find an annotation for an image stem, preferring the matching split.

    Tolerates ``_coll`` suffixes added by ``merge_plantdoc.py``. If a split is
    known, ``"<split>/<stem>"`` is tried before the flat stem, so that identical
    filenames in the train and test splits resolve correctly.
    """
    stripped = _COLL_SUFFIX.sub("", stem)
    candidates = []
    if split:
        candidates.append(f"{split}/{stem}")
    candidates.append(stem)
    if split:
        candidates.append(f"{split}/{stripped}")
    candidates.append(stripped)
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    return None


def collect_plantdoc_samples(
    root: str,
    mapping: Dict[str, str] = PLANTDOC_PV_MAPPING,
    annotation_index: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Enumerate the mapped PlantDoc images and attach their annotation path.

    Supports both a flat merged layout (``root/<class>/``) and the original
    split layout (``root/train/<class>/`` and ``root/test/<class>/``). The split
    is recorded so that split-aware annotation lookup can be used.
    """
    index = annotation_index or {}
    split_roots: List[Tuple[Optional[str], str]] = []
    for split in ("train", "test"):
        base = os.path.join(root, split)
        if os.path.isdir(base):
            split_roots.append((split, base))
    if not split_roots:
        split_roots = [(None, root)]

    samples: List[Dict] = []
    missing_dirs = []
    for pd_class, pv_class in mapping.items():
        pv_index = PV_CLASSES.index(pv_class)
        found_class = False
        for split, base in split_roots:
            class_dir = os.path.join(base, pd_class)
            if not os.path.isdir(class_dir):
                continue
            found_class = True
            for name in sorted(os.listdir(class_dir)):
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = os.path.splitext(name)[0]
                samples.append(
                    {
                        "path": os.path.join(class_dir, name),
                        "file": name,
                        "stem": stem,
                        "split": split,
                        "pd_class": pd_class,
                        "pv_class": pv_class,
                        "pv_index": pv_index,
                        "annotation": lookup_annotation(stem, index, split),
                    }
                )
        if not found_class:
            missing_dirs.append(os.path.join(root, pd_class))
    if verbose:
        n_classes = len({s["pd_class"] for s in samples})
        print(f"PlantDoc: {len(samples)} mapped images from {n_classes} classes")
        if missing_dirs:
            print(f"  [warn] {len(missing_dirs)} class directories not found, e.g. {missing_dirs[0]}")
    return samples


class PlantDocImageDataset:
    """Full-image PlantDoc dataset for zero-shot evaluation.

    Returns ``(tensor, pv_index, sample_index)``; the sample index lets the
    caller recover paths and labels without relying on a collate function.
    """

    def __init__(self, samples: Sequence[Dict], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["pv_index"]), index


class PlantDocCropDataset:
    """Paired full-image / leaf-crop dataset for the E4 intervention.

    Returns ``(full, crop, pv_index, sample_index, has_annotation, n_boxes)``.
    ``crop`` is the full image when no annotation exists, so the caller must
    use ``has_annotation`` to restrict the paired test to intervened images.
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        transform=None,
        margin: float = 0.05,
        cache_boxes: bool = True,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.margin = float(margin)
        self.cache_boxes = cache_boxes
        self._boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    def _boxes_for(self, index: int):
        if self.cache_boxes and index in self._boxes:
            return self._boxes[index]
        sample = self.samples[index]
        boxes = []
        if sample.get("annotation") and os.path.exists(sample["annotation"]):
            boxes = parse_voc_boxes(sample["annotation"])
        if self.cache_boxes:
            self._boxes[index] = boxes
        return boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        boxes = self._boxes_for(index)
        crop, used = crop_to_boxes(image, boxes, margin=self.margin)
        full_tensor = self.transform(image) if self.transform is not None else image
        crop_tensor = self.transform(crop) if self.transform is not None else crop
        return (
            full_tensor,
            crop_tensor,
            int(sample["pv_index"]),
            index,
            int(used is not None),
            len(boxes),
        )


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------


def remap_state_dict_keys(state_dict: Dict) -> Dict:
    """Apply the key renames used by older checkpoints in this repository."""
    remapped = {}
    for key, value in state_dict.items():
        if ".se.se." in key:
            key = key.replace(".se.se.", ".attention_module.se.")
        if ".module.channel_attention.fc1." in key:
            key = key.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
        if ".module.channel_attention.fc2." in key:
            key = key.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
        if ".module.spatial_attention.conv." in key:
            key = key.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
        if "bneck" in key and ".module." in key:
            key = key.replace(".module.", ".attention_module.")
        remapped[key] = value
    return remapped


def load_checkpoint(model, path: str, device="cpu", strict: bool = True) -> Dict:
    """Load a checkpoint saved by ``engine.EarlyStopping``.

    Returns a small info dict; raises with a readable message when the load is
    incomplete, because a partially loaded model silently invalidates metrics.
    """
    import torch

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location=device)
    state_dict = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    state_dict = remap_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Incomplete weight load from {path}: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. First missing: {missing[:3]}; "
            f"first unexpected: {unexpected[:3]}"
        )
    return {
        "path": path,
        "epoch": obj.get("epoch") if isinstance(obj, dict) else None,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar_exact(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> Dict:
    """Exact two-sided McNemar test on paired correctness vectors."""
    a = np.asarray(list(y_true)) == np.asarray(list(pred_a))
    b = np.asarray(list(y_true)) == np.asarray(list(pred_b))
    n01 = int(np.sum(~a & b))  # A wrong, B correct
    n10 = int(np.sum(a & ~b))  # A correct, B wrong
    n = n01 + n10
    if n == 0:
        return {"p_value": 1.0, "n01": 0, "n10": 0}
    k = min(n01, n10)
    try:  # scipy is available in the project environment
        from scipy.stats import binomtest

        p_value = float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:  # pure-Python fallback for tests
        log_terms = [
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            for i in range(k + 1)
        ]
        peak = max(log_terms)
        log_tail = peak + math.log(math.fsum(math.exp(t - peak) for t in log_terms))
        log_p = math.log(2.0) + log_tail - n * math.log(2.0)
        p_value = min(1.0, math.exp(log_p))
    return {"p_value": p_value, "n01": n01, "n10": n10}


def paired_bootstrap_difference(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict:
    """Paired bootstrap for ``accuracy(A) - accuracy(B)`` in percentage points."""
    correct_a = (np.asarray(list(y_true)) == np.asarray(list(pred_a))).astype(np.int8)
    correct_b = (np.asarray(list(y_true)) == np.asarray(list(pred_b))).astype(np.int8)
    diff = correct_a - correct_b
    n = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    alpha = 1.0 - confidence
    return {
        "mean_diff_pp": float(100 * diff.mean()),
        "ci_low_pp": float(100 * np.percentile(means, 100 * alpha / 2)),
        "ci_high_pp": float(100 * np.percentile(means, 100 * (1 - alpha / 2))),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def tost_equivalence(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    margin_pp: float = 1.0,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Bootstrap two-one-sided-test (TOST) equivalence decision.

    Equivalence is declared when the 90% CI of the paired accuracy difference
    lies entirely inside [-margin, +margin]. This is the bootstrap analogue of
    two one-sided tests at alpha = 0.05.
    """
    ci90 = paired_bootstrap_difference(
        y_true, pred_a, pred_b, n_bootstrap=n_bootstrap, seed=seed, confidence=0.90
    )
    ci95 = paired_bootstrap_difference(
        y_true, pred_a, pred_b, n_bootstrap=n_bootstrap, seed=seed, confidence=0.95
    )
    equivalent = (ci90["ci_low_pp"] > -margin_pp) and (ci90["ci_high_pp"] < margin_pp)
    return {
        "margin_pp": margin_pp,
        "equivalent": bool(equivalent),
        "diff_pp": ci95["mean_diff_pp"],
        "ci90_low_pp": ci90["ci_low_pp"],
        "ci90_high_pp": ci90["ci_high_pp"],
        "ci95_low_pp": ci95["ci_low_pp"],
        "ci95_high_pp": ci95["ci_high_pp"],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_predictions(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_predictions(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
