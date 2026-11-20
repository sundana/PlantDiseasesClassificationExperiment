"""Fine-tune a PlantVillage checkpoint on the PlantDoc training split.

This is the positive-result experiment for the PAA revision: the audit shows
that background removal and attention changes recover only a few points, so
the remaining question is whether labelled field data closes the gap.  The
script uses the official PlantDoc train/test split (1,614 / 169 images in the
20 mapped classes), holds out a stratified validation fraction from *train*,
and evaluates once on the official test split.

The zero-shot audit checkpoints have 28 or 29 outputs (PlantVillage classes).
The classifier rows for the 20 mapped classes are reused to initialise the new
20-class head; the rest of the network is loaded from the checkpoint.

Example
-------
.venv/bin/python scripts/audit_finetune.py \
    --model mobilenetv3_large \
    --checkpoint results/checkpoints/MobileNetV3_Large_SE_r4__mobilenetv3_large__imagenet__s42_epoch_23.pth \
    --seed 42 --epochs 25 --lr 1e-4

Outputs
-------
results/metrics/finetune_<model>__s<seed>.json
results/predictions/finetune_<model>__s<seed>.csv
results/checkpoints/finetune_<model>__s<seed>_best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.audit import classification_metrics, remap_state_dict_keys, write_predictions
from src.config import CHECKPOINT_DIR, PLANTDOC_DIR, RESULTS_DIR, set_seed
from src.constants import EVAL_CLASSES, PLANTDOC_PV_MAPPING, PV_CLASSES
from src.engine import train_step, valid_step
from src.models.factory import ALL_MODELS, build_model, set_bn_settings

METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


class MappedPlantDoc(Dataset):
    """PlantDoc images with labels mapped to the 20 shared PlantVillage classes."""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def collect_split(root: str, max_images: int | None = None) -> List[Tuple[str, int]]:
    """Collect mapped images from a PlantDoc split root.

    ``root`` is either ``.../train`` or ``.../test``; each immediate child is a
    PlantDoc class directory.  Only the 20 classes present in
    ``PLANTDOC_PV_MAPPING`` are retained.
    """
    samples: List[Tuple[str, int]] = []
    for plantdoc_class, pv_class in sorted(PLANTDOC_PV_MAPPING.items()):
        class_dir = os.path.join(root, plantdoc_class)
        if not os.path.isdir(class_dir):
            continue
        label = EVAL_CLASSES.index(pv_class)
        for name in sorted(os.listdir(class_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            samples.append((os.path.join(class_dir, name), label))
            if max_images is not None and len(samples) >= max_images:
                return samples
    return samples


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def load_backbone(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """Load a PlantVillage checkpoint into a 20-class model.

    The classifier rows for the 20 mapped classes are copied into the new head;
    all other tensors are loaded after the audit key remapping.  Returns a small
    report for the metrics JSON.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = remap_state_dict_keys(state)

    old_weight = state.get("linear4.weight")
    old_bias = state.get("linear4.bias")
    if old_weight is None or old_bias is None:
        raise RuntimeError("checkpoint has no linear4.weight / linear4.bias classifier")

    # The checkpoint's classifier follows PV_CLASSES; EVAL_CLASSES is sorted.
    pv_index = {name: i for i, name in enumerate(PV_CLASSES)}
    rows = [pv_index[name] for name in EVAL_CLASSES]
    with torch.no_grad():
        model.linear4.weight.copy_(old_weight[rows].to(model.linear4.weight.device))
        model.linear4.bias.copy_(old_bias[rows].to(model.linear4.bias.device))

    # Remove the classifier; everything else must load.
    state.pop("linear4.weight", None)
    state.pop("linear4.bias", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("linear4.")]
    if missing:
        raise RuntimeError(f"backbone tensors missing from checkpoint: {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected checkpoint keys: {unexpected[:5]}")
    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "mapped_classifier_rows": len(rows),
        "missing_backbone_tensors": missing,
        "unexpected_keys": unexpected,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[str], List[float], List[str], List[str]]:
    model.eval()
    paths: List[str] = []
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    pd_classes: List[str] = []

    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            for path, label, pidx, c in zip(batch_paths, labels.tolist(), pred_idx.tolist(), conf.tolist()):
                paths.append(path)
                y_true.append(EVAL_CLASSES[label])
                y_pred.append(EVAL_CLASSES[pidx])
                confidences.append(float(c))
                pd_classes.append(os.path.basename(os.path.dirname(path)))
    return paths, y_true, y_pred, confidences, pd_classes


class PathDataset(Dataset):
    """Dataset that also returns the image path, for prediction files."""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune on PlantDoc and evaluate on its official test split")
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--max-train", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--max-test", type=int, default=None, help="smoke-test cap")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    run_tag = args.run_tag or f"finetune_{args.model}__s{args.seed}"

    train_root = os.path.join(PLANTDOC_DIR, "train")
    test_root = os.path.join(PLANTDOC_DIR, "test")
    train_all = collect_split(train_root, args.max_train)
    test_samples = collect_split(test_root, args.max_test)
    if not train_all or not test_samples:
        raise RuntimeError(
            f"no mapped PlantDoc images under {train_root} or {test_root}; "
            "check the data layout and the symlinks"
        )

    labels = [label for _, label in train_all]
    train_samples, val_samples = train_test_split(
        train_all,
        test_size=args.val_fraction,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"PlantDoc mapped: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Classes ({len(EVAL_CLASSES)}): {EVAL_CLASSES}")

    train_transform, test_transform = build_transforms()
    train_loader = DataLoader(
        MappedPlantDoc(train_samples, train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        MappedPlantDoc(val_samples, test_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        PathDataset(test_samples, test_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    model, model_info = build_model(args.model, num_classes=len(EVAL_CLASSES), pretrained=False)
    # Match the BatchNorm settings used when the pretrained checkpoints were
    # trained (torchvision defaults), otherwise the running statistics and the
    # normalisation epsilon are inconsistent with the checkpoint.
    set_bn_settings(model, eps=0.001, momentum=0.01)
    load_report = load_backbone(model, args.checkpoint, device)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = valid_step(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(100 * train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(100 * val_acc),
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs}  train {100*train_acc:5.2f}%  "
            f"val {100*val_acc:5.2f}%  lr {scheduler.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val:
            best_val = float(val_acc)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch} (best {best_epoch}, val {100*best_val:.2f}%)")
            break

    model.load_state_dict(best_state)
    model.eval()

    paths, y_true, y_pred, confidences, pd_classes = evaluate(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred)
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]

    checkpoint_out = os.path.join(CHECKPOINT_DIR, f"{run_tag}_best.pth")
    torch.save(best_state, checkpoint_out)

    metrics_path = os.path.join(METRICS_DIR, f"{run_tag}.json")
    payload = {
        "run_tag": run_tag,
        "model": args.model,
        "model_info": model_info,
        "seed": args.seed,
        "pretrained_checkpoint": load_report,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": float(100 * best_val),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "field": metrics,
        "history": history,
        "checkpoint": checkpoint_out,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    prediction_path = os.path.join(PRED_DIR, f"{run_tag}.csv")
    write_predictions(
        prediction_path,
        [
            {
                "image": os.path.relpath(path, PLANTDOC_DIR),
                "pd_class": pd_class,
                "true": true,
                "pred": pred,
                "confidence": f"{conf:.6f}",
                "correct": correct,
            }
            for path, pd_class, true, pred, conf, correct in zip(
                paths, pd_classes, y_true, y_pred, confidences, correct
            )
        ],
        ["image", "pd_class", "true", "pred", "confidence", "correct"],
    )

    print(f"\nTest accuracy: {metrics['accuracy']:.2f}%  macro F1: {metrics['f1']:.2f}%")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {prediction_path}")
    print(f"Checkpoint: {checkpoint_out}")


if __name__ == "__main__":
    main()
