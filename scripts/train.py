"""
Trains a PyTorch image classification model using device-agnostic code.

Path B additions (all backward compatible):
  --pretrained {none,imagenet}   ImageNet-initialised backbone
  --lr FLOAT                     override the learning rate
  --exclude-class NAME           drop a class before splitting (repeatable)
  --run-tag TAG                  unique suffix for checkpoints and metrics
  --eval-plantdoc                run the zero-shot PlantDoc evaluation after training
  --metrics-dir DIR              where metrics.json is written
"""

import matplotlib
matplotlib.use('Agg')  # Prevents any matplotlib windows from opening

import sys
import os
import json
import subprocess
from datetime import datetime, timezone

# Fix Unicode emoji printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src import data_setup, engine
from src.audit import (
    PlantDocImageDataset,
    classification_metrics,
    collect_plantdoc_samples,
    write_predictions,
)
from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_SEED,
    PLANTDOC_DIR,
    PLANTVILLAGE_DIR,
    RESULTS_DIR,
    set_seed,
)
from src.constants import EVAL_CLASSES
from src.models.factory import ALL_MODELS, build_model
from src.utils import _get_model_display_name
from torchvision import transforms


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _find_best_checkpoint(display_name: str) -> str | None:
    """Return the checkpoint left behind by EarlyStopping for this run."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    prefix = f"{display_name}_epoch_"
    candidates = [
        os.path.join(CHECKPOINT_DIR, name)
        for name in os.listdir(CHECKPOINT_DIR)
        if name.startswith(prefix) and name.endswith(".pth")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _predict_loader(model, loader, device):
    """Return (y_true_idx, y_pred_idx, confidence) for a standard loader."""
    model.eval()
    y_true, y_pred, confidence = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence.extend(probs.max(dim=1).values.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(confidence)


def _evaluate_plantdoc(model, class_names, plantdoc_dir, batch_size, device):
    """Zero-shot PlantDoc evaluation on the 20 mapped classes."""
    samples = collect_plantdoc_samples(plantdoc_dir)
    if not samples:
        print(f"[warn] no PlantDoc images found under {plantdoc_dir}; skipping field evaluation")
        return None, None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PlantDocImageDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    rows = []
    y_true, y_pred, confidence = [], [], []
    with torch.inference_mode():
        for images, labels, indices in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for label, pred_idx, conf_value, sample_idx in zip(
                labels.tolist(), pred.tolist(), conf.tolist(), indices.tolist()
            ):
                sample = samples[sample_idx]
                pred_name = class_names[pred_idx]
                y_true.append(sample["pv_class"])
                y_pred.append(pred_name)
                confidence.append(conf_value)
                rows.append({
                    "image": os.path.relpath(sample["path"], plantdoc_dir),
                    "pd_class": sample["pd_class"],
                    "true": sample["pv_class"],
                    "pred": pred_name,
                    "confidence": f"{conf_value:.6f}",
                    "correct": int(pred_name == sample["pv_class"]),
                })
    metrics = classification_metrics(y_true, y_pred, EVAL_CLASSES)
    metrics["n_images"] = len(samples)
    return metrics, rows


def start_training(
    model_name: str,
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float | None = None,
    data_dir: str = PLANTVILLAGE_DIR,
    device_name: str = "auto",
    no_plot: bool = False,
    seed: int = DEFAULT_SEED,
    pretrained: bool = False,
    exclude_classes: list[str] | None = None,
    run_tag: str = "",
    eval_plantdoc: bool = False,
    plantdoc_dir: str = PLANTDOC_DIR,
    metrics_dir: str | None = None,
    patience: int = 5,
):
    set_seed(seed)
    # Setup target device
    if device_name == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_name

    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, test_dataloader, class_names = (
        data_setup.create_dataloader(
            data_path=data_dir,
            train_transform=train_transform,
            test_transform=test_transform,
            batch_size=batch_size,
            seed=seed,
            stratified=True,
            exclude_classes=exclude_classes or None,
        )
    )

    # Model selection. Attention modules and the classifier head are randomly
    # initialised even when the backbone is pretrained (see factory.py).
    model, model_info = build_model(
        model_name, num_classes=len(class_names), pretrained=pretrained
    )
    if run_tag:
        model.run_tag = run_tag
    model = model.to(device)

    lr = learning_rate if learning_rate is not None else (1e-4 if pretrained else 1e-3)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Show training config
    print(f"Model: {model.__class__.__name__}")
    print(f"Initialisation: {'ImageNet backbone' if pretrained else 'from scratch'}")
    print(f"Epoch: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Seed: {seed}")
    print(f"Excluded classes: {exclude_classes or []}")
    print(f"Run tag: {run_tag or '-'}")
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Optimizer: {optimizer}")
    print(f"Device: {device}")
    print(f"Data Directory: {data_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Start training with help from engine.py
    history = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=num_epochs,
        device=torch.device(device),
        patience=patience,
        early_stopping=True,
        min_delta=0,
        checkpoint_dir=CHECKPOINT_DIR,
        enable_live_plot=not no_plot,
    )

    # Restore the best (lowest validation loss) checkpoint before evaluation.
    display_name = _get_model_display_name(model)
    checkpoint_path = _find_best_checkpoint(display_name)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"Loaded best checkpoint: {checkpoint_path}")
    else:
        print("[warn] no checkpoint found; evaluating the final epoch weights")

    # In-domain test metrics on the full label space.
    y_true_idx, y_pred_idx, _ = _predict_loader(model, test_dataloader, device)
    y_true_names = [class_names[i] for i in y_true_idx.tolist()]
    y_pred_names = [class_names[i] for i in y_pred_idx.tolist()]
    lab_metrics = classification_metrics(y_true_names, y_pred_names, class_names)

    # The 20-class subset used in the paper's Table 4.
    mapped = set(EVAL_CLASSES)
    keep = [i for i, name in enumerate(y_true_names) if name in mapped]
    lab20_metrics = (
        classification_metrics(
            [y_true_names[i] for i in keep],
            [y_pred_names[i] for i in keep],
            EVAL_CLASSES,
        )
        if keep
        else None
    )

    # Optional zero-shot field evaluation.
    field_metrics, field_rows = (None, None)
    if eval_plantdoc:
        field_metrics, field_rows = _evaluate_plantdoc(
            model, class_names, plantdoc_dir, batch_size, device
        )

    # Write metrics.json
    metrics_dir = metrics_dir or os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    tag = run_tag or f"{model_name}__seed{seed}__{'pretrained' if pretrained else 'scratch'}"
    metrics_path = os.path.join(metrics_dir, f"{tag}.json")
    payload = {
        "run_tag": tag,
        "model": model_name,
        "model_info": model_info,
        "seed": seed,
        "pretrained": bool(pretrained),
        "num_classes": len(class_names),
        "class_names": class_names,
        "excluded_classes": exclude_classes or [],
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs_requested": num_epochs,
        "epochs_run": len(history.get("train_loss", [])),
        "best_val_acc": float(max(history.get("valid_acc", [0.0]))),
        "lab": lab_metrics,
        "lab_20class": lab20_metrics,
        "field": field_metrics,
        "checkpoint": checkpoint_path,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if field_rows is not None:
        pred_path = os.path.join(
            RESULTS_DIR, "predictions", f"{tag}_plantdoc.csv"
        )
        write_predictions(
            pred_path,
            field_rows,
            fieldnames=["image", "pd_class", "true", "pred", "confidence", "correct"],
        )
        print(f"PlantDoc predictions written to {pred_path}")

    print(f"Metrics written to {metrics_path}")
    lab20_text = (
        f"{lab20_metrics['accuracy']:.2f}%" if lab20_metrics else "n/a"
    )
    print(
        f"Lab acc {lab_metrics['accuracy']:.2f}% | "
        f"Lab-20 acc {lab20_text} | "
        + (
            f"Field acc {field_metrics['accuracy']:.2f}% | "
            f"Field macro F1 {field_metrics['f1']:.2f}%"
            if field_metrics
            else "Field: not evaluated"
        )
    )
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for train model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", default="mobilenetv3_large", choices=ALL_MODELS,
        help="name of the model",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="number of epochs to train for"
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, default=PLANTVILLAGE_DIR, help="path to the dataset"
    )
    parser.add_argument(
        "--device", default="auto", help="device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="random seed"
    )
    parser.add_argument(
        "--plot", action="store_true", help="enable real-time plotting"
    )
    parser.add_argument(
        "--pretrained", choices=["none", "imagenet"], default="none",
        help="initialise the backbone from ImageNet weights",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="learning rate (default: 1e-3 scratch, 1e-4 pretrained)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "--exclude-class", action="append", default=[],
        help="class directory to exclude before splitting (repeatable)",
    )
    parser.add_argument(
        "--run-tag", default="", help="unique suffix for checkpoint and metrics names"
    )
    parser.add_argument(
        "--eval-plantdoc", action="store_true",
        help="run the zero-shot PlantDoc evaluation after training",
    )
    parser.add_argument(
        "--plantdoc-dir", default=PLANTDOC_DIR, help="PlantDoc root directory"
    )
    parser.add_argument(
        "--metrics-dir", default=os.path.join(RESULTS_DIR, "metrics"),
        help="directory for metrics.json",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="early-stopping patience"
    )
    args = parser.parse_args()

    start_training(
        args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        device_name=args.device,
        no_plot=not args.plot,
        seed=args.seed,
        pretrained=(args.pretrained == "imagenet"),
        exclude_classes=args.exclude_class,
        run_tag=args.run_tag,
        eval_plantdoc=args.eval_plantdoc,
        plantdoc_dir=args.plantdoc_dir,
        metrics_dir=args.metrics_dir,
        patience=args.patience,
    )
