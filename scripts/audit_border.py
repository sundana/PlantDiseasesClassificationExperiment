"""E3: border-only probe.

Trains a classifier on the outer frame of PlantVillage images, with the centre
filled by a neutral grey. If this model classifies held-out PlantVillage images
well above chance, the background carries label information (capture bias).
The same model is then applied to PlantDoc, border-masked, to show what the
background statistics alone predict in the field.

Usage:
    python scripts/audit_border.py --model mobilenetv3_small \
        --epochs 30 --seed 42 --run-tag border_s42 --eval-plantdoc
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src import data_setup, engine
from src.audit import (
    BorderMask,
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


def _test_transform(fraction, fill):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        BorderMask(fraction=fraction, fill=fill),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _train_transform(fraction, fill):
    # Resize first so that the border fraction refers to the final 224x224
    # image; random-resized cropping before masking could leave only leaf
    # tissue in the "border".
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=124),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        BorderMask(fraction=fraction, fill=fill),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _find_best_checkpoint(display_name):
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    prefix = f"{display_name}_epoch_"
    candidates = [
        os.path.join(CHECKPOINT_DIR, name)
        for name in os.listdir(CHECKPOINT_DIR)
        if name.startswith(prefix) and name.endswith(".pth")
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _predict(model, loader, device):
    model.eval()
    y_true, y_pred, confidence = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence.extend(probs.max(dim=1).values.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred, confidence


def main():
    parser = argparse.ArgumentParser(description="Border-only probe (E3)")
    parser.add_argument("--model", default="mobilenetv3_small", choices=ALL_MODELS)
    parser.add_argument("--dataset-dir", default=PLANTVILLAGE_DIR)
    parser.add_argument("--plantdoc-dir", default=PLANTDOC_DIR)
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="frame width as a fraction of each image dimension")
    parser.add_argument("--fill", type=int, nargs=3, default=[124, 116, 104],
                        help="RGB fill for the masked centre")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-plantdoc", action="store_true")
    parser.add_argument("--metrics-dir", default=os.path.join(RESULTS_DIR, "metrics"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    tag = args.run_tag or f"border_{args.model}_f{args.fraction}_s{args.seed}"

    train_transform = _train_transform(args.fraction, tuple(args.fill))
    test_transform = _test_transform(args.fraction, tuple(args.fill))

    train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(
        data_path=args.dataset_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        seed=args.seed,
        stratified=True,
    )

    model, model_info = build_model(
        args.model, num_classes=len(class_names), pretrained=args.pretrained
    )
    model.run_tag = tag
    model = model.to(device)

    print(f"Border probe: fraction={args.fraction}, classes={len(class_names)}, device={device}")
    print(f"Chance accuracy: {100.0 / len(class_names):.2f}%")

    history = engine.train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        epochs=args.epochs,
        device=torch.device(device),
        patience=args.patience,
        early_stopping=True,
        min_delta=0,
        checkpoint_dir=CHECKPOINT_DIR,
        enable_live_plot=False,
    )

    display = _get_model_display_name(model)
    checkpoint = _find_best_checkpoint(display)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))

    y_true_idx, y_pred_idx, _ = _predict(model, test_loader, device)
    y_true = [class_names[i] for i in y_true_idx]
    y_pred = [class_names[i] for i in y_pred_idx]
    lab = classification_metrics(y_true, y_pred, class_names)
    lab["chance_accuracy"] = 100.0 / len(class_names)

    field = None
    if args.eval_plantdoc:
        samples = collect_plantdoc_samples(args.plantdoc_dir)
        if not samples:
            print(f"[warn] no PlantDoc images under {args.plantdoc_dir}; skipping field evaluation")
        else:
            dataset = PlantDocImageDataset(samples, transform=test_transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            rows, y_true_field, y_pred_field = [], [], []
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
                        y_true_field.append(sample["pv_class"])
                        y_pred_field.append(pred_name)
                        rows.append({
                            "image": os.path.relpath(sample["path"], args.plantdoc_dir),
                            "pd_class": sample["pd_class"],
                            "true": sample["pv_class"],
                            "pred": pred_name,
                            "confidence": f"{conf_value:.6f}",
                            "correct": int(pred_name == sample["pv_class"]),
                        })
            field = classification_metrics(y_true_field, y_pred_field, EVAL_CLASSES)
            pred_path = os.path.join(RESULTS_DIR, "predictions", f"{tag}_plantdoc.csv")
            write_predictions(
                pred_path, rows,
                fieldnames=["image", "pd_class", "true", "pred", "confidence", "correct"],
            )
            print(f"Wrote {pred_path}")

    os.makedirs(args.metrics_dir, exist_ok=True)
    payload = {
        "run_tag": tag,
        "probe": "border_only",
        "border_fraction": args.fraction,
        "border_fill": args.fill,
        "model": args.model,
        "model_info": model_info,
        "seed": args.seed,
        "pretrained": bool(args.pretrained),
        "num_classes": len(class_names),
        "class_names": class_names,
        "epochs_requested": args.epochs,
        "epochs_run": len(history.get("train_loss", [])),
        "lab": lab,
        "field": field,
        "checkpoint": checkpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(args.metrics_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Border probe lab accuracy: {lab['accuracy']:.2f}% "
          f"(chance {lab['chance_accuracy']:.2f}%)")
    if field:
        print(f"Border probe field accuracy: {field['accuracy']:.2f}%")
    print(f"Metrics written to {path}")


if __name__ == "__main__":
    main()
