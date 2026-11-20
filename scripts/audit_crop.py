"""E4: leaf-crop intervention on PlantDoc.

Evaluates one PlantVillage-trained checkpoint twice on the same PlantDoc
images: (a) the full image and (b) the image cropped to the union of its leaf
bounding boxes. If cropping to the leaf raises accuracy, background content is
causally responsible for part of the laboratory-to-field gap. Predictions on
the full image are also written to CSV so that paired tests (McNemar,
bootstrap) can be reproduced.

PlantDoc annotations are Pascal VOC XML files. They are often distributed
separately from the images; pass one or more roots with --annotations.

Usage:
    python scripts/audit_crop.py \
        --model mobilenetv3_large --weight V3L_SE4__scratch__s42 \
        --annotations data/plantdoc_annotations \
        --out results/predictions/crop_V3L_SE4_s42.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.audit import (
    PlantDocCropDataset,
    classification_metrics,
    collect_plantdoc_samples,
    index_voc_annotations,
    load_checkpoint,
    mcnemar_exact,
    paired_bootstrap_difference,
    write_predictions,
)
from src.config import CHECKPOINT_DIR, DATA_DIR, PLANTDOC_DIR, RESULTS_DIR
from src.constants import BACKGROUND_CLASS, EVAL_CLASSES, PV_CLASSES
from src.models.factory import ALL_MODELS, build_model, set_bn_settings


def _resolve_weight(weight: str) -> str:
    if os.path.exists(weight):
        return weight
    candidate = os.path.join(CHECKPOINT_DIR, f"{weight}.pth")
    if os.path.exists(candidate):
        return candidate
    matches = [
        os.path.join(CHECKPOINT_DIR, name)
        for name in os.listdir(CHECKPOINT_DIR)
        if name.startswith(weight) and name.endswith(".pth")
    ] if os.path.isdir(CHECKPOINT_DIR) else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"checkpoint not found for {weight!r}; tried {candidate}"
        + (f" and {len(matches)} prefix matches" if matches else "")
    )


def _classifier_out_features(model) -> int | None:
    for name in ("linear4", "classifier", "fc"):
        module = getattr(model, name, None)
        if module is None:
            continue
        if hasattr(module, "out_features"):
            return int(module.out_features)
        if isinstance(module, torch.nn.Sequential):
            last = module[-1]
            if hasattr(last, "out_features"):
                return int(last.out_features)
    return None


def _default_annotation_dirs(plantdoc_dir: str):
    candidates = [
        os.path.join(plantdoc_dir, "annotations"),
        os.path.join(os.path.dirname(plantdoc_dir), "annotations"),
        os.path.join(DATA_DIR, "plantdoc", "annotations"),
        os.path.join(DATA_DIR, "PlantDoc", "annotations"),
        os.path.join(DATA_DIR, "plantdoc_annotations"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def main():
    parser = argparse.ArgumentParser(description="Leaf-crop intervention (E4)")
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--weight", required=True,
                        help="checkpoint path, stem, or unique prefix")
    parser.add_argument("--plantdoc-dir", default=PLANTDOC_DIR)
    parser.add_argument("--annotations", nargs="+", default=None,
                        help="one or more directories containing Pascal VOC XML files")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="relative margin added around the union of leaf boxes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None,
                        help="prediction CSV (default: results/predictions/crop_<stem>.csv)")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-class-count-mismatch", action="store_true")
    parser.add_argument("--bn-eps", type=float, default=1e-5,
                        help="BatchNorm epsilon; use 0.001 for ImageNet-pretrained checkpoints")
    parser.add_argument("--bn-momentum", type=float, default=0.1,
                        help="BatchNorm momentum; use 0.01 for ImageNet-pretrained checkpoints")
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    annotation_dirs = args.annotations or _default_annotation_dirs(args.plantdoc_dir)
    annotation_index = index_voc_annotations(annotation_dirs)
    print(f"Annotation directories: {annotation_dirs or '(none found)'}")
    print(f"Indexed {len(annotation_index)} XML files")

    samples = collect_plantdoc_samples(
        args.plantdoc_dir, annotation_index=annotation_index
    )
    if not samples:
        raise SystemExit(f"No PlantDoc images found under {args.plantdoc_dir}")
    n_annotated = sum(1 for s in samples if s["annotation"])
    print(f"Images with annotations: {n_annotated}/{len(samples)} "
          f"({100.0 * n_annotated / len(samples):.1f}%)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PlantDocCropDataset(samples, transform=transform, margin=args.margin)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model, model_info = build_model(args.model, num_classes=len(PV_CLASSES), pretrained=False)
    set_bn_settings(model, eps=args.bn_eps, momentum=args.bn_momentum)
    checkpoint = _resolve_weight(args.weight)
    info = load_checkpoint(model, checkpoint, device=device)
    n_out = _classifier_out_features(model)
    print(f"Checkpoint: {checkpoint} (epoch {info.get('epoch')})")
    print(f"Classifier outputs: {n_out}, PlantVillage classes: {len(PV_CLASSES)}")
    if n_out == len(PV_CLASSES):
        label_space = list(PV_CLASSES)
    elif n_out == len(PV_CLASSES) - 1:
        # The only 28-class condition in the audit is the no-background model.
        label_space = [c for c in PV_CLASSES if c != BACKGROUND_CLASS]
        print("Using the 28-class label space (background class excluded)")
    elif n_out is not None and not args.allow_class_count_mismatch:
        raise SystemExit(
            f"Checkpoint has {n_out} outputs but the audit expects "
            f"{len(PV_CLASSES)} or {len(PV_CLASSES) - 1} PlantVillage classes. "
            "Pass --allow-class-count-mismatch only if you know the label order."
        )
    else:
        label_space = list(PV_CLASSES)
    model = model.to(device).eval()

    rows = []
    y_true, pred_full, pred_crop = [], [], []
    conf_full, conf_crop = [], []
    annotated_flags = []
    with torch.inference_mode():
        for full, cropped, labels, indices, has_annotation, n_boxes in loader:
            logits_full = model(full.to(device))
            logits_crop = model(cropped.to(device))
            probs_full = torch.softmax(logits_full, dim=1)
            probs_crop = torch.softmax(logits_crop, dim=1)
            cf, pf = probs_full.max(dim=1)
            cc, pc = probs_crop.max(dim=1)
            for label, p_full, p_crop, c_full, c_crop, idx, has_ann, n_box in zip(
                labels.tolist(), pf.tolist(), pc.tolist(), cf.tolist(), cc.tolist(),
                indices.tolist(), has_annotation.tolist(), n_boxes.tolist()
            ):
                sample = samples[idx]
                true_name = sample["pv_class"]
                full_name = label_space[p_full]
                crop_name = label_space[p_crop]
                y_true.append(true_name)
                pred_full.append(full_name)
                pred_crop.append(crop_name)
                conf_full.append(c_full)
                conf_crop.append(c_crop)
                annotated_flags.append(bool(has_ann))
                rows.append({
                    "image": os.path.relpath(sample["path"], args.plantdoc_dir),
                    "pd_class": sample["pd_class"],
                    "true": true_name,
                    "pred_full": full_name,
                    "pred_crop": crop_name,
                    "correct_full": int(full_name == true_name),
                    "correct_crop": int(crop_name == true_name),
                    "conf_full": f"{c_full:.6f}",
                    "conf_crop": f"{c_crop:.6f}",
                    "n_boxes": int(n_box),
                    "cropped": int(has_ann),
                })

    full_metrics = classification_metrics(y_true, pred_full, EVAL_CLASSES)
    crop_metrics = classification_metrics(y_true, pred_crop, EVAL_CLASSES)

    # The paired intervention test uses only images that actually have a box;
    # for the others the crop is identical to the full image by construction.
    keep = [i for i, flag in enumerate(annotated_flags) if flag]
    if not keep:
        print("[error] no annotations were found for the evaluated images")
        print("        pass --annotations <dir> or check the XML stems")
        annotated = None
    else:
        y_true_sub = [y_true[i] for i in keep]
        full_sub = [pred_full[i] for i in keep]
        crop_sub = [pred_crop[i] for i in keep]
        mcnemar = mcnemar_exact(y_true_sub, full_sub, crop_sub)
        bootstrap = paired_bootstrap_difference(
            y_true_sub, crop_sub, full_sub,
            n_bootstrap=args.bootstrap, seed=args.seed,
        )
        full_sub_metrics = classification_metrics(y_true_sub, full_sub, EVAL_CLASSES)
        crop_sub_metrics = classification_metrics(y_true_sub, crop_sub, EVAL_CLASSES)

        # Per-class recall change (percentage points).
        per_class_delta = {}
        for cls in EVAL_CLASSES:
            per_class_delta[cls] = {
                "recall_full": full_sub_metrics["per_class"][cls]["recall"],
                "recall_crop": crop_sub_metrics["per_class"][cls]["recall"],
                "delta_pp": crop_sub_metrics["per_class"][cls]["recall"]
                            - full_sub_metrics["per_class"][cls]["recall"],
            }

        annotated = {
            "n": len(keep),
            "full": full_sub_metrics,
            "crop": crop_sub_metrics,
            "delta_accuracy_pp": crop_sub_metrics["accuracy"] - full_sub_metrics["accuracy"],
            "mcnemar": mcnemar,
            "bootstrap_crop_minus_full": bootstrap,
            "per_class_delta": per_class_delta,
        }

    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    out = args.out or os.path.join(RESULTS_DIR, "predictions", f"crop_{stem}.csv")
    write_predictions(
        out, rows,
        fieldnames=[
            "image", "pd_class", "true", "pred_full", "pred_crop",
            "correct_full", "correct_crop", "conf_full", "conf_crop",
            "n_boxes", "cropped",
        ],
    )

    payload = {
        "probe": "leaf_crop_intervention",
        "model": args.model,
        "model_info": model_info,
        "checkpoint": checkpoint,
        "checkpoint_epoch": info.get("epoch"),
        "plantdoc_dir": args.plantdoc_dir,
        "annotation_dirs": annotation_dirs,
        "n_images": len(samples),
        "n_annotated": n_annotated,
        "annotation_coverage": 100.0 * n_annotated / len(samples),
        "margin": args.margin,
        "all_images": {"full": full_metrics, "crop": crop_metrics},
        "annotated_subset": annotated,
        "predictions_csv": out,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = args.metrics_json or os.path.join(
        RESULTS_DIR, "metrics", f"crop_{stem}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Leaf-crop intervention ===")
    print(f"Annotated subset: {len(keep)} images")
    if annotated is not None:
        print(f"Full   acc {annotated['full']['accuracy']:.2f}% | "
              f"macro F1 {annotated['full']['f1']:.2f}%")
        print(f"Cropped acc {annotated['crop']['accuracy']:.2f}% | "
              f"macro F1 {annotated['crop']['f1']:.2f}%")
        print(f"Delta accuracy: {annotated['delta_accuracy_pp']:+.2f} pp")
        print(f"Paired bootstrap 95% CI: "
              f"[{annotated['bootstrap_crop_minus_full']['ci_low_pp']:+.2f}, "
              f"{annotated['bootstrap_crop_minus_full']['ci_high_pp']:+.2f}] pp")
        print(f"McNemar exact p = {annotated['mcnemar']['p_value']:.3g} "
              f"(n01={annotated['mcnemar']['n01']}, n10={annotated['mcnemar']['n10']})")
    else:
        print("No paired intervention was possible (missing annotations).")
    print(f"Predictions: {out}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
