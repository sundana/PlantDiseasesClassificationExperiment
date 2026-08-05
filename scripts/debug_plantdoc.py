"""
Diagnostic script to investigate the 0.00% PlantDoc accuracy for proposed_large_16.
Checks: class mapping alignment, checkpoint state dict keys, prediction distribution.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from functools import partial

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.config import DATA_DIR, CHECKPOINT_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR, RESULTS_DIR
from scripts.validate_plantdoc import PlantDocMappedDataset, PLANTDOC_PV_MAPPING, PV_CLASSES

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Get actual class order from PlantVillage ImageFolder
    pv_dataset = datasets.ImageFolder(PLANTVILLAGE_DIR)
    actual_classes = pv_dataset.classes
    print(f"\n=== PlantVillage classes from ImageFolder ({len(actual_classes)}):")
    for i, c in enumerate(actual_classes):
        print(f"  [{i}] {c}")

    # 2. Compare with PV_CLASSES hardcoded list
    print(f"\n=== Hardcoded PV_CLASSES ({len(PV_CLASSES)}):")
    for i, c in enumerate(PV_CLASSES):
        print(f"  [{i}] {c}")

    print(f"\n=== Class ordering match: {actual_classes == PV_CLASSES}")
    if actual_classes != PV_CLASSES:
        print("MISMATCHES:")
        for i, (a, b) in enumerate(zip(actual_classes, PV_CLASSES)):
            if a != b:
                print(f"  Index {i}: ImageFolder='{a}' vs PV_CLASSES='{b}'")

    # 3. Check PlantDoc mapping coverage
    print(f"\n=== PlantDoc mapping ({len(PLANTDOC_PV_MAPPING)} classes):")
    mapped_indices = set()
    unmapped_pv = []
    for pd_class, pv_class in PLANTDOC_PV_MAPPING.items():
        idx = PV_CLASSES.index(pv_class)
        mapped_indices.add(idx)
        print(f"  '{pd_class}' -> [{idx}] '{pv_class}'")
    
    all_indices = set(range(len(PV_CLASSES)))
    unmapped = all_indices - mapped_indices
    print(f"\nMapped PV indices: {sorted(mapped_indices)}")
    print(f"Unmapped PV indices: {sorted(unmapped)}")
    for idx in sorted(unmapped):
        print(f"  [{idx}] {PV_CLASSES[idx]} (NOT in PlantDoc mapping)")

    # 4. Load the problematic checkpoint and check predictions
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PlantDocMappedDataset(
        root_dir=PLANTDOC_DIR,
        mapping=PLANTDOC_PV_MAPPING,
        pv_classes=PV_CLASSES,
        transform=transform
    )

    if len(dataset) == 0:
        print("No PlantDoc samples found!")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load both problematic model (CBAM r16) and working model (SE Small)
    models_to_test = [
        ("mobilenetv3_small", partial(MobileNetV3_Small, attention_type='se'),
         "MobileNetV3_Small_SE_epoch_19", "Small SE (baseline)"),
        ("proposed_large_16", partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
         "MobileNetV3_Large_CBAM_r16_epoch_22", "Large CBAM r16 (0% on PlantDoc)"),
        ("mobilenetv3_large", partial(MobileNetV3_Large, attention_type='se'),
         "MobileNetV3_Large_SE_epoch_39", "Large SE"),
    ]

    for model_name, model_fn, ckpt_name, label in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"  Model key: {model_name}")
        print(f"  Checkpoint: {ckpt_name}.pth")

        model = model_fn(num_classes=len(PV_CLASSES)).to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pth")

        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Apply key remapping
        new_state_dict = {}
        for k, v in state_dict.items():
            if ".se.se." in k:
                k = k.replace(".se.se.", ".attention_module.se.")
            if ".module.channel_attention.fc1." in k:
                k = k.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
            if ".module.channel_attention.fc2." in k:
                k = k.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
            if ".module.spatial_attention.conv." in k:
                k = k.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
            if "bneck" in k and ".module." in k:
                k = k.replace(".module.", ".attention_module.")
            new_state_dict[k] = v

        # Attempt load
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("  State dict loaded successfully (strict=True)")
        except RuntimeError as e:
            missing = [k for k in model.state_dict() if k not in new_state_dict]
            unexpected = [k for k in new_state_dict if k not in model.state_dict()]
            print(f"  Strict load failed. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"    First 5 missing: {missing[:5]}")
            if unexpected:
                print(f"    First 5 unexpected: {unexpected[:5]}")
            # Try non-strict
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"  Non-strict load: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        model.eval()

        # Run inference and collect predictions
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Analysis
        accuracy = (all_preds == all_labels).mean() * 100
        print(f"\n  Accuracy: {accuracy:.2f}% ({int(accuracy * len(all_labels) / 100)}/{len(all_labels)})")

        # Prediction distribution
        pred_counts = Counter(all_preds)
        print(f"\n  Prediction distribution (top 10 predicted classes):")
        for pred_idx, count in pred_counts.most_common(10):
            class_name = PV_CLASSES[pred_idx]
            is_mapped = pred_idx in mapped_indices
            pct = count / len(all_preds) * 100
            print(f"    [{pred_idx}] {class_name}: {count} ({pct:.1f}%) {'[MAPPED]' if is_mapped else '[UNMAPPED]'}")

        # How many predictions fall on unmapped classes?
        unmapped_preds = sum(1 for p in all_preds if p not in mapped_indices)
        mapped_preds = sum(1 for p in all_preds if p in mapped_indices)
        print(f"\n  Predictions on mapped classes: {mapped_preds} ({mapped_preds/len(all_preds)*100:.1f}%)")
        print(f"  Predictions on unmapped classes: {unmapped_preds} ({unmapped_preds/len(all_preds)*100:.1f}%)")

        # Top-1 predicted class
        top_pred_idx = pred_counts.most_common(1)[0][0]
        print(f"\n  Most common prediction: [{top_pred_idx}] {PV_CLASSES[top_pred_idx]} "
              f"({pred_counts[top_pred_idx]} times, {pred_counts[top_pred_idx]/len(all_preds)*100:.1f}%)")

        # Per-class accuracy for mapped classes
        print(f"\n  Per-class accuracy on mapped classes:")
        for idx in sorted(mapped_indices):
            mask = all_labels == idx
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == idx).mean() * 100
                print(f"    [{idx}] {PV_CLASSES[idx]}: {cls_acc:.1f}% ({int(cls_acc*mask.sum()/100)}/{mask.sum()})")
            else:
                print(f"    [{idx}] {PV_CLASSES[idx]}: no samples")

        # Check if model weight statistics look reasonable
        print(f"\n  Model parameter stats:")
        total_params = sum(p.numel() for p in model.parameters())
        for name, param in model.named_parameters():
            if 'weight' in name and param.ndim > 1:
                print(f"    {name}: mean={param.mean():.4f}, std={param.std():.4f}, "
                      f"min={param.min():.4f}, max={param.max():.4f}")
                break

    # 5. Compare training class order vs PV_CLASSES
    print(f"\n{'='*60}")
    print("Verifying checkpoint was trained with same class order...")
    
    # Check a checkpoint's classifier weight shape
    for ckpt_name in ["MobileNetV3_Large_CBAM_r16_epoch_22", "MobileNetV3_Large_SE_epoch_39"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pth")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            classifier_key = None
            for k in sd:
                if 'linear4.weight' in k:
                    classifier_key = k
                    break
            if classifier_key:
                shape = sd[classifier_key].shape
                print(f"  {ckpt_name}: classifier shape = {shape}, expected (29, 1280)")

if __name__ == "__main__":
    main()
