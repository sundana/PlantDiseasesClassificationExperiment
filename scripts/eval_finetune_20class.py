"""Evaluate fine-tuned (28-class) models on the 20-class mapped PlantDoc subset
for direct comparison with zero-shot results."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.config import CHECKPOINT_DIR, PLANTDOC_DIR, DA_CHECKPOINT_DIR, DOMAIN_ADAPT_DIR
from scripts.validate_plantdoc import PLANTDOC_PV_MAPPING, PV_CLASSES

MODELS_CONFIG = [
    ("Large SE r=4 (finetuned)", partial(MobileNetV3_Large, attention_type='se'),
     "mobilenetv3_large_finetune_best.pth", "Finetune", "#2196F3"),
    ("Large CBAM r=32 (finetuned)", partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=32),
     "MobileNetV3_Large_finetune_best.pth", "FinetuneCBAM", "#FF9800"),
    ("Large No Attn (finetuned)", partial(MobileNetV3_Large, attention_type='none'),
     "mobilenetv3_large_none_finetune_best.pth", "FinetuneNone", "#4CAF50"),
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load PlantDoc ImageFolder to get class name -> index mapping
    pd_dataset = datasets.ImageFolder(root=PLANTDOC_DIR)
    pd_class_to_idx = pd_dataset.class_to_idx
    pd_idx_to_class = {v: k for k, v in pd_class_to_idx.items()}
    print(f"PlantDoc classes ({len(pd_dataset.classes)}): {pd_dataset.classes}")

    # Build mapping: PlantDoc class index -> PlantVillage class index (if mapped)
    pd_idx_to_pv_idx = {}
    for pd_class_name, pv_class_name in PLANTDOC_PV_MAPPING.items():
        pd_idx = pd_class_to_idx[pd_class_name]
        pv_idx = PV_CLASSES.index(pv_class_name)
        pd_idx_to_pv_idx[pd_idx] = pv_idx

    mapped_pd_indices = set(pd_idx_to_pv_idx.keys())
    print(f"Mapped PlantDoc indices: {sorted(mapped_pd_indices)}")

    # Gather all samples and their PlantDoc class indices
    all_paths = [s[0] for s in pd_dataset.samples]
    all_pd_targets = [s[1] for s in pd_dataset.samples]

    # Identify test samples from the fine-tune data split
    # We'll evaluate on ALL mapped images (same as zero-shot evaluation)
    results = []

    for label, model_fn, ckpt_name, tag, color in MODELS_CONFIG:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        ckpt_path = os.path.join(DA_CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        model = model_fn(num_classes=len(pd_dataset.classes)).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Key remapping (same as eval.py)
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

        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("  Loaded successfully (strict=True)")
        except RuntimeError as e:
            missing = [k for k in model.state_dict() if k not in new_state_dict]
            unexpected = [k for k in new_state_dict if k not in model.state_dict()]
            print(f"  Strict load failed. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"    First 5 missing: {missing[:5]}")
            if unexpected:
                print(f"    First 5 unexpected: {unexpected[:5]}")
            model.load_state_dict(new_state_dict, strict=False)
            print("  Loaded with strict=False")

        model.eval()

        # Run inference on each mapped image
        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}

        for img_path, pd_target in zip(all_paths, all_pd_targets):
            if pd_target not in mapped_pd_indices:
                continue

            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.inference_mode():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()

            total += 1
            pd_class_name = pd_idx_to_class[pd_target]
            per_class_total[pd_class_name] = per_class_total.get(pd_class_name, 0) + 1

            if pred == pd_target:
                correct += 1
                per_class_correct[pd_class_name] = per_class_correct.get(pd_class_name, 0) + 1

        acc = correct / total * 100 if total > 0 else 0.0
        print(f"\n  20-class subset accuracy: {acc:.2f}% ({correct}/{total})")

        print(f"\n  Per-class accuracy:")
        for cls in sorted(per_class_total.keys()):
            c = per_class_correct.get(cls, 0)
            t = per_class_total[cls]
            print(f"    {cls}: {c/t*100:.1f}% ({c}/{t})")

        results.append({
            "label": label,
            "acc": acc,
            "correct": correct,
            "total": total,
        })

    print(f"\n{'='*60}")
    print("Summary:")
    for r in results:
        print(f"  {r['label']}: {r['acc']:.2f}% ({r['correct']}/{r['total']})")

if __name__ == "__main__":
    main()
