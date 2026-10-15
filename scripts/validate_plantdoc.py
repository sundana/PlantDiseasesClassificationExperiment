import sys
import os
import warnings

# Suppress visualization windows and noisy sklearn warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Only one class.*')
warnings.filterwarnings('ignore', message='.*No positive samples.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')

import torch

# Fix Unicode emoji printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import csv
from functools import partial
import numpy as np
from tqdm import tqdm

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2, get_efficientnet_b0
from src.config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLANTDOC_DIR
from src.utils import evaluate_model

# Mapping from PlantDoc directory names to PlantVillage class names
PLANTDOC_PV_MAPPING = {
    'Apple leaf': 'Apple___healthy',
    'Apple rust leaf': 'Apple___Cedar_apple_rust',
    'Apple Scab Leaf': 'Apple___Apple_scab',
    'Blueberry leaf': 'Blueberry___healthy',
    'Cherry leaf': 'Cherry___healthy',
    'grape leaf': 'Grape___healthy',
    'grape leaf black rot': 'Grape___Black_rot',
    'Peach leaf': 'Peach___healthy',
    'Raspberry leaf': 'Raspberry___healthy',
    'Squash Powdery mildew leaf': 'Squash___Powdery_mildew',
    'Strawberry leaf': 'Strawberry___healthy',
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite'
}

PV_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Raspberry___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ---------------------------------------------------------------------------
# Deterministic checkpoint selection
# ---------------------------------------------------------------------------
# Previously, checkpoints were auto-discovered from an unsorted os.listdir()
# scan and the first match was used. Because augment / DANN / fine-tuned
# checkpoints share the same filename prefixes, this silently evaluated the
# wrong model (e.g. model key "proposed_large_32" could resolve to
# "proposed_large_32_augment_epoch_best.pth", turning a 6.51% result into
# 9.93%), making the reported zero-shot numbers unreproducible.
#
# Every stem below MUST resolve to a 29-class PlantVillage model. PlantDoc
# fine-tuned checkpoints (28 classes) and augment/DANN checkpoints must never
# be used for zero-shot PlantVillage -> PlantDoc evaluation.
ZERO_SHOT_CHECKPOINTS = {
    # MobileNetV3 Small
    "mobilenetv3_small":        "MobileNetV3_Small_SE_epoch_19",
    "mobilenetv3_small_se_r16": "MobileNetV3_Small_SE_r16_epoch_24",
    "mobilenetv3_small_se_r32": "MobileNetV3_Small_SE_r32_epoch_33",
    "mobilenetv3_small_none":   "MobileNetV3_Small_NoAttn_epoch_28",
    "proposed_small_4":         "MobileNetV3_Small_CBAM_r4_epoch_15",
    "proposed_small_16":        "MobileNetV3_Small_CBAM_r16_epoch_12",
    "proposed_small_32":        "MobileNetV3_Small_CBAM_r32_epoch_26",
    # MobileNetV3 Large
    "mobilenetv3_large":        "MobileNetV3_Large_SE_epoch_39",
    "mobilenetv3_large_se_r16": "MobileNetV3_Large_SE_r16_epoch_16",
    "mobilenetv3_large_se_r32": "MobileNetV3_Large_SE_r32_epoch_19",
    "mobilenetv3_large_none":   "MobileNetV3_Large_NoAttn_epoch_33",
    "proposed_large_4":         "MobileNetV3_Large_CBAM_r4_epoch_22",
    "proposed_large_16":        "MobileNetV3_Large_CBAM_r16_epoch_22",
    "proposed_large_32":        "MobileNetV3_Large_CBAM_r32_epoch_26",
    # Baselines
    "mobilenetv2":              "MobileNetV2_epoch_34",
    "shufflenetv2":             "ShuffleNetV2_epoch_35",
    "efficientnet_b0":          "EfficientNet_epoch_25",
}

# Final-classifier keys used by the three backbone families in this repo.
_CLASSIFIER_KEYS = ("linear4.weight", "classifier.1.weight", "fc.weight")


def _state_dict_classifier(state_dict):
    """Return (key, out_features) for a checkpoint's final classifier layer."""
    for key in _CLASSIFIER_KEYS:
        if key in state_dict:
            weight = state_dict[key]
            if weight.dim() == 2:
                return key, int(weight.shape[0])
    return None, None


def resolve_checkpoint(model_name, weight):
    """Resolve a checkpoint path deterministically.

    Explicit ``--weight`` wins. Otherwise the model key is looked up in
    ZERO_SHOT_CHECKPOINTS. There is intentionally no filesystem-order fallback:
    failing loudly is preferable to evaluating the wrong checkpoint.
    """
    if weight:
        return weight, os.path.join(CHECKPOINT_DIR, f"{weight}.pth"), "--weight"
    stem = ZERO_SHOT_CHECKPOINTS.get(model_name)
    if stem is None:
        return None, None, None
    return stem, os.path.join(CHECKPOINT_DIR, f"{stem}.pth"), "ZERO_SHOT_CHECKPOINTS"


class PlantDocMappedDataset(Dataset):
    def __init__(self, root_dir, mapping, pv_classes, transform=None):
        self.root_dir = root_dir
        self.mapping = mapping
        self.pv_classes = pv_classes
        self.transform = transform
        
        self.samples = []
        print(f"📦 Filtering PlantDoc dataset for mapped classes...")
        for pd_class, pv_class in mapping.items():
            pd_path = os.path.join(root_dir, pd_class)
            if not os.path.exists(pd_path):
                print(f"⚠️  Warning: PD class path not found: {pd_path}")
                continue
            
            pv_idx = pv_classes.index(pv_class)
            class_samples = 0
            for img_name in os.listdir(pd_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(pd_path, img_name), pv_idx))
                    class_samples += 1
            # print(f"   • {pd_class} -> {pv_class}: {class_samples} images")

        print(f"✅ Total mapped samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a dummy image or handle appropriately
            return torch.zeros((3, 224, 224)), label

def main():
    parser = argparse.ArgumentParser(description="Validate model on PlantDoc dataset")
    parser.add_argument("-m", "--model", help="Model name (or 'all' for all variants)", required=True)
    parser.add_argument("-w", "--weight", help="Model weight file name (ignored if --model=all)", default=None)
    parser.add_argument("-d", "--device", help="Device (cuda/cpu)", default="auto")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs for accuracy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from src.config import set_seed
    set_seed(args.seed)

    print(f"PlantDoc Validation on device: {device}")

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
        print("❌ No samples found in PlantDoc dataset.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model_map = {
        "mobilenetv3_small": partial(MobileNetV3_Small, attention_type='se', reduction_ratio=4),
        "mobilenetv3_large": partial(MobileNetV3_Large, attention_type='se', reduction_ratio=4),
        "mobilenetv3_small_none": partial(MobileNetV3_Small, attention_type='none'),
        "mobilenetv3_large_none": partial(MobileNetV3_Large, attention_type='none'),
        "mobilenetv3_small_se_r16": partial(MobileNetV3_Small, attention_type='se', reduction_ratio=16),
        "mobilenetv3_large_se_r16": partial(MobileNetV3_Large, attention_type='se', reduction_ratio=16),
        "mobilenetv3_small_se_r32": partial(MobileNetV3_Small, attention_type='se', reduction_ratio=32),
        "mobilenetv3_large_se_r32": partial(MobileNetV3_Large, attention_type='se', reduction_ratio=32),
        "proposed_large_4": partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=4),
        "proposed_large_16": partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
        "proposed_large_32": partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=32),
        "proposed_small_4": partial(MobileNetV3_Small, attention_type='cbam', reduction_ratio=4),
        "proposed_small_16": partial(MobileNetV3_Small, attention_type='cbam', reduction_ratio=16),
        "proposed_small_32": partial(MobileNetV3_Small, attention_type='cbam', reduction_ratio=32),
        "mobilenetv2": get_mobilenet_v2,
        "shufflenetv2": get_shufflenet_v2,
        "efficientnet_b0": get_efficientnet_b0,
    }

    # Determine which models to evaluate
    if args.model == "all":
        models_to_eval = list(model_map.keys())
    else:
        if args.model not in model_map:
            print(f"Unknown model: {args.model}")
            return
        models_to_eval = [args.model]

    all_results = {}

    for model_name in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")

        model_factory = model_map[model_name]
        model = model_factory(num_classes=len(PV_CLASSES)).to(device)

        # --- Deterministic checkpoint selection (no os.listdir() guessing) ---
        weight_stem, checkpoint_path, source = resolve_checkpoint(model_name, args.weight)
        if weight_stem is None:
            print(f"[SKIP] No zero-shot checkpoint registered for '{model_name}'. "
                  f"Pass --weight explicitly to override.")
            continue
        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] Checkpoint not found: {checkpoint_path}")
            continue
        print(f"Checkpoint: {os.path.basename(checkpoint_path)} (via {source})")

        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # --- Label-space guard ---------------------------------------------
        # Zero-shot evaluation maps PlantDoc images onto the 29-class
        # PlantVillage label set, so the checkpoint's classifier MUST have 29
        # outputs. Fine-tuned (28-class) or otherwise mis-typed checkpoints
        # would silently produce meaningless accuracy; reject them loudly.
        cls_key, ckpt_classes = _state_dict_classifier(state_dict)
        if ckpt_classes is None:
            print(f"[WARN] Could not locate a classifier layer in "
                  f"{os.path.basename(checkpoint_path)}; class count unverified.")
        elif ckpt_classes != len(PV_CLASSES):
            print(f"[SKIP] {os.path.basename(checkpoint_path)} has {ckpt_classes} "
                  f"outputs but zero-shot evaluation requires {len(PV_CLASSES)} "
                  f"PlantVillage classes ({cls_key}). This is likely a PlantDoc "
                  f"fine-tuned or augment/DANN checkpoint, not a PlantVillage model.")
            continue

        # Handle key mismatch
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
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"[SKIP] Failed to load checkpoint: {e}")
            continue
        if missing or unexpected:
            # A partial/silent load invalidates the metric, so fail fast rather
            # than reporting an accuracy computed from partly-random weights.
            print(f"[SKIP] Incomplete weight load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys")
            if missing:
                print(f"       Missing (first 3): {missing[:3]}")
            if unexpected:
                print(f"       Unexpected (first 3): {unexpected[:3]}")
            continue

        print(f"Loaded weights from {checkpoint_path}")

        # Run evaluation
        results_dir = os.path.join(RESULTS_DIR, "plantdoc_validation", model_name)
        os.makedirs(results_dir, exist_ok=True)

        cm, metrics, final_loss = evaluate_model(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            test_loader=dataloader,
            class_names=PV_CLASSES,
            device=device,
            results_dir=results_dir
        )

        all_results[model_name] = {
            "checkpoint": f"{weight_stem}.pth",
            "samples": len(dataset),
            "accuracy": metrics.get("overall", {}).get("Accuracy", 0.0),
            "macro_f1": float(np.mean([v["F1 Score"] for v in metrics["per_class"].values()]))
            if metrics.get("per_class") else 0.0,
        }

        print(f"Accuracy: {all_results[model_name]['accuracy'] * 100:.2f}% "
              f"({int(all_results[model_name]['accuracy'] * len(dataset))}/{len(dataset)})")

        # Bootstrap CI if requested
        if args.bootstrap:
            from src.statistical_tests import bootstrap_accuracy_ci
            # Collect predictions
            all_preds, all_labels = [], []
            model.eval()
            with torch.inference_mode():
                for images, labels in dataloader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.numpy().tolist())
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            ci_low, ci_mean, ci_high = bootstrap_accuracy_ci(all_labels, all_preds, seed=args.seed)
            print(f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%] (mean: {ci_mean*100:.2f}%)")

    # Summary table
    if all_results:
        print(f"\n{'='*76}")
        print("PLANTDOC ZERO-SHOT SUMMARY (mapped 20-class subset, 1,783 samples)")
        print(f"{'Model':<32}{'Checkpoint':<38}{'Acc (%)':>9}")
        print("-" * 79)
        for name, res in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"{name:<32}{res['checkpoint']:<38}{res['accuracy'] * 100:>9.2f}")

        summary_path = os.path.join(RESULTS_DIR, "plantdoc_validation_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["model", "checkpoint", "samples", "accuracy_pct", "macro_f1"])
            for name, res in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
                writer.writerow([
                    name, res["checkpoint"], res["samples"],
                    f"{res['accuracy'] * 100:.2f}", f"{res['macro_f1']:.4f}",
                ])
        print(f"\nSummary written to {summary_path}")

    print(f"\nValidation on PlantDoc finished!")

if __name__ == "__main__":
    main()
