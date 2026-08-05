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

        # Try to find the checkpoint
        if args.weight:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.weight}.pth")
        else:
            # Auto-discover checkpoint
            candidates = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(model_name.split('_')[0])]
            if not candidates:
                # Try by looking for model name prefix
                candidates = [f for f in os.listdir(CHECKPOINT_DIR) if f.lower().startswith(model_name.lower())]
            if not candidates:
                print(f"No checkpoint found for {model_name}, skipping.")
                continue
            checkpoint_path = os.path.join(CHECKPOINT_DIR, candidates[0])
            print(f"Using checkpoint: {candidates[0]}")

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping.")
            continue

        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

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
            if missing or unexpected:
                print(f"Non-strict load: {len(missing)} missing, {len(unexpected)} unexpected keys")
                if missing:
                    print(f"  Missing (first 3): {missing[:3]}")
                if unexpected:
                    print(f"  Unexpected (first 3): {unexpected[:3]}")
        except RuntimeError as e:
            print(f"Failed to load checkpoint: {e}")
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
            "accuracy": metrics.get("accuracy", 0.0),
            "macro_f1": metrics.get("macro avg", {}).get("f1-score", 0.0) if "macro avg" in metrics else metrics.get("macro_f1", 0.0),
        }

        print(f"Accuracy: {all_results[model_name]['accuracy']:.2f}%")

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
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("PLANTDOC ZERO-SHOT SUMMARY")
        print(f"{'Model':<35} {'Accuracy':>10}")
        print("-" * 47)
        for name, res in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"{name:<35} {res['accuracy']:>9.2f}%")

    print(f"\nValidation on PlantDoc finished!")

if __name__ == "__main__":
    main()
