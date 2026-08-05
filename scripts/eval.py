import sys
import os

# Fix Unicode emoji printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.utils import evaluate_model
from src.data_setup import create_dataloader
from src.config import DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLANTVILLAGE_DIR
from torchvision import transforms
import argparse
from functools import partial


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
    parser.add_argument("-m", "--model", help="Model name", required=True)
    parser.add_argument("-w", "--weight", help="Model weight file name", required=True)
    parser.add_argument("-d", "--device", help="Device (cuda/cpu)", default="auto")
    parser.add_argument("-data", "--data_dir", default=PLANTVILLAGE_DIR, help="Path to evaluation dataset")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--results_dir", default=RESULTS_DIR, help="Directory to save results"
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🔍 Starting evaluation on device: {device}")
    
    # Print hardware information
    import platform
    import psutil
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"🧠 CPU: {platform.processor()}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📼 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"🐏 RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("📁 Loading datasets...")
    train_loader, val_loader, test_loader, class_names = create_dataloader(
        data_path=args.data_dir,
        train_transform=transform,
        test_transform=transform,
        batch_size=args.batch_size,
    )

    num_classes = len(class_names)
    print(f"📊 Found {num_classes} classes: {class_names}")

    # Load model based on model type
    print(f"🤖 Loading model: {args.model}")

    from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
    from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2, get_efficientnet_b0

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

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.weight}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    try:
        model_factory = model_map.get(args.model)
        if model_factory is None:
            print(f"❌ Unknown model type: {args.model}")
            return
        
        model = model_factory(num_classes=num_classes)

        # Load model weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Unpack state_dict if it was saved as a dict with 'model_state_dict'
        if "model_state_dict" in state_dict:
            state_dict_to_load = state_dict["model_state_dict"]
            print(f"✅ Loaded model checkpoint from epoch {state_dict.get('epoch', 'unknown')}")
        else:
            state_dict_to_load = state_dict
            print("✅ Loaded model weights")

        # Map keys to handle architecture refactoring (.se.se -> .attention_module.se)
        mapped_state_dict = {}
        for k, v in state_dict_to_load.items():
            if ".se.se." in k:
                k = k.replace(".se.se.", ".attention_module.se.")
            if ".module.channel_attention.fc1." in k:
                k = k.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
            if ".module.channel_attention.fc2." in k:
                k = k.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
            if ".module.spatial_attention.conv." in k:
                k = k.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
            mapped_state_dict[k] = v
            
        model.load_state_dict(mapped_state_dict)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print("\n🔬 Starting comprehensive evaluation...")

    # Enhanced evaluation with better visualization
    cm, performance_metrics, final_loss = evaluate_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        results_dir=args.results_dir,
    )

    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
