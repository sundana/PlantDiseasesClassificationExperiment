"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import matplotlib
matplotlib.use('Agg') # Prevents any matplotlib windows from opening

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
from src import data_setup, engine
from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2, get_efficientnet_b0
from src.config import DATA_DIR, CHECKPOINT_DIR, PLANTVILLAGE_DIR, DEFAULT_SEED, set_seed
from torchvision import transforms
from functools import partial


def start_training(
    model_name: str,
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    data_dir: str = PLANTVILLAGE_DIR,
    device_name: str = "auto",
    no_plot: bool = False,
    seed: int = DEFAULT_SEED,
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
        )
    )

    # Model selection using dictionary mapping
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

    # Get model class from map or default to MobileNetV3_Large (SE)
    model_factory = model_map.get(model_name, partial(MobileNetV3_Large, attention_type='se'))
    model = model_factory(num_classes=len(class_names)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Show training config
    print(f"Model: {model.__class__.__name__}")
    print(f"Epoch: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Loss function: {loss_fn.__class__.__name__}")
    print(f"Optimizer: {optimizer}")
    print(f"Device: {device}")
    print(f"Data Directory: {data_dir}")

    # Start training with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=num_epochs,
        device=torch.device(device),
        patience=5,
        early_stopping=True,
        min_delta=0,
        checkpoint_dir=CHECKPOINT_DIR,
        enable_live_plot=not no_plot,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for train model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="mobilenetv3_large",
        choices=[
            "mobilenetv3_small",
            "mobilenetv3_large",
            "mobilenetv3_small_none",
            "mobilenetv3_large_none",
            "mobilenetv3_small_se_r16",
            "mobilenetv3_large_se_r16",
            "mobilenetv3_small_se_r32",
            "mobilenetv3_large_se_r32",
            "proposed_large_4",
            "proposed_large_16",
            "proposed_large_32",
            "proposed_small_4",
            "proposed_small_16",
            "proposed_small_32",
            "mobilenetv2",
            "shufflenetv2",
            "efficientnet_b0",
        ],
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
    args = parser.parse_args()
    model_name = args.model
    num_epochs = args.epochs
    data_dir = args.data_dir
    device_name = args.device
    no_plot = not args.plot
    start_training(
        model_name,
        num_epochs=num_epochs,
        data_dir=data_dir,
        device_name=device_name,
        no_plot=no_plot,
        seed=args.seed,
    )
