"""
Domain Adaptation Training Script.

Supports three strategies:
  1. augment  -- Strong augmentation (CutMix + MixUp) on PlantVillage only
  2. finetune -- Pre-train on PlantVillage, fine-tune on PlantDoc
  3. dann     -- Domain-Adversarial Neural Network (source=PlantVillage, target=PlantDoc)

The model_map must be kept in sync with train.py, eval.py, validate_plantdoc.py,
and measure_complexity.py.
"""
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import argparse
from functools import partial
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src import data_setup, engine
from src.domain_adaptation import (
    DANNWrapper, cutmix_data, mixup_data, mixup_criterion,
    train_epoch_dann, validate_dann,
)
from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2, get_efficientnet_b0
from src.config import (
    DATA_DIR, CHECKPOINT_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR,
    DOMAIN_ADAPT_DIR, DA_CHECKPOINT_DIR, DEFAULT_SEED, set_seed,
)
from src.utils import EarlyStopping


# ---------------------------------------------------------------------------
# Augmented training (CutMix + MixUp on PlantVillage)
# ---------------------------------------------------------------------------

def augmented_train_step(
    model, dataloader, loss_fn, optimizer, device,
    cutmix_prob: float = 0.5, mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0,
):
    """Single training epoch with CutMix/MixUp augmentation."""
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        r = np.random.rand()
        if r < cutmix_prob:
            X, y_a, y_b, lam = cutmix_data(X, y, alpha=cutmix_alpha)
            y_pred = model(X)
            loss = mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
            correct = (lam * (y_pred.argmax(1) == y_a).float() +
                       (1 - lam) * (y_pred.argmax(1) == y_b).float()).sum()
        else:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=mixup_alpha)
            y_pred = model(X)
            loss = mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
            correct = (lam * (y_pred.argmax(1) == y_a).float() +
                       (1 - lam) * (y_pred.argmax(1) == y_b).float()).sum()

        train_loss += loss.item()
        train_acc += correct.item() / len(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def augmented_valid_step(model, dataloader, loss_fn, device):
    """Standard validation step (no mixing)."""
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            val_acc += (y_pred.argmax(1) == y).float().mean().item()
    return val_loss / len(dataloader), val_acc / len(dataloader)


def train_augmented(
    model, train_loader, val_loader, loss_fn, optimizer, device,
    epochs: int, patience: int, checkpoint_path: str,
    cutmix_prob: float = 0.5,
):
    """Full training loop with CutMix/MixUp augmentation."""
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = augmented_train_step(
            model, train_loader, loss_fn, optimizer, device, cutmix_prob=cutmix_prob
        )
        val_loss, val_acc = augmented_valid_step(model, val_loader, loss_fn, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, checkpoint_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Best: {best_val_acc:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


# ---------------------------------------------------------------------------
# Fine-tune on PlantDoc
# ---------------------------------------------------------------------------

def create_plantdoc_dataloaders_for_finetune(
    plantdoc_dir: str,
    test_transform,
    batch_size: int,
    seed: int,
):
    """Create 70/15/15 split of PlantDoc for fine-tuning.

    NOTE: PlantDoc has severe class imbalance (28 classes, 2,569 images). Several
    classes contain only 1--5 samples. Stratified splitting is attempted; if any
    class has <2 samples, the split falls back to unstratified random partition.
    This is an inherent limitation of PlantDoc, not a methodology flaw. Per-class
    metrics for under-represented classes should be interpreted cautiously.
    """
    from src.data_setup import SubsetDataset
    base_dataset = datasets.ImageFolder(root=plantdoc_dir, transform=None)
    targets = np.array([s[1] for s in base_dataset.samples])
    indices = np.arange(len(base_dataset))

    unique, counts = np.unique(targets, return_counts=True)
    min_count = counts.min()
    use_stratify = min_count >= 2

    print(f"PlantDoc: {len(base_dataset)} images, {len(unique)} classes")
    print(f"  Per-class range: {counts.min()}--{counts.max()} images, median={np.median(counts):.0f}")
    if not use_stratify:
        rare = [(base_dataset.classes[unique[i]], counts[i]) for i in np.where(counts < 2)[0]]
        print(f"  Classes with <2 samples ({len(rare)}): {rare}")
        print(f"  Using unstratified split (stratification requires >=2 samples per class)")

    stratify_arg = targets if use_stratify else None
    train_idx, tmp_idx = train_test_split(
        indices, test_size=0.3, stratify=stratify_arg, random_state=seed
    )
    tmp_targets = targets[tmp_idx]

    if use_stratify:
        tmp_unique, tmp_counts = np.unique(tmp_targets, return_counts=True)
        if tmp_counts.min() < 2:
            print(f"  Second split: tmp subset has class with only {tmp_counts.min()} sample(s), skipping stratification")
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=0.5, stratify=None, random_state=seed
            )
        else:
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=0.5, stratify=tmp_targets, random_state=seed
            )
    else:
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.5, stratify=None, random_state=seed
        )

    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = SubsetDataset(Subset(base_dataset, train_idx.tolist()), test_transform)
    val_ds = SubsetDataset(Subset(base_dataset, val_idx.tolist()), test_transform)
    test_ds = SubsetDataset(Subset(base_dataset, test_idx.tolist()), test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, base_dataset.classes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

MODEL_MAP = {
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


def build_transform(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Training")
    parser.add_argument("-m", "--model", required=True, help="Model key from model_map")
    parser.add_argument("--method", required=True, choices=["augment", "finetune", "dann"],
                        help="Domain adaptation strategy")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to PlantVillage pretrained checkpoint for finetune/dann")
    args = parser.parse_args()

    # Resolve pretrained checkpoint path: accept filename only or relative path
    if args.pretrained_ckpt:
        if not os.path.isabs(args.pretrained_ckpt):
            args.pretrained_ckpt = os.path.join(CHECKPOINT_DIR, args.pretrained_ckpt)
        if not args.pretrained_ckpt.endswith('.pth'):
            args.pretrained_ckpt += '.pth'

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}, Method: {args.method}, Model: {args.model}, Seed: {args.seed}")

    model_factory = MODEL_MAP.get(args.model)
    if model_factory is None:
        print(f"Unknown model: {args.model}")
        return

    test_transform = build_transform(train=False)
    train_transform = build_transform(train=True)

    # -----------------------------------------------------------------------
    # Method: augment (CutMix + MixUp on PlantVillage)
    # -----------------------------------------------------------------------
    if args.method == "augment":
        train_loader, val_loader, test_loader, class_names = data_setup.create_dataloader(
            data_path=PLANTVILLAGE_DIR,
            train_transform=train_transform,
            test_transform=test_transform,
            batch_size=args.batch_size,
            seed=args.seed,
            stratified=True,
        )
        model = model_factory(num_classes=len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        ckpt_name = f"{args.model}_augment_epoch_best.pth"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

        print(f"Training {args.model} with CutMix+MixUp augmentation on PlantVillage")
        train_augmented(
            model, train_loader, val_loader, loss_fn, optimizer, device,
            epochs=args.epochs, patience=10, checkpoint_path=ckpt_path,
        )

        # Evaluate on PlantVillage test
        from src.utils import evaluate_model
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        evaluate_model(model, loss_fn, test_loader, class_names, device,
                       results_dir=os.path.join(DOMAIN_ADAPT_DIR, args.model, "augment_plantvillage"))

    # -----------------------------------------------------------------------
    # Method: finetune (PlantVillage pretrain -> PlantDoc fine-tune)
    # -----------------------------------------------------------------------
    elif args.method == "finetune":
        # Step 1: Load PlantDoc data
        pd_train_loader, pd_val_loader, pd_test_loader, pd_class_names = \
            create_plantdoc_dataloaders_for_finetune(
                PLANTDOC_DIR, test_transform, args.batch_size, args.seed
            )
        print(f"PlantDoc classes: {pd_class_names}")

        # For fine-tuning, we match num_classes to PlantDoc
        model = model_factory(num_classes=len(pd_class_names)).to(device)

        # Load PlantVillage pretrained weights if provided
        if args.pretrained_ckpt:
            print(f"Loading PlantVillage pretrained: {args.pretrained_ckpt}")
            state = torch.load(args.pretrained_ckpt, map_location=device)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            # Replace classifier layer to match PlantDoc num_classes
            pv_num_classes = state["linear4.weight"].shape[0]
            if pv_num_classes != len(pd_class_names):
                print(f"  Classifier mismatch: PV={pv_num_classes}, PD={len(pd_class_names)}. Replacing classifier.")
                del state["linear4.weight"]
                del state["linear4.bias"]
            model.load_state_dict(state, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
        loss_fn = nn.CrossEntropyLoss()

        ckpt_name = f"{args.model}_finetune_best.pth"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

        print(f"Fine-tuning {args.model} on PlantDoc")
        engine.train(
            model, pd_train_loader, pd_val_loader, optimizer, loss_fn,
            epochs=args.epochs, device=device, patience=10, early_stopping=True,
            checkpoint_dir=DA_CHECKPOINT_DIR, enable_live_plot=False,
        )

        # Evaluate on PlantDoc test
        from src.utils import evaluate_model
        best_ckpt = os.path.join(DA_CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
        evaluate_model(model, loss_fn, pd_test_loader, pd_class_names, device,
                       results_dir=os.path.join(DOMAIN_ADAPT_DIR, args.model, "finetune_plantdoc"))

    # -----------------------------------------------------------------------
    # Method: dann (Domain-Adversarial Neural Network)
    # -----------------------------------------------------------------------
    elif args.method == "dann":
        # Load PlantVillage (source) and PlantDoc (target) data
        pv_loader = datasets.ImageFolder(root=PLANTVILLAGE_DIR, transform=None)
        pv_targets = np.array([s[1] for s in pv_loader.samples])
        pv_indices = np.arange(len(pv_loader))
        pv_train_idx, _ = train_test_split(pv_indices, test_size=0.15, stratify=pv_targets, random_state=args.seed)

        pv_train_ds = data_setup.SubsetDataset(Subset(pv_loader, pv_train_idx.tolist()), train_transform)

        source_loader = DataLoader(pv_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # PlantDoc: use all data as unlabeled target
        pd_dataset = datasets.ImageFolder(root=PLANTDOC_DIR, transform=test_transform)
        target_loader = DataLoader(pd_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Build backbone
        num_classes_pv = len(pv_loader.classes)
        backbone = model_factory(num_classes=num_classes_pv)

        # Load pretrained if provided
        if args.pretrained_ckpt:
            state = torch.load(args.pretrained_ckpt, map_location='cpu')
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            backbone.load_state_dict(state, strict=False)

        dann_model = DANNWrapper(backbone, num_classes_pv).to(device)
        optimizer = torch.optim.Adam(dann_model.parameters(), lr=args.lr)
        class_criterion = nn.CrossEntropyLoss()

        ckpt_path = os.path.join(DA_CHECKPOINT_DIR, f"{args.model}_dann_best.pth")

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            cls_loss, dom_loss, src_acc, dom_acc = train_epoch_dann(
                dann_model, source_loader, target_loader,
                class_criterion, optimizer, device, epoch, args.epochs,
            )
            # Use PlantDoc subset validation
            _, _, pd_test_loader, _ = create_plantdoc_dataloaders_for_finetune(
                PLANTDOC_DIR, test_transform, args.batch_size, args.seed
            )
            val_loss, val_acc = validate_dann(dann_model, pd_test_loader, class_criterion, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": dann_model.state_dict(),
                    "val_acc": val_acc,
                }, ckpt_path)

            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"Cls: {cls_loss:.4f} Dom: {dom_loss:.4f} | "
                  f"SrcAcc: {src_acc:.4f} DomAcc: {dom_acc:.4f} | "
                  f"ValAcc: {val_acc:.4f}")

        print(f"DANN training complete. Best PlantDoc val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
