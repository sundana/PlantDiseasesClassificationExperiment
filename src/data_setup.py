import os
from typing import Optional, List, Tuple
import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from sklearn.model_selection import StratifiedKFold

NUM_WORKERS = 4

from src.config import DATA_DIR, DEFAULT_SEED


class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def create_dataloader(
    data_path: str = DATA_DIR,
    train_transform: transforms.Compose = None,
    test_transform: transforms.Compose = None,
    batch_size: int = 64,
    num_workers: Optional[int] = NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    stratified: bool = False,
    exclude_classes: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    base_dataset = datasets.ImageFolder(root=data_path, transform=None)
    if exclude_classes:
        # Filter before the split so that every condition uses the same
        # stratified sampling logic on the same remaining images.
        exclude = set(exclude_classes)
        unknown = exclude - set(base_dataset.classes)
        if unknown:
            print(f"[data_setup] exclude_classes not present, ignoring: {sorted(unknown)}")
        old_classes = list(base_dataset.classes)
        keep_classes = [c for c in old_classes if c not in exclude]
        remap = {old_classes.index(c): new for new, c in enumerate(keep_classes)}
        kept_samples = [
            (path, remap[target])
            for path, target in base_dataset.samples
            if target in remap
        ]
        if not kept_samples:
            raise ValueError("exclude_classes removed every sample")
        base_dataset.classes = keep_classes
        base_dataset.class_to_idx = {c: i for i, c in enumerate(keep_classes)}
        base_dataset.samples = kept_samples
        base_dataset.targets = [target for _, target in kept_samples]
        try:  # torchvision keeps ``imgs`` as an alias in some versions
            base_dataset.imgs = base_dataset.samples
        except Exception:
            pass
        print(
            f"[data_setup] excluded {sorted(exclude & set(old_classes))}; "
            f"{len(keep_classes)} classes, {len(kept_samples)} images remain"
        )
    class_names = base_dataset.classes
    targets = np.array([s[1] for s in base_dataset.samples])
    n = len(base_dataset)

    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    if stratified:
        from sklearn.model_selection import train_test_split
        indices = np.arange(n)
        train_idx, tmp_idx = train_test_split(
            indices, test_size=1 - 0.7, stratify=targets,
            random_state=seed
        )
        tmp_targets = targets[tmp_idx]
        val_ratio = 0.15 / 0.30  # val gets half of the 30% remaining
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=1 - val_ratio, stratify=tmp_targets,
            random_state=seed
        )
        train_subset = Subset(base_dataset, train_idx.tolist())
        val_subset = Subset(base_dataset, val_idx.tolist())
        test_subset = Subset(base_dataset, test_idx.tolist())
    else:
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset, test_subset = random_split(
            base_dataset, [train_size, val_size, test_size],
            generator=generator
        )

    train_dataset = SubsetDataset(train_subset, train_transform)
    val_dataset = SubsetDataset(val_subset, test_transform)
    test_dataset = SubsetDataset(test_subset, test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers if num_workers else 0, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers if num_workers else 0, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers if num_workers else 0, pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names


def create_stratified_kfold_dataloaders(
    data_path: str = DATA_DIR,
    train_transform: transforms.Compose = None,
    test_transform: transforms.Compose = None,
    batch_size: int = 64,
    n_splits: int = 5,
    num_workers: Optional[int] = NUM_WORKERS,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[DataLoader, DataLoader, DataLoader, List[str], int]]:
    """Create dataloaders for stratified k-fold cross-validation.

    Returns a list of tuples, one per fold:
    (train_loader, val_loader, test_loader, class_names, fold_id)
    where val_loader uses the held-out fold as validation.
    The test set is a fixed 15% hold-out, independent of the folds.
    """
    base_dataset = datasets.ImageFolder(root=data_path, transform=None)
    class_names = base_dataset.classes
    targets = np.array([s[1] for s in base_dataset.samples])
    indices = np.arange(len(base_dataset))
    n = len(base_dataset)

    # Hold out 15% as an independent test set (same across all folds)
    from sklearn.model_selection import train_test_split
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=targets, random_state=seed
    )
    trainval_targets = targets[trainval_idx]

    test_subset = Subset(base_dataset, test_idx.tolist())
    test_dataset = SubsetDataset(test_subset, test_transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers if num_workers else 0, pin_memory=True
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_loaders = []

    for fold_id, (train_fold_idx, val_fold_idx) in enumerate(
        skf.split(trainval_idx, trainval_targets)
    ):
        train_abs = trainval_idx[train_fold_idx]
        val_abs = trainval_idx[val_fold_idx]

        train_subset = Subset(base_dataset, train_abs.tolist())
        val_subset = Subset(base_dataset, val_abs.tolist())

        train_dataset = SubsetDataset(train_subset, train_transform)
        val_dataset = SubsetDataset(val_subset, test_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers if num_workers else 0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers if num_workers else 0, pin_memory=True
        )

        fold_loaders.append(
            (train_loader, val_loader, test_dataloader, class_names, fold_id)
        )

    return fold_loaders


def create_plantdoc_dataloaders(
    plantdoc_dir: str,
    test_transform: transforms.Compose = None,
    batch_size: int = 64,
    num_workers: Optional[int] = NUM_WORKERS,
    seed: int = DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/val/test dataloaders from PlantDoc for fine-tuning."""
    base_dataset = datasets.ImageFolder(root=plantdoc_dir, transform=None)
    class_names = base_dataset.classes
    targets = np.array([s[1] for s in base_dataset.samples])

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(base_dataset))
    train_idx, tmp_idx = train_test_split(
        indices, test_size=0.3, stratify=targets, random_state=seed
    )
    tmp_targets = targets[tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.5, stratify=tmp_targets, random_state=seed
    )

    train_subset = SubsetDataset(Subset(base_dataset, train_idx.tolist()), test_transform)
    val_subset = SubsetDataset(Subset(base_dataset, val_idx.tolist()), test_transform)
    test_subset = SubsetDataset(Subset(base_dataset, test_idx.tolist()), test_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers if num_workers else 0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers if num_workers else 0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers if num_workers else 0, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    create_dataloader(
        data_path='data',
        batch_size=64,
        num_workers=2,
        train_transform=train_transform,
        test_transform=test_transform,
    )
