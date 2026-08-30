from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .data import build_transforms


EXPECTED_PBC_CLASSES = (
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
)


def discover_pbc_splits(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"PBC data directory not found: {root_path}")
    splits: dict[str, Path] = {}
    for key, candidates in {
        "train": ("Train", "train"),
        "validation": ("Val", "val", "Validation", "validation"),
        "test": ("Test", "test"),
    }.items():
        path = next((root_path / name for name in candidates if (root_path / name).is_dir()), None)
        if path is None:
            raise FileNotFoundError(f"PBC {key} split was not found under {root_path}")
        classes = tuple(sorted(item.name for item in path.iterdir() if item.is_dir()))
        if classes != EXPECTED_PBC_CLASSES:
            raise ValueError(f"Unexpected PBC classes in {path}: {classes}")
        splits[key] = path
    return splits


def create_pbc_loaders(
    root: str | Path, image_size: int, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    splits = discover_pbc_splits(root)
    train_transform, evaluation_transform = build_transforms(image_size)
    datasets = (
        ImageFolder(splits["train"], transform=train_transform),
        ImageFolder(splits["validation"], transform=evaluation_transform),
        ImageFolder(splits["test"], transform=evaluation_transform),
    )
    class_names = datasets[0].classes
    if any(dataset.classes != class_names for dataset in datasets[1:]):
        raise ValueError("PBC class order differs between dataset splits")
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 4
    return (
        DataLoader(datasets[0], shuffle=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
        class_names,
    )
