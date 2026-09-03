from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FileListDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[tuple[Path, int]], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


def class_directories(root: str | Path) -> tuple[Path, list[str]]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")
    classes = sorted(path.name for path in root_path.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected at least two class directories under {root_path}")
    return root_path, classes


def collect_samples(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    samples = []
    for label, class_name in enumerate(classes):
        samples.extend(
            (path, label) for path in sorted((root / class_name).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not samples:
        raise ValueError(f"No supported images found under {root}")
    return samples


def hash_split(samples: list[tuple[Path, int]], validation_percent=15, test_percent=15):
    train, validation, test = [], [], []
    for sample in samples:
        stable_name = f"{sample[0].parent.name}/{sample[0].name}"
        bucket = int(hashlib.sha256(stable_name.encode()).hexdigest()[:8], 16) % 100
        if bucket < test_percent:
            test.append(sample)
        elif bucket < test_percent + validation_percent:
            validation.append(sample)
        else:
            train.append(sample)
    if not train or not validation or (test_percent > 0 and not test):
        raise ValueError("A deterministic dataset split is empty")
    return train, validation, test


def split_train_with_external_test(train_root: str | Path, test_root: str | Path):
    train_path, classes = class_directories(train_root)
    test_path, test_classes = class_directories(test_root)
    if test_classes != classes:
        raise ValueError("Train and test class names differ")
    source = collect_samples(train_path, classes)
    train, validation, _ = hash_split(source, validation_percent=10, test_percent=0)
    return train, validation, collect_samples(test_path, classes), classes


def create_loaders_from_samples(samples, classes, image_size, batch_size, num_workers):
    train_transform, evaluation_transform = build_transforms(image_size)
    datasets = (
        FileListDataset(samples[0], train_transform),
        FileListDataset(samples[1], evaluation_transform),
        FileListDataset(samples[2], evaluation_transform),
    )
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    if num_workers > 0:
        common["prefetch_factor"] = 4
    return (DataLoader(datasets[0], shuffle=True, **common),
            DataLoader(datasets[1], shuffle=False, **common),
            DataLoader(datasets[2], shuffle=False, **common), classes)


def create_single_root_loaders(root, image_size, batch_size, num_workers):
    root_path, classes = class_directories(root)
    samples = hash_split(collect_samples(root_path, classes))
    return create_loaders_from_samples(samples, classes, image_size, batch_size, num_workers)


def create_external_test_loaders(
    train_root, test_root, image_size, batch_size, num_workers
):
    train, validation, test, classes = split_train_with_external_test(train_root, test_root)
    return create_loaders_from_samples(
        (train, validation, test), classes, image_size, batch_size, num_workers
    )
