from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class PCamFiles:
    train_x: Path
    train_y: Path
    validation_x: Path
    validation_y: Path
    test_x: Path
    test_y: Path


def _find_one(root: Path, candidates: tuple[str, ...]) -> Path:
    files = [path for path in root.rglob("*.h5") if path.is_file()]
    lowered = [(path, path.name.lower()) for path in files]
    matches = [path for path, name in lowered if any(token in name for token in candidates)]
    if not matches:
        raise FileNotFoundError(
            f"Could not find an HDF5 file matching {candidates} under {root}. "
            "See README.md for the expected PCam files."
        )
    matches.sort(key=lambda path: (len(path.parts), len(path.name)))
    return matches[0]


def discover_pcam_files(root: str | Path) -> PCamFiles:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"PCam data directory not found: {root_path}")
    return PCamFiles(
        train_x=_find_one(root_path, ("training_split", "train_x")),
        train_y=_find_one(root_path, ("train_y", "training_split_y")),
        validation_x=_find_one(root_path, ("validation_split", "valid_x", "val_x")),
        validation_y=_find_one(root_path, ("valid_y", "validation_y", "val_y")),
        test_x=_find_one(root_path, ("test_split", "test_x")),
        test_y=_find_one(root_path, ("test_y",)),
    )


def _first_dataset_key(handle: h5py.File, preferred: str) -> str:
    if preferred in handle:
        return preferred
    keys = list(handle.keys())
    if len(keys) != 1:
        raise KeyError(f"Expected key '{preferred}' or one dataset, found: {keys}")
    return keys[0]


class PCamH5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        transform: Callable | None = None,
        limit: int | None = None,
    ) -> None:
        self.image_path = str(Path(image_path).resolve())
        self.label_path = str(Path(label_path).resolve())
        self.transform = transform
        self._images: h5py.File | None = None
        self._labels: h5py.File | None = None
        with h5py.File(self.image_path, "r") as image_file:
            self.image_key = _first_dataset_key(image_file, "x")
            image_count = len(image_file[self.image_key])
        with h5py.File(self.label_path, "r") as label_file:
            self.label_key = _first_dataset_key(label_file, "y")
            label_count = len(label_file[self.label_key])
        if image_count != label_count:
            raise ValueError(f"Image/label count mismatch: {image_count} != {label_count}")
        self.length = min(image_count, limit) if limit else image_count

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._images is None:
            self._images = h5py.File(self.image_path, "r")
            self._labels = h5py.File(self.label_path, "r")
        image_array = np.asarray(self._images[self.image_key][index], dtype=np.uint8)
        label_array = np.asarray(self._labels[self.label_key][index]).reshape(-1)
        image = Image.fromarray(image_array).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(label_array[0]), dtype=torch.float32)
        return image, label

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state

    def close(self) -> None:
        for handle in (self._images, self._labels):
            if handle is not None:
                handle.close()
        self._images = None
        self._labels = None

    def __del__(self) -> None:
        self.close()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return train_transform, evaluation_transform


def create_loaders(
    files: PCamFiles,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, evaluation_transform = build_transforms(image_size)
    datasets = (
        PCamH5Dataset(files.train_x, files.train_y, train_transform, train_limit),
        PCamH5Dataset(
            files.validation_x, files.validation_y, evaluation_transform, validation_limit
        ),
        PCamH5Dataset(files.test_x, files.test_y, evaluation_transform, test_limit),
    )
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
    )
