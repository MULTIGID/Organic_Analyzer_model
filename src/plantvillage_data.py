from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import build_transforms


class PlantVillageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, paths: list[str], classes: list[str], transform: Callable):
        self.root = root
        self.paths = paths
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        relative_path = self.paths[index]
        image = Image.open(self.root / Path(relative_path)).convert("RGB")
        class_name = Path(relative_path).parts[-2]
        return self.transform(image), self.class_to_index[class_name]


def _read_paths(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"PlantVillage split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _leaf_group(relative_path: str, leaf_map: dict[str, list[str]]) -> str:
    filename = Path(relative_path).name
    identifier = filename.rsplit("___", 1)[-1]
    identifier = Path(identifier).stem.lower().strip()
    class_name = Path(relative_path).parts[-2]
    suggestions = leaf_map.get(identifier, [])
    match = next((item for item in suggestions if item.startswith(f"{class_name}:::")), None)
    return match or f"{class_name}:::{identifier}"


def discover_plantvillage(root: str | Path) -> tuple[list[str], list[str], list[str], list[str]]:
    root_path = Path(root).resolve()
    color_root = root_path / "raw" / "color"
    if not color_root.is_dir():
        raise FileNotFoundError(f"PlantVillage color images not found: {color_root}")
    classes = sorted(path.name for path in color_root.iterdir() if path.is_dir())
    if len(classes) != 38:
        raise ValueError(f"Expected 38 PlantVillage classes, found {len(classes)}")
    official_train = _read_paths(root_path / "splits" / "color_train.txt")
    test_paths = _read_paths(root_path / "splits" / "color_test.txt")
    leaf_map = json.loads((root_path / "leaf_grouping" / "leaf-map.json").read_text())
    train_paths, validation_paths = [], []
    for path in official_train:
        group = _leaf_group(path, leaf_map)
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 8
        (validation_paths if bucket == 0 else train_paths).append(path)
    if not train_paths or not validation_paths or not test_paths:
        raise ValueError("PlantVillage split is empty")
    return train_paths, validation_paths, test_paths, classes


def create_plantvillage_loaders(root, image_size, batch_size, num_workers):
    root_path = Path(root).resolve()
    train_paths, validation_paths, test_paths, classes = discover_plantvillage(root_path)
    train_transform, evaluation_transform = build_transforms(image_size)
    datasets = (
        PlantVillageDataset(root_path, train_paths, classes, train_transform),
        PlantVillageDataset(root_path, validation_paths, classes, evaluation_transform),
        PlantVillageDataset(root_path, test_paths, classes, evaluation_transform),
    )
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    if num_workers > 0:
        common["prefetch_factor"] = 4
    return (DataLoader(datasets[0], shuffle=True, **common),
            DataLoader(datasets[1], shuffle=False, **common),
            DataLoader(datasets[2], shuffle=False, **common), classes)
