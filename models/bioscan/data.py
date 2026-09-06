from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_classes(path: Path) -> dict[str, int]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("BIOSCAN classes file must contain a name-to-index mapping")
    classes = {str(name): int(index) for name, index in mapping.items()}
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("BIOSCAN class indices must be contiguous from zero")
    return classes


class BioscanDataset(Dataset):
    def __init__(
        self,
        metadata: Path,
        image_dir: Path,
        split: str,
        classes: dict[str, int],
        image_size: int,
        train: bool = False,
    ) -> None:
        table = pq.read_table(metadata, columns=["processid", "split", "species"])
        table = table.filter(pc.equal(table["split"], split))
        rows = zip(table["processid"].to_pylist(), table["species"].to_pylist())
        self.samples = [
            (image_dir / split / f"{process_id}.jpg", classes[label])
            for process_id, label in rows
            if label in classes
        ]
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        resize = (
            [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
            if train
            else [transforms.Resize(image_size + 16), transforms.CenterCrop(image_size)]
        )
        self.transform = transforms.Compose(resize + [transforms.ToTensor(), normalize])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def create_loaders(config, include_train: bool = True):
    data = config.section("data")
    training = config.section("training")
    metadata = config.path("data", "metadata")
    image_dir = config.path("data", "image_dir")
    classes = load_classes(config.path("data", "classes"))
    image_size = int(data["image_size"])
    workers = int(training["num_workers"])
    batch_size = int(training["batch_size"])
    options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 4
    train_loader = None
    if include_train:
        train_set = BioscanDataset(
            metadata, image_dir, "train", classes, image_size, train=True
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True, **options
        )
    validation_set = BioscanDataset(
        metadata, image_dir, "val", classes, image_size, train=False
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size * 2, shuffle=False, **options
    )
    return train_loader, validation_loader, classes
