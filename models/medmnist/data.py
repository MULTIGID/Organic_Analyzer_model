from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import build_transforms


PATHMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth_muscle", "normal_colon_mucosa", "cancer_stroma", "colorectal_adenocarcinoma",
]


class NpzDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels.reshape(-1), transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def create_loaders(config):
    data, training = config.section("data"), config.section("training")
    path = Path(config.path("data", "npz_path"))
    if not path.exists():
        raise FileNotFoundError(f"PathMNIST NPZ file not found: {path}")
    archive = np.load(path)
    train_transform, evaluation_transform = build_transforms(int(data["image_size"]))
    datasets = (
        NpzDataset(archive["train_images"], archive["train_labels"], train_transform),
        NpzDataset(archive["val_images"], archive["val_labels"], evaluation_transform),
        NpzDataset(archive["test_images"], archive["test_labels"], evaluation_transform),
    )
    workers = int(training["num_workers"])
    common = {"batch_size": int(training["batch_size"]), "num_workers": workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": workers > 0}
    if workers > 0:
        common["prefetch_factor"] = 4
    return (DataLoader(datasets[0], shuffle=True, **common),
            DataLoader(datasets[1], shuffle=False, **common),
            DataLoader(datasets[2], shuffle=False, **common), PATHMNIST_CLASSES)
