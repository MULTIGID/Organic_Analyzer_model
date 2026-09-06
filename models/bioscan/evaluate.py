from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import load_config
from src.model import load_checkpoint
from src.utils import resolve_device, save_json

from .data import create_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BIOSCAN-5M model.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(Path(__file__).with_name("config.yaml"))
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(config.path("paths", "checkpoint"), device)
    model.eval()
    _, validation_loader, classes = create_loaders(config, include_train=False)
    correct = total = 0
    with torch.inference_mode():
        for images, targets in tqdm(validation_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(images).argmax(1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
    metrics = {
        "validation_accuracy": correct / max(total, 1),
        "images": total,
        "classes": len(classes),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
    }
    save_json(metrics, config.path("paths", "results_dir") / "validation_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
