from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from src.config import load_config
from src.metrics import multiclass_metrics_from_predictions
from src.model import load_checkpoint
from src.pbc_data import create_pbc_loaders
from src.utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 8-class PBC model.")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    data, training = config.section("data"), config.section("training")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model, checkpoint = load_checkpoint(config.path("paths", "checkpoint"), device)
    model.eval()
    _, _, loader, classes = create_pbc_loaders(
        config.path("data", "root"), int(data["image_size"]),
        int(training["batch_size"]), int(training["num_workers"]),
    )
    labels, predictions = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Testing PBC"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = multiclass_metrics_from_predictions(labels, predictions, len(classes))
    metrics.update({"images": len(labels), "milliseconds_per_image":
                    (time.perf_counter() - started) * 1000 / max(len(labels), 1),
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
                    "class_names": classes})
    results = config.path("paths", "results_dir")
    save_json(metrics, results / "test_metrics.json")
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    results.mkdir(parents=True, exist_ok=True)
    plt.savefig(results / "confusion_matrix.png", dpi=180)
    plt.close()
    print(metrics)


if __name__ == "__main__":
    main()
