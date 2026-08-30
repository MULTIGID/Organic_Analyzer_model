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
from src.data import create_loaders, discover_pcam_files
from src.metrics import classification_metrics
from src.model import load_checkpoint
from src.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PCam model.")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config.section("data")
    training = config.section("training")
    model_config = config.section("model")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(config.path("paths", "checkpoint"), device)
    model.eval()
    files = discover_pcam_files(config.path("data", "root"))
    _, _, test_loader = create_loaders(
        files,
        int(data["image_size"]),
        int(training["batch_size"]),
        int(training["num_workers"]),
        data.get("train_limit"),
        data.get("validation_limit"),
        data.get("test_limit"),
    )
    labels_all: list[float] = []
    probabilities_all: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing"):
            logits = model(images.to(device, non_blocking=True))
            probabilities_all.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
            labels_all.extend(labels.numpy().reshape(-1).tolist())
    duration = time.perf_counter() - started
    metrics = classification_metrics(
        labels_all, probabilities_all, float(model_config["decision_threshold"])
    )
    metrics["images"] = len(labels_all)
    metrics["milliseconds_per_image"] = duration * 1000 / max(len(labels_all), 1)
    metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1)) + 1
    results_dir = config.path("paths", "results_dir")
    save_json(metrics, results_dir / "test_metrics.json")
    matrix = metrics["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close()
    print(metrics)
    print(f"Saved evaluation results to: {results_dir}")


if __name__ == "__main__":
    main()
