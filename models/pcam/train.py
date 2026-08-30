from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.data import create_loaders, discover_pcam_files
from src.metrics import classification_metrics
from src.model import create_resnet50, save_checkpoint
from src.utils import resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on PCam.")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--smoke-test", action="store_true", help="Use small subsets for setup testing.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    description: str,
) -> tuple[float, list[float], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    sample_count = 0
    labels_all: list[float] = []
    probabilities_all: list[float] = []
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and device.type == "cuda"
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.detach() * labels.size(0)
        sample_count += labels.size(0)
        if not training:
            labels_all.extend(labels.cpu().numpy().reshape(-1).tolist())
            probabilities_all.extend(
                torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist()
            )
    return float((total_loss / max(sample_count, 1)).item()), labels_all, probabilities_all


def save_history_plot(history: list[dict[str, float]], output_path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_auc"] for item in history], label="AUC")
    axes[1].plot(
        epochs, [item["validation_accuracy"] for item in history], label="Accuracy"
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project = config.section("project")
    data_config = config.section("data")
    training = config.section("training")
    model_config = config.section("model")
    set_seed(int(project["seed"]))
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = config.path("data", "root")
    files = discover_pcam_files(data_root)
    limits = {
        "train_limit": data_config.get("train_limit"),
        "validation_limit": data_config.get("validation_limit"),
        "test_limit": data_config.get("test_limit"),
    }
    if args.smoke_test:
        limits = {"train_limit": 512, "validation_limit": 256, "test_limit": 256}
        print("Smoke-test mode: using small data subsets.")
    train_loader, validation_loader, _ = create_loaders(
        files=files,
        image_size=int(data_config["image_size"]),
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        **limits,
    )

    model = create_resnet50(pretrained=bool(model_config["pretrained"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training["use_amp"])
    )
    best_path = config.path("paths", "checkpoint")
    last_path = config.path("paths", "last_checkpoint")
    start_epoch = 0
    best_auc = 0.0
    history: list[dict[str, float]] = []

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_auc", 0.0))
        history = checkpoint.get("history", [])
        print(f"Resuming from epoch {start_epoch + 1}.")

    no_improvement = 0
    epochs = int(training["epochs"])
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            f"Epoch {epoch + 1}/{epochs} - training",
        )
        validation_loss, labels, probabilities = run_epoch(
            model, validation_loader, criterion, device, None, None,
            f"Epoch {epoch + 1}/{epochs} - validation",
        )
        metrics = classification_metrics(
            labels, probabilities, float(model_config["decision_threshold"])
        )
        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_auc": float(metrics["auc"]),
            "duration_seconds": float(time.perf_counter() - started),
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}, val_accuracy={metrics['accuracy']:.4f}, "
            f"val_auc={metrics['auc']:.4f}"
        )
        current_auc = float(metrics["auc"])
        if current_auc > best_auc:
            best_auc = current_auc
            no_improvement = 0
            save_checkpoint(
                best_path, model, optimizer, epoch, best_auc, history, config.raw
            )
            print(f"Saved improved model: {best_path}")
        else:
            no_improvement += 1
        save_checkpoint(last_path, model, optimizer, epoch, best_auc, history, config.raw)
        results_dir = config.path("paths", "results_dir")
        save_json(
            {"history": history, "best_auc": best_auc},
            results_dir / "training_history.json",
        )
        save_history_plot(history, results_dir / "training_history.png")
        if no_improvement >= int(training["patience"]):
            print("Early stopping: validation AUC did not improve.")
            break
    print(f"Training finished. Best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
