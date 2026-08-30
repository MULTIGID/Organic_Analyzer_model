from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from models.pbc.train import artifact_paths, run_epoch, save_pbc_checkpoint, save_plot
from src.config import load_config
from src.metrics import multiclass_metrics_from_predictions
from src.model import create_resnet50
from src.plantvillage_data import create_plantvillage_loaders
from src.utils import resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 38-class ResNet-50 on PlantVillage.")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project, data = config.section("project"), config.section("data")
    training, model_config = config.section("training"), config.section("model")
    set_seed(int(project["seed"]))
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, validation_loader, _, classes = create_plantvillage_loaders(
        config.path("data", "root"), int(data["image_size"]),
        int(training["batch_size"]), int(training["num_workers"]),
    )
    print(
        f"PlantVillage: {len(train_loader.dataset)} train, "
        f"{len(validation_loader.dataset)} validation, {len(classes)} classes"
    )
    model = create_resnet50(bool(model_config["pretrained"]), len(classes)).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training["use_amp"])
    )
    best_path, last_path, results = artifact_paths(config, args.smoke_test)
    start_epoch, best_accuracy, history = 0, -1.0, []
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", -1))
        history = checkpoint.get("history", [])
        print(f"Resuming PlantVillage training from epoch {start_epoch + 1}.")
    epochs = 1 if args.smoke_test else int(training["epochs"])
    if start_epoch >= epochs:
        print(f"Training is already complete ({start_epoch}/{epochs} epochs).")
        return
    no_improvement = 0
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        batch_limit = 2 if args.smoke_test else None
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, batch_limit
        )
        validation_loss, labels, predictions = run_epoch(
            model, validation_loader, criterion, device, limit=batch_limit
        )
        metrics = multiclass_metrics_from_predictions(labels, predictions, len(classes))
        row = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_f1_macro": metrics["f1_macro"],
            "duration_seconds": time.perf_counter() - started,
        }
        history.append(row)
        accuracy = float(metrics["accuracy"])
        if accuracy > best_accuracy:
            best_accuracy, no_improvement = accuracy, 0
            save_pbc_checkpoint(
                best_path, model, optimizer, epoch, best_accuracy, history, config.raw, classes
            )
        else:
            no_improvement += 1
        save_pbc_checkpoint(
            last_path, model, optimizer, epoch, best_accuracy, history, config.raw, classes
        )
        save_json(
            {"history": history, "best_accuracy": best_accuracy, "class_names": classes},
            results / "training_history.json",
        )
        save_plot(history, results / "training_history.png")
        print(
            f"Epoch {epoch + 1}: val_accuracy={accuracy:.4f}, "
            f"val_f1_macro={metrics['f1_macro']:.4f}"
        )
        if no_improvement >= int(training["patience"]):
            print("Early stopping: validation accuracy did not improve.")
            break
    print(f"Training finished. Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
