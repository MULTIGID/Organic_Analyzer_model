from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.metrics import multiclass_metrics_from_predictions
from src.model import create_resnet50, load_checkpoint
from src.utils import resolve_device, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, limit=None):
    training = optimizer is not None
    model.train(training)
    loss_total = torch.zeros((), device=device)
    count = 0
    labels_all, predictions_all = [], []
    for batch_index, (images, labels) in enumerate(tqdm(loader, leave=False)):
        if limit is not None and batch_index >= limit:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
        loss_total += loss.detach() * labels.size(0)
        count += labels.size(0)
        if not training:
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    return float((loss_total / max(count, 1)).item()), labels_all, predictions_all


def save_checkpoint(path, model, optimizer, epoch, best_accuracy, history, config, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "best_accuracy": best_accuracy, "history": history,
        "config": config, "num_classes": len(classes), "class_names": classes,
    }, path)


def artifact_paths(config, smoke_test: bool) -> tuple[Path, Path, Path]:
    best_path = config.path("paths", "checkpoint")
    last_path = config.path("paths", "last_checkpoint")
    results_path = config.path("paths", "results_dir")
    if smoke_test:
        best_path = best_path.parent / "smoke" / best_path.name
        last_path = last_path.parent / "smoke" / last_path.name
        results_path = results_path / "smoke"
    return best_path, last_path, results_path


def save_plot(history, path):
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history])
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def train_multiclass(config_path: Path, loader_factory, device_name, resume, smoke_test):
    config = load_config(config_path)
    project, data = config.section("project"), config.section("data")
    training, model_config = config.section("training"), config.section("model")
    set_seed(int(project["seed"]))
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    loaders = loader_factory(config)
    train_loader, validation_loader, _, classes = loaders
    expected = int(model_config["num_classes"])
    if len(classes) != expected:
        raise ValueError(f"Expected {expected} classes, found {len(classes)}")
    print(f"{project['name']}: {len(train_loader.dataset)} train, "
          f"{len(validation_loader.dataset)} validation, {len(classes)} classes")
    model = create_resnet50(bool(model_config["pretrained"]), len(classes)).to(device)
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]),
                      weight_decay=float(training["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training["use_amp"])
    )
    best_path, last_path, results = artifact_paths(config, smoke_test)
    start_epoch, best_accuracy, history = 0, -1.0, []
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", -1))
        history = checkpoint.get("history", [])
    epochs = 1 if smoke_test else int(training["epochs"])
    no_improvement = 0
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        limit = 2 if smoke_test else None
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, limit
        )
        validation_loss, labels, predictions = run_epoch(
            model, validation_loader, criterion, device, limit=limit
        )
        metrics = multiclass_metrics_from_predictions(labels, predictions, len(classes))
        accuracy = float(metrics["accuracy"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": accuracy,
                        "validation_f1_macro": metrics["f1_macro"],
                        "duration_seconds": time.perf_counter() - started})
        if accuracy > best_accuracy:
            best_accuracy, no_improvement = accuracy, 0
            save_checkpoint(best_path, model, optimizer, epoch, best_accuracy,
                            history, config.raw, classes)
        else:
            no_improvement += 1
        save_checkpoint(last_path, model, optimizer, epoch, best_accuracy,
                        history, config.raw, classes)
        save_json({"history": history, "best_accuracy": best_accuracy,
                   "class_names": classes}, results / "training_history.json")
        save_plot(history, results / "training_history.png")
        print(f"Epoch {epoch + 1}: val_accuracy={accuracy:.4f}")
        if no_improvement >= int(training["patience"]):
            print("Early stopping: validation accuracy did not improve.")
            break


def evaluate_multiclass(config_path: Path, loader_factory, device_name):
    config = load_config(config_path)
    data = config.section("data")
    training = config.section("training")
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model, checkpoint = load_checkpoint(config.path("paths", "checkpoint"), device)
    model.eval()
    _, _, loader, classes = loader_factory(config)
    labels, predictions = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Testing"):
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
    if metrics["confusion_matrix"]:
        plt.figure(figsize=(max(8, len(classes) * 0.5), max(7, len(classes) * 0.45)))
        sns.heatmap(metrics["confusion_matrix"], cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=90)
        plt.tight_layout()
        results.mkdir(parents=True, exist_ok=True)
        plt.savefig(results / "confusion_matrix.png", dpi=180)
        plt.close()
