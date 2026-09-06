from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from src.config import load_config
from src.utils import set_seed

from .data import create_loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the image-only BIOSCAN-5M insect species classifier."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for BIOSCAN training")
    config = load_config(Path(__file__).with_name("config.yaml"))
    project = config.section("project")
    training = config.section("training")
    set_seed(int(project["seed"]))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_loader, validation_loader, classes = create_loaders(config)
    expected_classes = int(config.section("model")["num_classes"])
    if len(classes) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(classes)}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.cuda().to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["use_amp"]))
    best_path = config.path("paths", "checkpoint")
    last_path = config.path("paths", "last_checkpoint")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state")))
        optimizer.load_state_dict(
            checkpoint.get("optimizer", checkpoint.get("optimizer_state"))
        )
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))

    epochs = args.epochs or int(training["epochs"])
    accumulation = int(training["accumulation"])
    print(
        f"{project['name']}: {len(train_loader.dataset):,} train, "
        f"{len(validation_loader.dataset):,} validation, {len(classes):,} classes"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(training["use_amp"])):
                loss = criterion(model(images), targets) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={loss.item() * accumulation:.4f} "
                    f"it/s={step / elapsed:.2f}",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for val_step, (images, targets) in enumerate(validation_loader, start=1):
                images = images.cuda(non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                targets = targets.cuda(non_blocking=True)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=bool(training["use_amp"])
                ):
                    predictions = model(images).argmax(1)
                correct += (predictions == targets).sum().item()
                total += targets.numel()
                if args.max_val_steps and val_step >= args.max_val_steps:
                    break
        accuracy = correct / max(total, 1)
        is_best = accuracy >= best_accuracy
        best_accuracy = max(best_accuracy, accuracy)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_accuracy": best_accuracy,
            "num_classes": len(classes),
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
        print(
            f"Epoch {epoch + 1}: val_accuracy={accuracy:.4%} "
            f"best={best_accuracy:.4%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
