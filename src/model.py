from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def create_resnet50(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_auc: float,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "history": history,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    load_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state")
    if model_state is None:
        model_state = checkpoint.get("model")
    if model_state is None:
        raise KeyError("Checkpoint does not contain model_state or model weights")
    inferred_classes = model_state["fc.weight"].shape[0]
    num_classes = int(checkpoint.get("num_classes", inferred_classes))
    model = create_resnet50(pretrained=False, num_classes=num_classes)
    model.load_state_dict(model_state)
    model.to(device)
    if load_optimizer is not None and "optimizer_state" in checkpoint:
        load_optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, checkpoint
