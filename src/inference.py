from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .data import build_transforms
from .model import load_checkpoint


@dataclass(frozen=True)
class Prediction:
    probability: float
    predicted_class: int
    label: str
    confidence: float


def validate_input_image(image: Image.Image) -> list[str]:
    warnings: list[str] = []
    width, height = image.size
    if width < 32 or height < 32:
        warnings.append("The image is too small for reliable analysis.")
    ratio = width / max(height, 1)
    if ratio < 0.6 or ratio > 1.67:
        warnings.append("PCam patches are square; this image has an unusual aspect ratio.")
    rgb = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32)
    if rgb.std() < 8:
        warnings.append("The image has very low visual variation.")
    saturation_proxy = np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))
    if saturation_proxy < 6:
        warnings.append("The image has unusually low color variation for an H&E patch.")
    return warnings


class PCamPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        image_size: int = 224,
        threshold: float = 0.5,
        positive_label: str = "Metastatic tissue detected",
        negative_label: str = "No metastatic tissue detected",
    ) -> None:
        self.device = device
        self.model, self.metadata = load_checkpoint(checkpoint_path, device)
        self.model.eval()
        _, self.transform = build_transforms(image_size)
        self.threshold = threshold
        self.positive_label = positive_label
        self.negative_label = negative_label

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self.preprocess(image)
        with torch.inference_mode():
            probability = float(torch.sigmoid(self.model(tensor)).item())
        predicted_class = int(probability >= self.threshold)
        label = self.positive_label if predicted_class else self.negative_label
        confidence = probability if predicted_class else 1.0 - probability
        return Prediction(probability, predicted_class, label, confidence)

    def grad_cam(self, image: Image.Image) -> Image.Image:
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        target_layer: nn.Module = self.model.layer4[-1]

        def forward_hook(_module, _inputs, output):
            activations.append(output.detach())
            output.register_hook(lambda gradient: gradients.append(gradient.detach()))

        forward_handle = target_layer.register_forward_hook(forward_hook)
        try:
            tensor = self.preprocess(image)
            self.model.zero_grad(set_to_none=True)
            logit = self.model(tensor).squeeze()
            logit.backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze()
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-8)
            cam_array = (cam.cpu().numpy() * 255).astype(np.uint8)
        finally:
            forward_handle.remove()

        base = image.convert("RGB")
        heat = Image.fromarray(cam_array).resize(base.size, Image.Resampling.BILINEAR)
        heat_values = np.asarray(heat, dtype=np.float32) / 255.0
        color = np.zeros((*heat_values.shape, 3), dtype=np.uint8)
        color[..., 0] = (255 * heat_values).astype(np.uint8)
        color[..., 1] = (100 * np.sqrt(heat_values)).astype(np.uint8)
        overlay = Image.fromarray(color, mode="RGB")
        return Image.blend(base, overlay, alpha=0.42)
