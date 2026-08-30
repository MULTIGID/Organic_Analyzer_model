from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .gradcam import render_gradcam_overlay
from .model import load_checkpoint


@dataclass(frozen=True)
class PBCPrediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class MulticlassPredictor:
    def __init__(self, checkpoint_path: str | Path, device: torch.device, image_size: int) -> None:
        self.device = device
        self.model, metadata = load_checkpoint(checkpoint_path, device)
        self.model.eval()
        self.class_names = list(metadata["class_names"])
        self.class_indices = {name: index for index, name in enumerate(self.class_names)}
        _, self.transform = build_transforms(image_size)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> PBCPrediction:
        tensor = self.preprocess(image)
        with torch.inference_mode():
            values = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().tolist()
        index = max(range(len(values)), key=values.__getitem__)
        return PBCPrediction(
            self.class_names[index],
            float(values[index]),
            dict(zip(self.class_names, map(float, values), strict=True)),
        )

    def grad_cam(
        self, image: Image.Image, class_name: str, intensity: float = 0.55
    ) -> Image.Image:
        if class_name not in self.class_indices:
            raise ValueError(f"Unknown checkpoint class: {class_name}")
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        target_layer = self.model.layer4[-1].conv3

        def forward_hook(_module, _inputs, output):
            activations.append(output.detach())
            output.register_hook(lambda gradient: gradients.append(gradient.detach()))

        handle = target_layer.register_forward_hook(forward_hook)
        try:
            tensor = self.preprocess(image)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self.model(tensor)
                logits[0, self.class_indices[class_name]].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze()
            cam_array = cam.cpu().numpy()
        finally:
            handle.remove()

        return render_gradcam_overlay(image, cam_array, intensity=intensity)


class PBCPredictor(MulticlassPredictor):
    def __init__(self, checkpoint_path: str | Path, device: torch.device, image_size: int) -> None:
        super().__init__(checkpoint_path, device, image_size)
        if len(self.class_names) != 8:
            raise ValueError("PBC checkpoint must contain exactly 8 class names")
