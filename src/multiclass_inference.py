from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from PIL import Image

from .gradcam import render_gradcam_overlay
from .model import load_checkpoint
from .transforms import build_transforms


@dataclass(frozen=True)
class MulticlassPrediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class MulticlassPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        image_size: int,
        class_names_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.model, metadata = load_checkpoint(checkpoint_path, device)
        self.model.eval()
        class_names = metadata.get("class_names")
        if class_names is None and class_names_path is not None:
            mapping = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                raise ValueError("Class names file must contain a name-to-index mapping")
            class_names = [None] * len(mapping)
            for name, index in mapping.items():
                if not isinstance(index, int) or not 0 <= index < len(mapping):
                    raise ValueError("Class indices must be contiguous integers from zero")
                if class_names[index] is not None:
                    raise ValueError(f"Duplicate class index: {index}")
                class_names[index] = str(name)
            if any(name is None for name in class_names):
                raise ValueError("Class indices must be contiguous integers from zero")
        if class_names is None:
            raise KeyError("Checkpoint does not contain class_names")
        if len(class_names) != self.model.fc.out_features:
            raise ValueError("Class names count does not match checkpoint output size")
        self.class_names = list(class_names)
        self.class_indices = {name: index for index, name in enumerate(self.class_names)}
        _, self.transform = build_transforms(image_size)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> MulticlassPrediction:
        tensor = self.preprocess(image)
        with torch.inference_mode():
            values = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().tolist()
        index = max(range(len(values)), key=values.__getitem__)
        return MulticlassPrediction(
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
