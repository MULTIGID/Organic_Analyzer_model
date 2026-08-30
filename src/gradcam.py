from __future__ import annotations

import numpy as np
from PIL import Image


def render_gradcam_overlay(
    image: Image.Image,
    activation_map: np.ndarray,
    intensity: float = 0.55,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    activation_threshold: float = 0.15,
    gamma: float = 0.75,
) -> Image.Image:
    """Render a contrast-enhanced blue-yellow-red Grad-CAM overlay."""
    if activation_map.ndim != 2:
        raise ValueError("Grad-CAM activation map must be two-dimensional")

    base = image.convert("RGB")
    values = np.nan_to_num(
        activation_map.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0
    )
    resized = Image.fromarray(values).resize(base.size, Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32)

    lower, upper = np.percentile(values, (lower_percentile, upper_percentile))
    if upper <= lower + 1e-8:
        lower, upper = float(values.min()), float(values.max())
    if upper <= lower + 1e-8:
        return base

    values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    values = np.clip(
        (values - activation_threshold) / max(1.0 - activation_threshold, 1e-8),
        0.0,
        1.0,
    )
    values = np.power(values, gamma)

    red = np.clip(2.0 * values, 0.0, 1.0)
    green = np.clip(1.0 - 2.0 * np.abs(values - 0.5), 0.0, 1.0)
    blue = np.clip(1.0 - 2.0 * values, 0.0, 1.0)
    color = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    overlay = Image.fromarray(color)

    opacity = np.clip(values * float(intensity), 0.0, 1.0)
    mask = Image.fromarray((opacity * 255).astype(np.uint8))
    return Image.composite(overlay, base, mask)
