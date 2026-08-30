from __future__ import annotations

import numpy as np
from PIL import Image


SQUARE_IMAGE_MODULES = {
    "PCam",
    "PBC",
    "NIH Malaria",
    "MedMNIST",
    "NCT-CRC-HE-100K",
}
HISTOLOGY_MODULES = {"PCam", "MedMNIST", "NCT-CRC-HE-100K"}


def validate_module_input(image: Image.Image, module: str) -> list[str]:
    """Return non-blocking warning codes for an uploaded inference image.

    The checks are intentionally inexpensive heuristics. They detect common input
    problems but do not determine whether an image truly belongs to a dataset.
    """

    warnings: list[str] = []
    width, height = image.size
    minimum_side = 96 if module == "iNaturalist mini" else 64
    if width < minimum_side or height < minimum_side:
        warnings.append("too_small")

    ratio = width / max(height, 1)
    if module in SQUARE_IMAGE_MODULES and not 0.65 <= ratio <= 1.54:
        warnings.append("square_aspect")
    elif module not in SQUARE_IMAGE_MODULES and not 0.25 <= ratio <= 4.0:
        warnings.append("extreme_aspect")

    rgb = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32)
    gray = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    p05, p95 = np.percentile(gray, (5, 95))
    visual_range = float(p95 - p05)
    mean_luminance = float(gray.mean())

    if mean_luminance < 24:
        warnings.append("too_dark")
    elif mean_luminance > 238:
        warnings.append("too_bright")

    if visual_range < 14:
        warnings.append("low_variation")

    saturation_proxy = float(
        np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))
    )
    if module in HISTOLOGY_MODULES and saturation_proxy < 7:
        warnings.append("low_color_histology")
    elif module == "PlantVillage" and saturation_proxy < 10:
        warnings.append("low_color_leaf")

    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    edge_strength = float(horizontal_edges + vertical_edges)
    if visual_range >= 14 and edge_strength < 2.2:
        warnings.append("possibly_blurry")

    return warnings
