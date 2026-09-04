from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class BatchImageAugmentation(nn.Module):
    """Apply independent image augmentations to an NCHW batch on its device."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        count = images.shape[0]
        horizontal = torch.rand(count, device=images.device) < 0.5
        vertical = torch.rand(count, device=images.device) < 0.5
        images = torch.where(horizontal[:, None, None, None], images.flip(-1), images)
        images = torch.where(vertical[:, None, None, None], images.flip(-2), images)

        angles = (torch.rand(count, device=images.device) * 30.0 - 15.0) * math.pi / 180.0
        cosine, sine = angles.cos(), angles.sin()
        theta = torch.zeros((count, 2, 3), device=images.device, dtype=images.dtype)
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
        grid = functional.affine_grid(theta, images.shape, align_corners=False)
        images = functional.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        brightness = 0.9 + 0.2 * torch.rand(count, 1, 1, 1, device=images.device)
        contrast = 0.9 + 0.2 * torch.rand(count, 1, 1, 1, device=images.device)
        saturation = 0.95 + 0.1 * torch.rand(count, 1, 1, 1, device=images.device)
        images = images * brightness
        channel_mean = images.mean(dim=(-2, -1), keepdim=True)
        images = (images - channel_mean) * contrast + channel_mean
        grayscale = (
            images[:, 0:1] * 0.2989
            + images[:, 1:2] * 0.5870
            + images[:, 2:3] * 0.1140
        )
        images = (images - grayscale) * saturation + grayscale
        images = images.clamp_(0.0, 1.0)
        return (images - self.mean) / self.std
