import unittest

import numpy as np
from PIL import Image

from src.gradcam import render_gradcam_overlay


class GradCamOverlayTests(unittest.TestCase):
    def test_uniform_gradcam_map_returns_original_image(self) -> None:
        image = Image.new("RGB", (16, 16), (30, 60, 90))
        result = render_gradcam_overlay(image, np.zeros((4, 4), dtype=np.float32))

        self.assertTrue(np.array_equal(np.asarray(result), np.asarray(image)))

    def test_gradcam_intensity_changes_overlay_strength(self) -> None:
        image = Image.new("RGB", (16, 16), (80, 80, 80))
        activation = np.zeros((4, 4), dtype=np.float32)
        activation[1:3, 1:3] = 1.0

        low = render_gradcam_overlay(image, activation, intensity=0.2)
        high = render_gradcam_overlay(image, activation, intensity=0.8)
        original = np.asarray(image, dtype=np.int16)
        low_difference = np.abs(np.asarray(low, dtype=np.int16) - original).sum()
        high_difference = np.abs(np.asarray(high, dtype=np.int16) - original).sum()

        self.assertGreater(high_difference, low_difference)


if __name__ == "__main__":
    unittest.main()
