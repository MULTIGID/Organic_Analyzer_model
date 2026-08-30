import numpy as np
from PIL import Image

from src.input_validation import validate_module_input


def test_small_image_is_rejected_by_warning():
    image = Image.new("RGB", (16, 16), "white")
    warnings = validate_module_input(image, "PBC")
    assert "too_small" in warnings


def test_regular_colored_patch_has_no_geometry_warning():
    image = Image.new("RGB", (96, 96), (160, 80, 130))
    warnings = validate_module_input(image, "PCam")
    assert "square_aspect" not in warnings


def test_square_patch_model_warns_about_panorama():
    image = Image.new("RGB", (400, 100), (160, 80, 130))
    assert "square_aspect" in validate_module_input(image, "NIH Malaria")


def test_inaturalist_accepts_an_ordinary_landscape_ratio():
    image = Image.new("RGB", (400, 225), (80, 140, 70))
    assert "extreme_aspect" not in validate_module_input(image, "iNaturalist mini")


def test_bright_image_is_reported():
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    assert "too_bright" in validate_module_input(image, "PlantVillage")


def test_grayscale_histology_image_has_color_warning():
    gradient = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    rgb = np.stack((gradient, gradient, gradient), axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    assert "low_color_histology" in validate_module_input(image, "MedMNIST")


def test_low_color_warning_is_specific_to_the_image_domain():
    gradient = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    rgb = np.stack((gradient, gradient, gradient), axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    warnings = validate_module_input(image, "iNaturalist mini")
    assert "low_color_histology" not in warnings
    assert "low_color_leaf" not in warnings
