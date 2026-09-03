from PIL import Image

from src.input_validation import validate_module_input


def test_small_inaturalist_image_has_warning():
    image = Image.new("RGB", (16, 16), "white")
    assert "too_small" in validate_module_input(image, "iNaturalist mini")


def test_ordinary_landscape_ratio_is_accepted():
    image = Image.new("RGB", (400, 225), (80, 140, 70))
    assert "extreme_aspect" not in validate_module_input(image, "iNaturalist mini")


def test_extreme_panorama_has_warning():
    image = Image.new("RGB", (600, 100), (80, 140, 70))
    assert "extreme_aspect" in validate_module_input(image, "iNaturalist mini")


def test_bright_image_has_warning():
    image = Image.new("RGB", (128, 128), "white")
    assert "too_bright" in validate_module_input(image, "iNaturalist mini")
