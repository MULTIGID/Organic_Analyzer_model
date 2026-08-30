from pathlib import Path

from src.plantvillage_data import _leaf_group


def test_leaf_group_uses_class_specific_mapping():
    path = "raw/color/Apple___healthy/id___RS_HL 100.JPG"
    leaf_map = {"rs_hl 100": ["Soybean___healthy:::1.0", "Apple___healthy:::2.0"]}
    assert _leaf_group(path, leaf_map) == "Apple___healthy:::2.0"


def test_leaf_group_has_stable_fallback():
    path = "raw/color/Apple___healthy/id___RS_HL 100.JPG"
    assert _leaf_group(path, {}) == "Apple___healthy:::rs_hl 100"


def test_path_class_is_parent_directory():
    path = Path("raw/color/Tomato___healthy/image.JPG")
    assert path.parts[-2] == "Tomato___healthy"
