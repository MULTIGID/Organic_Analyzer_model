from pathlib import Path

import torch

from src.metrics import multiclass_metrics
from src.model import create_resnet50
from src.pbc_data import EXPECTED_PBC_CLASSES, discover_pbc_splits


def test_resnet50_pbc_output_shape():
    model = create_resnet50(pretrained=False, num_classes=8)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 8)


def test_multiclass_metrics_for_perfect_predictions():
    result = multiclass_metrics([0, 1], [[0.9, 0.1], [0.1, 0.9]], 2)
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_pbc_split_discovery(tmp_path: Path):
    for split in ("Train", "Val", "Test"):
        for class_name in EXPECTED_PBC_CLASSES:
            (tmp_path / split / class_name).mkdir(parents=True)
    result = discover_pbc_splits(tmp_path)
    assert set(result) == {"train", "validation", "test"}
