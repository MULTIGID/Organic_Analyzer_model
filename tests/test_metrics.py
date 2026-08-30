import pytest

from src.metrics import classification_metrics, multiclass_metrics_from_predictions


def test_classification_metrics_for_perfect_predictions():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["auc"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_multiclass_metrics_from_predictions():
    result = multiclass_metrics_from_predictions(
        labels=[0, 1, 2, 2], predictions=[0, 1, 2, 1], class_count=3
    )
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
