from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_probability >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_probability)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def multiclass_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray,
    class_count: int,
) -> dict[str, float | list[list[int]]]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    y_pred = y_probability.argmax(axis=1)
    class_labels = list(range(class_count))
    matrix = (
        confusion_matrix(y_true, y_pred, labels=class_labels).tolist()
        if class_count <= 200 else []
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": matrix,
    }


def multiclass_metrics_from_predictions(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    class_count: int,
) -> dict[str, float | list[list[int]]]:
    """Compute multiclass metrics without retaining a full probability matrix."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    class_labels = list(range(class_count))
    matrix = (
        confusion_matrix(y_true, y_pred, labels=class_labels).tolist()
        if class_count <= 200 else []
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": matrix,
    }
