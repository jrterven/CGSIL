from __future__ import annotations

import numpy as np


def compute_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, preds), 1)
    return confusion


def compute_classification_metrics(targets, preds, num_classes: int) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    confusion = compute_confusion_matrix(targets, preds, num_classes)
    true_positives = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)

    recall = np.divide(true_positives, support, out=np.zeros_like(true_positives), where=support > 0)
    precision = np.divide(true_positives, predicted, out=np.zeros_like(true_positives), where=predicted > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(true_positives), where=(precision + recall) > 0)

    accuracy = float((targets == preds).mean()) if targets.size > 0 else 0.0
    balanced_accuracy = float(recall.mean()) if recall.size > 0 else 0.0
    macro_f1 = float(f1.mean()) if f1.size > 0 else 0.0
    macro_precision = float(precision.mean()) if precision.size > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "confusion_matrix": confusion.tolist(),
    }
