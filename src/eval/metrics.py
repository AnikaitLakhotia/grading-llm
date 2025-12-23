# src/eval/metrics.py
"""Evaluation metrics for ordinal grading tasks: QWK, ±1 accuracy, confusion helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix


def qwk(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """
    Quadratic Weighted Kappa using sklearn's Cohen Kappa with weights='quadratic'.
    y_true and y_pred should be integer labels (1..6).
    """
    return float(cohen_kappa_score(list(y_true), list(y_pred), weights="quadratic"))


def within_one_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Proportion of predictions within ±1 of the true label."""
    yt = np.array(list(y_true), dtype=int)
    yp = np.array(list(y_pred), dtype=int)
    return float(np.mean(np.abs(yt - yp) <= 1))


def confusion(
    y_true: Iterable[int], y_pred: Iterable[int], labels: List[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """Compute confusion matrix and label ordering returned."""
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    return cm, labels


def classification_report_dict(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """Return classification report (precision/recall/f1) as a dict keyed by label (string)."""
    return classification_report(list(y_true), list(y_pred), output_dict=True)
