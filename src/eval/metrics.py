# src/eval/metrics.py
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix


def qwk(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Quadratic weighted kappa for ordinal labels (1..K)."""
    # ensure lists/ndarrays
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    return float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic"))


def within_one_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Proportion of predictions within ±1 of true label."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    return float((np.abs(y_true_arr - y_pred_arr) <= 1).mean())


def confusion(
    y_true: Iterable[int], y_pred: Iterable[int], labels: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[int]]:
    """Return confusion matrix (rows=true, cols=pred) and labels ordering."""
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    return cm, labels


def classification_report_dict(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """
    Return classification report as a dict. Uses zero_division=0 to avoid warnings
    when labels have no predicted/true samples (useful for small datasets / tests).
    """
    return classification_report(
        list(y_true), list(y_pred), output_dict=True, zero_division=0
    )
