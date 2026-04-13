"""Statistical analysis helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: list[str],
) -> dict[str, float | None]:
    """Compute common classification metrics for binary and multiclass probes."""
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if len(classes) == 2:
            positive_index = 1
            true_binary = (y_true == classes[positive_index]).astype(int)
            metrics["auroc_macro_ovr"] = float(
                roc_auc_score(true_binary, y_score[:, positive_index])
            )
        else:
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["auroc_macro_ovr"] = float(
                roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["auroc_macro_ovr"] = None
    return metrics


def summarize_permutation_metrics(
    metric_rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, dict[str, float | None]]:
    """Summarize repeated permutation-baseline metrics."""
    summary: dict[str, dict[str, float | None]] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in metric_rows
            if row.get(metric_name) is not None
        ]
        if not values:
            summary[metric_name] = {"mean": None, "std": None}
            continue
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


def clustering_metrics(true_labels: np.ndarray, cluster_labels: np.ndarray) -> dict[str, float]:
    """Compare unsupervised clusters to framing labels without fitting on those labels."""
    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, cluster_labels)),
    }
