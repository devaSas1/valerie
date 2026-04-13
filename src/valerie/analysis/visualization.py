"""Visualization helpers."""
# ruff: noqa: E402

from __future__ import annotations

import os
from pathlib import Path
from tempfile import mkdtemp

_DEFAULT_MPL_DIR = Path(".cache/matplotlib")
if "MPLCONFIGDIR" not in os.environ:
    try:
        _DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_DEFAULT_MPL_DIR.resolve())
    except OSError:
        os.environ["MPLCONFIGDIR"] = mkdtemp(prefix="valerie-mpl-")

import matplotlib.pyplot as plt
import pandas as pd


def plot_layer_metric_profile(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_name: str = "accuracy",
    baseline_metric_name: str = "permutation_accuracy_mean",
    baseline_std_name: str = "permutation_accuracy_std",
    title: str = "Layerwise Accuracy",
) -> Path:
    """Plot a metric as a function of layer depth."""
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(metrics_df["layer"], metrics_df[metric_name], marker="o", label="Probe")

    if baseline_metric_name in metrics_df.columns:
        axis.plot(
            metrics_df["layer"],
            metrics_df[baseline_metric_name],
            linestyle="--",
            color="gray",
            label="Permutation mean",
        )
        if baseline_std_name in metrics_df.columns:
            lower = metrics_df[baseline_metric_name] - metrics_df[baseline_std_name]
            upper = metrics_df[baseline_metric_name] + metrics_df[baseline_std_name]
            axis.fill_between(metrics_df["layer"], lower, upper, color="gray", alpha=0.2)

    axis.set_xlabel("Layer")
    axis.set_ylabel(metric_name.replace("_", " ").title())
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_pca_scatter(
    coordinates_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    hue_column: str = "condition_name",
    x_column: str = "pc1",
    y_column: str = "pc2",
) -> Path:
    """Create a basic PCA scatter plot."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, subset in coordinates_df.groupby(hue_column):
        axis.scatter(subset[x_column], subset[y_column], label=label, alpha=0.8)
    axis.set_xlabel(x_column.upper())
    axis.set_ylabel(y_column.upper())
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output
