"""Probe training entrypoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut

from valerie.analysis.statistics import (
    classification_metrics,
    clustering_metrics,
    summarize_permutation_metrics,
)
from valerie.analysis.visualization import plot_layer_metric_profile, plot_pca_scatter
from valerie.config import ProbeConfig, load_probe_config
from valerie.probes.dataset import ActivationRunDataset, ActivationSample, load_activation_run
from valerie.probes.linear import build_logistic_regression_probe


@dataclass(frozen=True)
class LayerDataset:
    """Feature matrix plus labels for one layer."""

    layer: int
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: list[str]
    variant_indices: np.ndarray
    task_ids: np.ndarray
    condition_names: np.ndarray


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _build_results_directory(base_dir: str | Path, probe_name: str, activation_dir: Path) -> Path:
    activation_name = activation_dir.name
    safe_probe = probe_name.replace(" ", "-")
    return Path(base_dir) / f"{safe_probe}_{activation_name}_{_timestamp()}"


def _select_condition_samples(
    samples: list[ActivationSample],
    condition_names: set[str],
) -> list[ActivationSample]:
    return [sample for sample in samples if sample.condition_name in condition_names]


def _build_layer_dataset(samples: list[ActivationSample], layer: int) -> LayerDataset:
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    x = np.stack([sample.features_by_layer[layer] for sample in ordered])
    y = np.array([sample.condition_name for sample in ordered], dtype=object)
    groups = np.array([sample.task_id for sample in ordered], dtype=object)
    variant_indices = np.array([sample.variant_index for sample in ordered], dtype=int)
    task_ids = np.array([sample.task_id for sample in ordered], dtype=object)
    condition_names = np.array([sample.condition_name for sample in ordered], dtype=object)
    sample_ids = [sample.sample_id for sample in ordered]
    return LayerDataset(
        layer=layer,
        x=x,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        variant_indices=variant_indices,
        task_ids=task_ids,
        condition_names=condition_names,
    )


def _ensure_task_generalization_is_possible(groups: np.ndarray) -> None:
    if len(np.unique(groups)) < 2:
        raise ValueError("task-held-out evaluation requires at least two unique task ids")


def _groupwise_permute_labels(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    permuted = y.copy()
    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def _cross_validated_predictions(
    base_model,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = LeaveOneGroupOut()
    _ensure_task_generalization_is_possible(groups)

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_score_parts: list[np.ndarray] = []

    for train_index, test_index in splitter.split(x, y, groups):
        model = clone(base_model)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_labels = set(y_train.tolist())
        if train_labels != set(classes):
            raise ValueError(
                "each training fold must contain all target classes for held-out task evaluation"
            )

        model.fit(x_train, y_train)
        fold_predictions = model.predict(x_test)
        fold_probabilities = model.predict_proba(x_test)
        probability_df = pd.DataFrame(fold_probabilities, columns=model.classes_)
        aligned_probabilities = probability_df.reindex(columns=classes, fill_value=0.0).to_numpy()

        y_true_parts.append(y_test)
        y_pred_parts.append(np.array(fold_predictions, dtype=object))
        y_score_parts.append(aligned_probabilities)

    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(y_score_parts),
    )


def _fit_full_model(base_model, x: np.ndarray, y: np.ndarray):
    model = clone(base_model)
    model.fit(x, y)
    return model


def _run_supervised_layer_analysis(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
    output_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected_samples = _select_condition_samples(
        dataset.samples,
        set(probe_config.framing_conditions),
    )
    classes = list(probe_config.framing_conditions)
    base_model = build_logistic_regression_probe(probe_config)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    model_dir = output_root / "models" / "multiclass"
    model_dir.mkdir(parents=True, exist_ok=True)

    for layer in dataset.layers:
        layer_dataset = _build_layer_dataset(selected_samples, layer)
        y_true, y_pred, y_score = _cross_validated_predictions(
            base_model,
            layer_dataset.x,
            layer_dataset.y,
            layer_dataset.groups,
            classes,
        )
        metrics = classification_metrics(y_true, y_pred, y_score, classes)

        rng = np.random.default_rng(probe_config.random_seed + layer)
        permutation_rows: list[dict[str, Any]] = []
        for permutation_index in range(probe_config.num_permutations):
            permuted_labels = _groupwise_permute_labels(layer_dataset.y, layer_dataset.groups, rng)
            perm_true, perm_pred, perm_score = _cross_validated_predictions(
                base_model,
                layer_dataset.x,
                permuted_labels,
                layer_dataset.groups,
                classes,
            )
            permutation_metrics = classification_metrics(perm_true, perm_pred, perm_score, classes)
            permutation_metrics["permutation_index"] = permutation_index
            permutation_rows.append(permutation_metrics)

        permutation_summary = summarize_permutation_metrics(
            permutation_rows,
            metric_names=["accuracy", "f1_macro", "auroc_macro_ovr"],
        )

        metric_row = {
            "layer": layer,
            "n_samples": int(layer_dataset.x.shape[0]),
            "n_features": int(layer_dataset.x.shape[1]),
            "conditions": classes,
            **metrics,
            "permutation_accuracy_mean": permutation_summary["accuracy"]["mean"],
            "permutation_accuracy_std": permutation_summary["accuracy"]["std"],
            "permutation_f1_macro_mean": permutation_summary["f1_macro"]["mean"],
            "permutation_f1_macro_std": permutation_summary["f1_macro"]["std"],
            "permutation_auroc_macro_ovr_mean": permutation_summary["auroc_macro_ovr"]["mean"],
            "permutation_auroc_macro_ovr_std": permutation_summary["auroc_macro_ovr"]["std"],
        }
        metric_rows.append(metric_row)

        full_model = _fit_full_model(base_model, layer_dataset.x, layer_dataset.y)
        if probe_config.save_models:
            joblib.dump(full_model, model_dir / f"layer_{layer:02d}.joblib")

        for sample_id, task_id, true_label, predicted_label in zip(
            layer_dataset.sample_ids,
            layer_dataset.task_ids,
            y_true,
            y_pred,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "layer": layer,
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("layer").reset_index(drop=True)
    _write_csv(output_root / "metrics" / "multiclass_metrics.csv", metrics_df)
    _write_json(
        output_root / "metrics" / "multiclass_metrics.json",
        {"rows": metrics_df.to_dict(orient="records")},
    )
    _write_csv(
        output_root / "metrics" / "multiclass_predictions.csv",
        pd.DataFrame(prediction_rows),
    )

    plot_layer_metric_profile(
        metrics_df,
        output_root / "plots" / "multiclass_accuracy_by_layer.png",
        metric_name="accuracy",
        baseline_metric_name="permutation_accuracy_mean",
        baseline_std_name="permutation_accuracy_std",
        title="Threat vs Care vs Neutral Accuracy by Layer",
    )
    return metrics_df, metric_rows


def _run_narrative_control_analysis(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
    output_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    target_conditions = {
        probe_config.narrative_reference_condition,
        probe_config.narrative_control_condition,
    }
    selected_samples = _select_condition_samples(dataset.samples, target_conditions)
    classes = [probe_config.narrative_reference_condition, probe_config.narrative_control_condition]
    base_model = build_logistic_regression_probe(probe_config)
    metric_rows: list[dict[str, Any]] = []
    model_dir = output_root / "models" / "narrative_control"
    model_dir.mkdir(parents=True, exist_ok=True)

    for layer in dataset.layers:
        layer_dataset = _build_layer_dataset(selected_samples, layer)
        y_true, y_pred, y_score = _cross_validated_predictions(
            base_model,
            layer_dataset.x,
            layer_dataset.y,
            layer_dataset.groups,
            classes,
        )
        metrics = classification_metrics(y_true, y_pred, y_score, classes)

        rng = np.random.default_rng(probe_config.random_seed + 1000 + layer)
        permutation_rows: list[dict[str, Any]] = []
        for permutation_index in range(probe_config.num_permutations):
            permuted_labels = _groupwise_permute_labels(layer_dataset.y, layer_dataset.groups, rng)
            perm_true, perm_pred, perm_score = _cross_validated_predictions(
                base_model,
                layer_dataset.x,
                permuted_labels,
                layer_dataset.groups,
                classes,
            )
            permutation_metrics = classification_metrics(perm_true, perm_pred, perm_score, classes)
            permutation_metrics["permutation_index"] = permutation_index
            permutation_rows.append(permutation_metrics)

        permutation_summary = summarize_permutation_metrics(
            permutation_rows,
            metric_names=["accuracy", "f1_macro", "auroc_macro_ovr"],
        )
        metric_row = {
            "layer": layer,
            "n_samples": int(layer_dataset.x.shape[0]),
            "n_features": int(layer_dataset.x.shape[1]),
            "conditions": classes,
            **metrics,
            "permutation_accuracy_mean": permutation_summary["accuracy"]["mean"],
            "permutation_accuracy_std": permutation_summary["accuracy"]["std"],
            "permutation_f1_macro_mean": permutation_summary["f1_macro"]["mean"],
            "permutation_f1_macro_std": permutation_summary["f1_macro"]["std"],
            "permutation_auroc_macro_ovr_mean": permutation_summary["auroc_macro_ovr"]["mean"],
            "permutation_auroc_macro_ovr_std": permutation_summary["auroc_macro_ovr"]["std"],
        }
        metric_rows.append(metric_row)

        full_model = _fit_full_model(base_model, layer_dataset.x, layer_dataset.y)
        if probe_config.save_models:
            joblib.dump(full_model, model_dir / f"layer_{layer:02d}.joblib")

    metrics_df = pd.DataFrame(metric_rows).sort_values("layer").reset_index(drop=True)
    _write_csv(output_root / "metrics" / "narrative_control_metrics.csv", metrics_df)
    _write_json(
        output_root / "metrics" / "narrative_control_metrics.json",
        {"rows": metrics_df.to_dict(orient="records")},
    )
    plot_layer_metric_profile(
        metrics_df,
        output_root / "plots" / "narrative_control_accuracy_by_layer.png",
        metric_name="accuracy",
        baseline_metric_name="permutation_accuracy_mean",
        baseline_std_name="permutation_accuracy_std",
        title="Threat vs Narrative Threat Accuracy by Layer",
    )
    return metrics_df, metric_rows


def _condition_color_subset(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
) -> list[ActivationSample]:
    allowed = set(probe_config.framing_conditions)
    allowed.add(probe_config.narrative_control_condition)
    return _select_condition_samples(dataset.samples, allowed)


def _run_activation_pca(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
    output_root: Path,
) -> list[dict[str, Any]]:
    selected_samples = _condition_color_subset(dataset, probe_config)
    rows: list[dict[str, Any]] = []
    for layer in dataset.layers:
        layer_dataset = _build_layer_dataset(selected_samples, layer)
        if np.allclose(np.var(layer_dataset.x, axis=0).sum(), 0.0):
            rows.append(
                {
                    "layer": layer,
                    "n_components": 0,
                    "explained_variance_ratio": [],
                    "warning": "skipped because activation variance is zero",
                }
            )
            continue
        n_components = min(
            probe_config.pca_components,
            layer_dataset.x.shape[0],
            layer_dataset.x.shape[1],
        )
        if n_components < 2:
            continue
        pca = PCA(n_components=n_components, random_state=probe_config.random_seed)
        coordinates = pca.fit_transform(layer_dataset.x)
        coordinates_df = pd.DataFrame(
            {
                "sample_id": layer_dataset.sample_ids,
                "task_id": layer_dataset.task_ids,
                "condition_name": layer_dataset.condition_names,
                "variant_index": layer_dataset.variant_indices,
                "pc1": coordinates[:, 0],
                "pc2": coordinates[:, 1],
            }
        )
        if n_components >= 3:
            coordinates_df["pc3"] = coordinates[:, 2]
        _write_csv(
            output_root / "tables" / "activation_pca" / f"layer_{layer:02d}.csv",
            coordinates_df,
        )
        plot_pca_scatter(
            coordinates_df,
            output_root / "plots" / "activation_pca" / f"layer_{layer:02d}_pc12.png",
            title=f"Activation PCA Layer {layer}",
        )
        if "pc3" in coordinates_df.columns:
            plot_pca_scatter(
                coordinates_df,
                output_root / "plots" / "activation_pca" / f"layer_{layer:02d}_pc13.png",
                title=f"Activation PCA Layer {layer} (PC1 vs PC3)",
                y_column="pc3",
            )
        rows.append(
            {
                "layer": layer,
                "n_components": n_components,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
        )
    _write_json(output_root / "metrics" / "activation_pca_summary.json", {"rows": rows})
    return rows


def _paired_threat_care_samples(dataset: ActivationRunDataset, probe_config: ProbeConfig):
    threat_samples = {
        (sample.task_id, sample.variant_index): sample
        for sample in dataset.samples
        if sample.condition_name == "threat"
    }
    care_samples = {
        (sample.task_id, sample.variant_index): sample
        for sample in dataset.samples
        if sample.condition_name == "care"
    }
    shared_keys = sorted(set(threat_samples) & set(care_samples))
    if not shared_keys:
        return shared_keys, threat_samples, care_samples
    return shared_keys, threat_samples, care_samples


def _run_difference_pca(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
    output_root: Path,
) -> list[dict[str, Any]]:
    shared_keys, threat_samples, care_samples = _paired_threat_care_samples(dataset, probe_config)
    rows: list[dict[str, Any]] = []
    if not shared_keys:
        _write_json(
            output_root / "metrics" / "difference_pca_summary.json",
            {"rows": rows, "warning": "no paired threat/care samples were found"},
        )
        return rows

    for layer in dataset.layers:
        differences = []
        task_ids = []
        variant_indices = []
        for task_id, variant_index in shared_keys:
            threat_vector = threat_samples[(task_id, variant_index)].features_by_layer[layer]
            care_vector = care_samples[(task_id, variant_index)].features_by_layer[layer]
            differences.append(threat_vector - care_vector)
            task_ids.append(task_id)
            variant_indices.append(variant_index)
        x = np.stack(differences)
        if np.allclose(np.var(x, axis=0).sum(), 0.0):
            rows.append(
                {
                    "layer": layer,
                    "n_components": 0,
                    "n_pairs": int(x.shape[0]),
                    "explained_variance_ratio": [],
                    "warning": "skipped because paired differences have zero variance",
                }
            )
            continue
        n_components = min(probe_config.pca_components, x.shape[0], x.shape[1])
        if n_components < 1:
            continue
        pca = PCA(n_components=n_components, random_state=probe_config.random_seed)
        coordinates = pca.fit_transform(x)
        coordinates_df = pd.DataFrame(
            {
                "task_id": task_ids,
                "variant_index": variant_indices,
                "pc1": coordinates[:, 0],
            }
        )
        if n_components >= 2:
            coordinates_df["pc2"] = coordinates[:, 1]
        if n_components >= 3:
            coordinates_df["pc3"] = coordinates[:, 2]
        _write_csv(
            output_root / "tables" / "difference_pca" / f"layer_{layer:02d}.csv",
            coordinates_df,
        )
        rows.append(
            {
                "layer": layer,
                "n_components": n_components,
                "n_pairs": int(x.shape[0]),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
        )
    _write_json(output_root / "metrics" / "difference_pca_summary.json", {"rows": rows})
    return rows


def _run_clustering(
    dataset: ActivationRunDataset,
    probe_config: ProbeConfig,
    output_root: Path,
) -> pd.DataFrame:
    selected_samples = _condition_color_subset(dataset, probe_config)
    rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for layer in dataset.layers:
        layer_dataset = _build_layer_dataset(selected_samples, layer)
        unique_points = int(np.unique(layer_dataset.x, axis=0).shape[0])
        n_clusters = min(len(np.unique(layer_dataset.condition_names)), unique_points)
        if probe_config.clustering_method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(layer_dataset.x)
        else:
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=probe_config.random_seed,
                n_init=10,
            )
            cluster_labels = clusterer.fit_predict(layer_dataset.x)

        metrics = clustering_metrics(layer_dataset.condition_names, cluster_labels)
        rows.append(
            {
                "layer": layer,
                "n_clusters": int(n_clusters),
                "method": probe_config.clustering_method,
                **metrics,
            }
        )
        for sample_id, task_id, condition_name, cluster_label in zip(
            layer_dataset.sample_ids,
            layer_dataset.task_ids,
            layer_dataset.condition_names,
            cluster_labels,
            strict=True,
        ):
            assignment_rows.append(
                {
                    "layer": layer,
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "condition_name": condition_name,
                    "cluster_label": int(cluster_label),
                }
            )

    metrics_df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    _write_csv(output_root / "metrics" / "clustering_metrics.csv", metrics_df)
    _write_json(output_root / "metrics" / "clustering_metrics.json", {"rows": rows})
    _write_csv(output_root / "tables" / "cluster_assignments.csv", pd.DataFrame(assignment_rows))
    return metrics_df


def _copy_source_metadata(dataset: ActivationRunDataset, output_root: Path) -> None:
    _write_json(output_root / "source_activation_manifest.json", dataset.manifest)
    _write_json(output_root / "source_model_config.json", dataset.resolved_model_config)
    _write_json(output_root / "source_experiment_config.json", dataset.resolved_experiment_config)


def run_probe_analysis(
    activation_dir: str | Path,
    probe_config_path: str | Path,
    output_dir: str | Path = "data/results",
) -> Path:
    """Run the full phase-4 probe analysis on a saved activation directory."""
    probe_config = load_probe_config(probe_config_path)
    dataset = load_activation_run(activation_dir, component=probe_config.input_component)
    output_root = _build_results_directory(output_dir, probe_config.name, dataset.activation_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _copy_source_metadata(dataset, output_root)
    _write_json(output_root / "resolved_probe_config.json", probe_config.model_dump(mode="json"))

    multiclass_metrics_df, multiclass_metric_rows = _run_supervised_layer_analysis(
        dataset,
        probe_config,
        output_root,
    )
    narrative_metrics_df, narrative_metric_rows = _run_narrative_control_analysis(
        dataset,
        probe_config,
        output_root,
    )
    activation_pca_rows = _run_activation_pca(dataset, probe_config, output_root)
    difference_pca_rows = _run_difference_pca(dataset, probe_config, output_root)
    clustering_metrics_df = _run_clustering(dataset, probe_config, output_root)

    summary = {
        "component": probe_config.input_component,
        "layers_analyzed": dataset.layers,
        "multiclass_best_accuracy_layer": int(
            multiclass_metrics_df.sort_values("accuracy", ascending=False).iloc[0]["layer"]
        ),
        "multiclass_best_accuracy": float(
            multiclass_metrics_df.sort_values("accuracy", ascending=False).iloc[0]["accuracy"]
        ),
        "narrative_best_accuracy_layer": int(
            narrative_metrics_df.sort_values("accuracy", ascending=False).iloc[0]["layer"]
        ),
        "narrative_best_accuracy": float(
            narrative_metrics_df.sort_values("accuracy", ascending=False).iloc[0]["accuracy"]
        ),
    }
    manifest = {
        "probe_name": probe_config.name,
        "probe_task": probe_config.task,
        "input_component": probe_config.input_component,
        "source_activation_dir": str(dataset.activation_dir),
        "source_model_name": dataset.resolved_model_config["name"],
        "source_experiment_name": dataset.manifest["experiment_name"],
        "n_samples": len(dataset.samples),
        "task_counts": dict(Counter(sample.task_id for sample in dataset.samples)),
        "condition_counts": dict(Counter(sample.condition_name for sample in dataset.samples)),
        "summary": summary,
        "artifacts": {
            "multiclass_metrics_csv": "metrics/multiclass_metrics.csv",
            "narrative_control_metrics_csv": "metrics/narrative_control_metrics.csv",
            "clustering_metrics_csv": "metrics/clustering_metrics.csv",
            "multiclass_accuracy_plot": "plots/multiclass_accuracy_by_layer.png",
            "narrative_control_accuracy_plot": "plots/narrative_control_accuracy_by_layer.png",
        },
        "rows": {
            "multiclass": multiclass_metric_rows,
            "narrative_control": narrative_metric_rows,
            "activation_pca": activation_pca_rows,
            "difference_pca": difference_pca_rows,
            "clustering": clustering_metrics_df.to_dict(orient="records"),
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Valerie framing probes on saved activations."
    )
    parser.add_argument(
        "--activation-dir",
        required=True,
        help="Path to an activation artifact directory produced by valerie-run-experiment.",
    )
    parser.add_argument("--probe-config", required=True, help="Path to a YAML probe config.")
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Base directory where probe results will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = run_probe_analysis(
        activation_dir=args.activation_dir,
        probe_config_path=args.probe_config,
        output_dir=args.output_dir,
    )
    print(output_root)


if __name__ == "__main__":
    main()
