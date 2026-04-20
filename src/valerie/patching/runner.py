"""CLI runner for activation patching experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

from valerie.config import load_model_config, load_probe_config
from valerie.models.loader import load_model
from valerie.patching.patcher import PairPatchingResult, run_patch_experiment
from valerie.probes.dataset import ActivationSample, load_activation_run


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _build_output_directory(base_dir: str | Path, activation_dir: Path) -> Path:
    return Path(base_dir) / f"patching_{activation_dir.name}_{_timestamp()}"


def _pair_samples(
    samples: list[ActivationSample],
    clean_condition: str,
    corrupted_condition: str,
) -> list[tuple[ActivationSample, ActivationSample]]:
    """Return matched (clean, corrupted) pairs by task_id and variant_index."""
    clean_index: dict[tuple[str, int], ActivationSample] = {}
    corrupted_index: dict[tuple[str, int], ActivationSample] = {}
    for sample in samples:
        key = (sample.task_id, sample.variant_index)
        if sample.condition_name == clean_condition:
            clean_index[key] = sample
        elif sample.condition_name == corrupted_condition:
            corrupted_index[key] = sample
    shared = sorted(set(clean_index) & set(corrupted_index))
    return [(clean_index[k], corrupted_index[k]) for k in shared]


def _aggregate_by_layer(
    results: list[PairPatchingResult],
) -> pd.DataFrame:
    """Aggregate per-pair patching results into a per-layer summary."""
    rows_by_layer: dict[int, list[PairPatchingResult]] = defaultdict(list)
    for r in results:
        rows_by_layer[r.layer].append(r)
    summary_rows = []
    for layer in sorted(rows_by_layer):
        layer_results = rows_by_layer[layer]
        recoveries = [r.recovery_cosine for r in layer_results]
        kl_baselines = [r.kl_baseline for r in layer_results]
        kl_patcheds = [r.kl_patched for r in layer_results]
        summary_rows.append(
            {
                "layer": layer,
                "n_pairs": len(layer_results),
                "recovery_cosine_mean": float(np.mean(recoveries)),
                "recovery_cosine_std": float(np.std(recoveries)),
                "kl_baseline_mean": float(np.mean(kl_baselines)),
                "kl_patched_mean": float(np.mean(kl_patcheds)),
                "kl_reduction_mean": float(np.mean(kl_baselines) - np.mean(kl_patcheds)),
            }
        )
    return pd.DataFrame(summary_rows)


def run_patching_analysis(
    activation_dir: str | Path,
    model_config_path: str | Path,
    probe_config_path: str | Path,
    output_dir: str | Path = "data/patching",
    clean_condition: str = "care",
    corrupted_condition: str = "threat",
    component: str = "resid_post",
) -> Path:
    """Run the full activation patching experiment on a saved activation directory."""
    activation_dir = Path(activation_dir)
    output_root = _build_output_directory(output_dir, activation_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_activation_run(activation_dir, component=component)
    model_config = load_model_config(model_config_path)
    model = load_model(model_config)

    pairs = _pair_samples(dataset.samples, clean_condition, corrupted_condition)
    if not pairs:
        raise ValueError(
            f"no matched pairs found for conditions '{clean_condition}' and '{corrupted_condition}'"
        )

    all_results: list[PairPatchingResult] = []
    for clean_sample, corrupted_sample in pairs:
        pair_results = run_patch_experiment(
            model=model,
            clean_prompt=clean_sample.prompt,
            corrupted_prompt=corrupted_sample.prompt,
            layers=dataset.layers,
            component=component,
            clean_condition=clean_condition,
            corrupted_condition=corrupted_condition,
            task_id=clean_sample.task_id,
            variant_index=clean_sample.variant_index,
        )
        all_results.extend(pair_results)

    pair_rows = [
        {
            "layer": r.layer,
            "component": r.component,
            "task_id": r.task_id,
            "variant_index": r.variant_index,
            "clean_condition": r.clean_condition,
            "corrupted_condition": r.corrupted_condition,
            "recovery_cosine": r.recovery_cosine,
            "kl_baseline": r.kl_baseline,
            "kl_patched": r.kl_patched,
        }
        for r in all_results
    ]
    pair_df = pd.DataFrame(pair_rows)
    _write_csv(output_root / "metrics" / "patching_pairs.csv", pair_df)

    summary_df = _aggregate_by_layer(all_results)
    _write_csv(output_root / "metrics" / "patching_summary.csv", summary_df)
    _write_json(
        output_root / "metrics" / "patching_summary.json",
        {"rows": summary_df.to_dict(orient="records")},
    )

    _plot_recovery(summary_df, output_root / "plots" / "recovery_by_layer.png")

    best_layer_row = summary_df.sort_values("recovery_cosine_mean", ascending=False).iloc[0]
    manifest = {
        "source_activation_dir": str(activation_dir),
        "source_model_name": dataset.resolved_model_config["name"],
        "source_experiment_name": dataset.manifest["experiment_name"],
        "clean_condition": clean_condition,
        "corrupted_condition": corrupted_condition,
        "component": component,
        "n_pairs": len(pairs),
        "layers_analyzed": dataset.layers,
        "summary": {
            "best_recovery_layer": int(best_layer_row["layer"]),
            "best_recovery_cosine_mean": float(best_layer_row["recovery_cosine_mean"]),
            "kl_baseline_mean": float(summary_df["kl_baseline_mean"].mean()),
            "kl_patched_mean": float(summary_df["kl_patched_mean"].mean()),
        },
        "artifacts": {
            "patching_pairs_csv": "metrics/patching_pairs.csv",
            "patching_summary_csv": "metrics/patching_summary.csv",
            "recovery_plot": "plots/recovery_by_layer.png",
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return output_root


def _plot_recovery(summary_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = summary_df["layer"].tolist()
    means = summary_df["recovery_cosine_mean"].tolist()
    stds = summary_df["recovery_cosine_std"].tolist()

    ax.plot(layers, means, marker="o", color="steelblue", label="Recovery (cosine)", linewidth=2)
    ax.fill_between(
        layers,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        color="steelblue",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="No effect")
    ax.axhline(1, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Perfect recovery")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery (cosine similarity)")
    ax.set_title("Activation Patching Recovery by Layer\n(care → threat, last token resid_post)")
    ax.set_xticks(layers)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Valerie activation patching experiment on saved activations."
    )
    parser.add_argument(
        "--activation-dir",
        required=True,
        help="Path to an activation artifact directory produced by valerie-run-experiment.",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to the YAML model config used to produce the activation run.",
    )
    parser.add_argument(
        "--probe-config",
        required=False,
        help="Unused — reserved for future integration with probe results.",
    )
    parser.add_argument(
        "--clean-condition",
        default="care",
        help="Condition to use as the clean (source) activations. Default: care.",
    )
    parser.add_argument(
        "--corrupted-condition",
        default="threat",
        help="Condition to use as the corrupted (target) prompt. Default: threat.",
    )
    parser.add_argument(
        "--component",
        default="resid_post",
        help="Activation component to patch. Default: resid_post.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/patching",
        help="Base directory where patching results will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = run_patching_analysis(
        activation_dir=args.activation_dir,
        model_config_path=args.model_config,
        probe_config_path=args.probe_config,
        output_dir=args.output_dir,
        clean_condition=args.clean_condition,
        corrupted_condition=args.corrupted_condition,
        component=args.component,
    )
    print(output_root)


if __name__ == "__main__":
    main()
