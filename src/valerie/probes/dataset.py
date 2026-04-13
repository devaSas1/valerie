"""Utilities for loading saved activation runs into analysis-ready datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActivationSample:
    """Single saved activation sample with flattened per-layer features."""

    sample_id: str
    task_id: str
    condition_name: str
    condition_target: str
    variant_index: int
    prompt: str
    features_by_layer: dict[int, np.ndarray]


@dataclass(frozen=True)
class ActivationRunDataset:
    """Activation dataset loaded from a prior experiment run."""

    activation_dir: Path
    manifest: dict[str, Any]
    resolved_model_config: dict[str, Any]
    resolved_experiment_config: dict[str, Any]
    component: str
    layers: list[int]
    samples: list[ActivationSample]


def _parse_layer_from_key(key: str, component: str) -> int:
    prefix = f"{component}.layer_"
    if not key.startswith(prefix):
        raise ValueError(f"activation key '{key}' does not match component '{component}'")
    return int(key.removeprefix(prefix))


def _flatten_activation(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)


def load_activation_run(activation_dir: str | Path, component: str) -> ActivationRunDataset:
    """Load a saved activation directory into memory for probe analysis."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load saved activation payloads.") from exc

    root = Path(activation_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    model_config = json.loads((root / "resolved_model_config.json").read_text(encoding="utf-8"))
    experiment_config = json.loads(
        (root / "resolved_experiment_config.json").read_text(encoding="utf-8")
    )

    samples: list[ActivationSample] = []
    layers: set[int] = set()

    for sample_entry in manifest["samples"]:
        payload_path = root / sample_entry["path"]
        payload = torch.load(payload_path)
        matched_features: dict[int, np.ndarray] = {}
        for activation_key, value in payload["activations"].items():
            if activation_key.startswith(f"{component}.layer_"):
                layer = _parse_layer_from_key(activation_key, component)
                layers.add(layer)
                matched_features[layer] = _flatten_activation(value)
        if not matched_features:
            raise ValueError(
                f"sample '{sample_entry['sample_id']}' does not contain component '{component}'"
            )
        metadata = payload["metadata"]
        samples.append(
            ActivationSample(
                sample_id=metadata["sample_id"],
                task_id=metadata["task_id"],
                condition_name=metadata["condition_name"],
                condition_target=metadata["condition_target"],
                variant_index=int(metadata["variant_index"]),
                prompt=payload["prompt"],
                features_by_layer=matched_features,
            )
        )

    return ActivationRunDataset(
        activation_dir=root,
        manifest=manifest,
        resolved_model_config=model_config,
        resolved_experiment_config=experiment_config,
        component=component,
        layers=sorted(layers),
        samples=samples,
    )

