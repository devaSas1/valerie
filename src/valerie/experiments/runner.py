"""CLI runner for paired framing experiments."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from valerie.config import load_experiment_config, load_model_config
from valerie.experiments.framings import (
    build_framed_prompts,
    summarize_length_deltas,
    summarize_token_length_deltas,
)
from valerie.extraction.activations import extract_requested_activations, save_activation_payload
from valerie.models.loader import DependencyUnavailableError, load_model


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_output_directory(base_dir: str | Path, experiment_name: str, model_name: str) -> Path:
    safe_experiment = experiment_name.replace(" ", "-")
    safe_model = model_name.replace("/", "--")
    return Path(base_dir) / f"{safe_experiment}_{safe_model}_{_timestamp()}"


def run_experiment(
    model_config_path: str | Path,
    experiment_config_path: str | Path,
    output_dir: str | Path = "data/activations",
) -> Path:
    model_config = load_model_config(model_config_path)
    experiment_config = load_experiment_config(experiment_config_path)
    _set_deterministic_seed(experiment_config.controls.deterministic_seed)
    length_deltas = summarize_length_deltas(experiment_config)

    output_root = build_output_directory(output_dir, experiment_config.name, model_config.name)
    samples_dir = output_root / "samples"

    try:
        model = load_model(model_config)
    except DependencyUnavailableError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to load model '{model_config.name}': {exc}") from exc

    token_length_deltas: dict | None = None
    if model.tokenize("") is not None:
        token_length_deltas = summarize_token_length_deltas(experiment_config, model.tokenize)

    manifest = {
        "experiment_name": experiment_config.name,
        "experiment_description": experiment_config.description,
        "model_name": model_config.name,
        "model_backend": model_config.backend,
        "model_identifier": model_config.model_name,
        "device": model.device,
        "n_layers": model.n_layers,
        "deterministic_seed": experiment_config.controls.deterministic_seed,
        "extraction": experiment_config.extraction.model_dump(mode="json"),
        "length_deltas": length_deltas,
        "token_length_deltas": token_length_deltas,
        "warnings": [],
        "samples": [],
    }

    threshold = experiment_config.controls.warn_on_char_length_delta_over
    for task_id, length_summary in length_deltas.items():
        if length_summary["delta_chars"] > threshold:
            manifest["warnings"].append(
                "task "
                f"'{task_id}' has prompt length delta "
                f"{length_summary['delta_chars']} > {threshold}"
            )

    for framed_prompt in build_framed_prompts(experiment_config):
        payload = extract_requested_activations(
            model,
            framed_prompt.prompt,
            experiment_config.extraction,
        )
        payload["metadata"] = {
            "sample_id": framed_prompt.sample_id,
            "task_id": framed_prompt.task_id,
            "condition_name": framed_prompt.condition_name,
            "condition_target": framed_prompt.condition_target,
            "variant_index": framed_prompt.variant_index,
            "char_length": framed_prompt.char_length,
        }
        sample_path = samples_dir / f"{framed_prompt.sample_id}.pt"
        save_activation_payload(payload, sample_path)
        manifest["samples"].append(
            {
                "sample_id": framed_prompt.sample_id,
                "task_id": framed_prompt.task_id,
                "condition_name": framed_prompt.condition_name,
                "variant_index": framed_prompt.variant_index,
                "path": str(sample_path.relative_to(output_root)),
            }
        )

    _write_json(
        output_root / "manifest.json",
        manifest,
    )
    _write_json(
        output_root / "resolved_model_config.json",
        model_config.model_dump(mode="json"),
    )
    _write_json(
        output_root / "resolved_experiment_config.json",
        experiment_config.model_dump(mode="json"),
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Valerie framing experiment.")
    parser.add_argument("--model-config", required=True, help="Path to a YAML model config.")
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/activations",
        help="Base directory where activation artifacts will be saved.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = run_experiment(
        model_config_path=args.model_config,
        experiment_config_path=args.experiment_config,
        output_dir=args.output_dir,
    )
    print(output_root)


if __name__ == "__main__":
    main()
