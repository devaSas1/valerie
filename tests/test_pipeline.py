from pathlib import Path

import torch

from valerie.experiments.runner import run_experiment
from valerie.probes.trainer import run_probe_analysis


def test_dummy_pipeline_writes_artifacts(tmp_path: Path):
    output_root = run_experiment(
        model_config_path="configs/models/dummy.yaml",
        experiment_config_path="configs/experiments/threat-vs-care.yaml",
        output_dir=tmp_path,
    )
    manifest_path = output_root / "manifest.json"
    assert manifest_path.exists()

    samples = sorted((output_root / "samples").glob("*.pt"))
    assert samples

    payload = torch.load(samples[0])
    assert "activations" in payload
    assert any(key.startswith("resid_post.layer_") for key in payload["activations"])


def test_dummy_probe_analysis_writes_results(tmp_path: Path):
    activation_root = run_experiment(
        model_config_path="configs/models/dummy.yaml",
        experiment_config_path="configs/experiments/threat-vs-care.yaml",
        output_dir=tmp_path / "activations",
    )
    results_root = run_probe_analysis(
        activation_dir=activation_root,
        probe_config_path="configs/probes/linear-framing.yaml",
        output_dir=tmp_path / "results",
    )
    assert (results_root / "manifest.json").exists()
    assert (results_root / "metrics" / "multiclass_metrics.csv").exists()
    assert (results_root / "plots" / "multiclass_accuracy_by_layer.png").exists()
