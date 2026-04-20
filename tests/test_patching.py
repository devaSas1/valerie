from pathlib import Path

import numpy as np
import pytest

from valerie.experiments.runner import run_experiment
from valerie.patching.patcher import (
    PairPatchingResult,
    _cosine_similarity,
    _kl_divergence,
    run_patch_experiment,
)
from valerie.patching.runner import _pair_samples, _aggregate_by_layer, run_patching_analysis
from valerie.models.loader import load_model
from valerie.config import load_model_config
from valerie.probes.dataset import load_activation_run


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_kl_divergence_identical():
    p = np.array([0.5, 0.5])
    assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_dummy_patching_pipeline_runs(tmp_path: Path):
    activation_root = run_experiment(
        model_config_path="configs/models/dummy.yaml",
        experiment_config_path="configs/experiments/threat-vs-care.yaml",
        output_dir=tmp_path / "activations",
    )
    output_root = run_patching_analysis(
        activation_dir=activation_root,
        model_config_path="configs/models/dummy.yaml",
        probe_config_path=None,
        output_dir=tmp_path / "patching",
        clean_condition="care",
        corrupted_condition="threat",
        component="resid_post",
    )
    assert (output_root / "manifest.json").exists()
    assert (output_root / "metrics" / "patching_summary.csv").exists()
    assert (output_root / "metrics" / "patching_pairs.csv").exists()


def test_pair_samples_matches_by_task_and_variant(tmp_path: Path):
    activation_root = run_experiment(
        model_config_path="configs/models/dummy.yaml",
        experiment_config_path="configs/experiments/threat-vs-care.yaml",
        output_dir=tmp_path / "activations",
    )
    dataset = load_activation_run(activation_root, component="resid_post")
    pairs = _pair_samples(dataset.samples, "care", "threat")
    assert len(pairs) > 0
    for clean, corrupted in pairs:
        assert clean.task_id == corrupted.task_id
        assert clean.variant_index == corrupted.variant_index
        assert clean.condition_name == "care"
        assert corrupted.condition_name == "threat"
