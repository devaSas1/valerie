from valerie.config import load_experiment_config
from valerie.experiments.framings import (
    build_framed_prompts,
    summarize_length_deltas,
    summarize_token_length_deltas,
)


def test_build_framed_prompts_counts():
    config = load_experiment_config("configs/experiments/threat-vs-care.yaml")
    prompts = build_framed_prompts(config)
    assert len(prompts) == len(config.tasks) * len(config.conditions) * 3


def test_length_delta_summary_is_present():
    config = load_experiment_config("configs/experiments/threat-vs-care.yaml")
    summary = summarize_length_deltas(config)
    assert set(summary) == {task.id for task in config.tasks}
    assert all("delta_chars" in values for values in summary.values())


def test_token_length_deltas_uses_provided_tokenizer():
    config = load_experiment_config("configs/experiments/threat-vs-care.yaml")
    # Use a simple whitespace tokenizer as a stand-in for a real model tokenizer.
    tokenize = lambda text: text.split()
    summary = summarize_token_length_deltas(config, tokenize)
    assert set(summary) == {task.id for task in config.tasks}
    for task_summary in summary.values():
        assert "min_tokens" in task_summary
        assert "max_tokens" in task_summary
        assert "delta_tokens" in task_summary
        assert task_summary["delta_tokens"] >= 0
        assert task_summary["max_tokens"] >= task_summary["min_tokens"]


def test_v2_config_loads_and_has_more_tasks():
    config = load_experiment_config("configs/experiments/threat-vs-care-v2.yaml")
    assert len(config.tasks) > 2
    assert "narrative_threat_control" in config.conditions
    # narrative control variants should not all share the same opener
    narrative_variants = config.conditions["narrative_threat_control"].variants
    openers = {v.split("\n")[0][:20] for v in narrative_variants}
    assert len(openers) > 1, "narrative control variants should have distinct openers"

