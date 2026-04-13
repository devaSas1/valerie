"""Prompt framing utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from valerie.config import ExperimentConfig, TaskConfig


@dataclass(frozen=True)
class FramedPrompt:
    sample_id: str
    task_id: str
    condition_name: str
    condition_target: str
    variant_index: int
    prompt: str
    char_length: int


def _render_variant(template: str, task: TaskConfig) -> str:
    return template.format(task=task.prompt, task_id=task.id)


def build_framed_prompts(experiment_config: ExperimentConfig) -> list[FramedPrompt]:
    prompts: list[FramedPrompt] = []
    for task in experiment_config.tasks:
        for condition_name, condition in experiment_config.conditions.items():
            for variant_index, variant_template in enumerate(condition.variants):
                prompt = _render_variant(variant_template, task)
                sample_id = f"{task.id}__{condition_name}__variant_{variant_index}"
                prompts.append(
                    FramedPrompt(
                        sample_id=sample_id,
                        task_id=task.id,
                        condition_name=condition_name,
                        condition_target=condition.target,
                        variant_index=variant_index,
                        prompt=prompt,
                        char_length=len(prompt),
                    )
                )
    return prompts


def summarize_length_deltas(experiment_config: ExperimentConfig) -> dict[str, dict[str, int]]:
    """Compute per-task prompt character-length deltas across framing conditions."""
    summaries: dict[str, dict[str, int]] = {}
    for task in experiment_config.tasks:
        lengths_by_condition = {
            condition_name: [
                len(_render_variant(variant_template, task))
                for variant_template in condition.variants
            ]
            for condition_name, condition in experiment_config.conditions.items()
        }
        all_lengths = [length for values in lengths_by_condition.values() for length in values]
        summaries[task.id] = {
            "min_chars": min(all_lengths),
            "max_chars": max(all_lengths),
            "delta_chars": max(all_lengths) - min(all_lengths),
        }
    return summaries


def summarize_token_length_deltas(
    experiment_config: ExperimentConfig,
    tokenize: Callable[[str], list[int]],
) -> dict[str, dict[str, int]]:
    """Compute per-task prompt token-length deltas across framing conditions.

    Requires a callable that maps a prompt string to a list of token ids.
    Use the tokenizer from the loaded model backend to ensure token counts
    match what the model actually sees during inference.
    """
    summaries: dict[str, dict[str, int]] = {}
    for task in experiment_config.tasks:
        lengths_by_condition = {
            condition_name: [
                len(tokenize(_render_variant(variant_template, task)))
                for variant_template in condition.variants
            ]
            for condition_name, condition in experiment_config.conditions.items()
        }
        all_lengths = [length for values in lengths_by_condition.values() for length in values]
        summaries[task.id] = {
            "min_tokens": min(all_lengths),
            "max_tokens": max(all_lengths),
            "delta_tokens": max(all_lengths) - min(all_lengths),
        }
    return summaries

