"""Activation patching for causal verification of framing representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from valerie.extraction.hooks import cache_key_for
from valerie.models.loader import LoadedModel


@dataclass(frozen=True)
class PairPatchingResult:
    """Patching result for a single clean/corrupted prompt pair at one layer."""

    layer: int
    component: str
    task_id: str
    variant_index: int
    clean_condition: str
    corrupted_condition: str
    # How much of the clean-vs-corrupted logit difference does patching recover?
    # 1.0 = patch perfectly moved corrupted toward clean. 0.0 = no effect.
    recovery_cosine: float
    # KL(clean_probs || corrupted_probs) — baseline divergence between conditions
    kl_baseline: float
    # KL(clean_probs || patched_probs) — divergence after patching
    kl_patched: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. Clips q to avoid log(0)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-10, None)
    return float(entropy(p, q))


def _make_last_token_patch_hook(clean_last_token: Any):
    """Returns a TransformerLens hook that replaces the last token residual with clean_last_token.

    clean_last_token should be a 1-D tensor of shape [d_model].
    The hook receives value of shape [batch, seq_len, d_model] and replaces position -1.
    """
    def hook_fn(value, hook):
        patched = value.clone()
        patched[:, -1, :] = clean_last_token
        return patched
    return hook_fn


def run_patch_experiment(
    model: LoadedModel,
    clean_prompt: str,
    corrupted_prompt: str,
    layers: list[int],
    component: str,
    clean_condition: str,
    corrupted_condition: str,
    task_id: str,
    variant_index: int,
) -> list[PairPatchingResult]:
    """Run activation patching for one clean/corrupted pair across all target layers.

    For each layer:
    - Run clean prompt, cache activations
    - Run corrupted prompt normally (baseline)
    - Run corrupted prompt with clean last-token activation patched in at that layer
    - Measure how much the output shifts toward the clean distribution

    Returns one PairPatchingResult per layer.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for activation patching.") from exc

    # --- clean run: get logits and full activation cache ---
    clean_logits, clean_cache, _ = model.run_with_cache(clean_prompt)
    clean_last_logits = clean_logits[0, -1, :].detach().cpu().numpy().astype(np.float64)
    clean_probs = softmax(clean_last_logits)

    # --- corrupted run: baseline logits ---
    corrupted_logits, _, _ = model.run_with_cache(corrupted_prompt)
    corrupted_last_logits = corrupted_logits[0, -1, :].detach().cpu().numpy().astype(np.float64)
    corrupted_probs = softmax(corrupted_last_logits)

    # direction from corrupted to clean in logit space
    target_direction = clean_last_logits - corrupted_last_logits
    kl_baseline = _kl_divergence(clean_probs, corrupted_probs)

    results: list[PairPatchingResult] = []

    for layer in layers:
        cache_key = cache_key_for(component, layer)
        if cache_key not in clean_cache:
            continue

        # last-token activation from clean run at this layer, shape [d_model]
        clean_last_token_activation = clean_cache[cache_key][0, -1, :].detach()

        hook_name = cache_key
        patched_logits = model.run_with_hooks(
            corrupted_prompt,
            fwd_hooks=[(hook_name, _make_last_token_patch_hook(clean_last_token_activation))],
        )
        patched_last_logits = patched_logits[0, -1, :].detach().cpu().numpy().astype(np.float64)
        patched_probs = softmax(patched_last_logits)

        patch_effect = patched_last_logits - corrupted_last_logits
        recovery_cosine = _cosine_similarity(target_direction, patch_effect)
        kl_patched = _kl_divergence(clean_probs, patched_probs)

        results.append(
            PairPatchingResult(
                layer=layer,
                component=component,
                task_id=task_id,
                variant_index=variant_index,
                clean_condition=clean_condition,
                corrupted_condition=corrupted_condition,
                recovery_cosine=recovery_cosine,
                kl_baseline=kl_baseline,
                kl_patched=kl_patched,
            )
        )

    return results
