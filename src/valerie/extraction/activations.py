"""Standardized activation extraction and serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from valerie.config import ExtractionConfig
from valerie.extraction.hooks import cache_key_for
from valerie.models.loader import LoadedModel


def resolve_layers(layers: list[int] | str, n_layers: int) -> list[int]:
    if layers == "all":
        return list(range(n_layers))
    resolved = sorted(set(layers))
    for layer in resolved:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"requested layer {layer} outside valid range [0, {n_layers - 1}]")
    return resolved


def _select_positions(tensor: Any, component: str, strategy: str, index: int | None):
    if strategy == "all":
        return tensor.detach().cpu()
    if strategy == "last":
        if component == "attn_pattern" and tensor.ndim >= 4:
            return tensor[:, :, -1:, :].detach().cpu()
        if tensor.ndim < 2:
            return tensor.detach().cpu()
        return tensor[:, -1:, ...].detach().cpu()
    if strategy == "index":
        if component == "attn_pattern" and tensor.ndim >= 4:
            return tensor[:, :, index : index + 1, :].detach().cpu()
        if tensor.ndim < 2:
            return tensor.detach().cpu()
        return tensor[:, index : index + 1, ...].detach().cpu()
    if strategy == "mean_pool":
        if tensor.ndim < 2:
            return tensor.detach().cpu()
        if component == "attn_pattern" and tensor.ndim >= 4:
            return tensor.mean(dim=2, keepdim=True).detach().cpu()
        return tensor.mean(dim=1, keepdim=True).detach().cpu()
    raise ValueError(f"unsupported position strategy '{strategy}'")


def extract_requested_activations(
    model: LoadedModel,
    prompt: str,
    extraction_config: ExtractionConfig,
) -> dict[str, Any]:
    """Run a prompt and return a normalized activation payload."""
    logits, cache, tokens = model.run_with_cache(prompt)
    layers = resolve_layers(extraction_config.layers, model.n_layers)
    activations: dict[str, Any] = {}

    for component in extraction_config.components:
        for layer in layers:
            cache_key = cache_key_for(component, layer)
            if cache_key not in cache:
                available = ", ".join(sorted(cache.keys())[:10])
                raise KeyError(
                    f"cache key '{cache_key}' missing from model cache. "
                    f"Available keys include: {available}"
                )
            standardized_key = f"{component}.layer_{layer}"
            activations[standardized_key] = _select_positions(
                cache[cache_key],
                component,
                extraction_config.position.strategy,
                extraction_config.position.index,
            )

    result = {
        "prompt": prompt,
        "token_ids": tokens.detach().cpu(),
        "activations": activations,
    }
    if extraction_config.save_logits:
        result["logits"] = logits.detach().cpu()
    return result


def save_activation_payload(payload: dict[str, Any], destination: str | Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to serialize activation payloads.") from exc

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
