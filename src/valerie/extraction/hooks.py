"""Mapping between Valerie component names and backend cache keys."""

from __future__ import annotations

COMPONENT_TO_CACHE_KEY = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "mlp_pre": "blocks.{layer}.mlp.hook_pre",
    "mlp_post": "blocks.{layer}.mlp.hook_post",
    "attn_pattern": "blocks.{layer}.attn.hook_pattern",
    "head_result": "blocks.{layer}.attn.hook_result",
}


def cache_key_for(component: str, layer: int) -> str:
    try:
        template = COMPONENT_TO_CACHE_KEY[component]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPONENT_TO_CACHE_KEY))
        raise KeyError(f"unsupported component '{component}'. Supported: {supported}") from exc
    return template.format(layer=layer)

