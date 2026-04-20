"""Model loading and runtime abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from valerie.config import ModelConfig
from valerie.models.registry import get_backend, register_backend


class DependencyUnavailableError(ImportError):
    """Raised when an optional runtime dependency is missing."""


@dataclass
class LoadedModel:
    """Thin wrapper around a backend-specific runtime."""

    config: ModelConfig
    runtime: Any
    device: str
    n_layers: int

    def run_with_cache(self, prompt: str) -> tuple[Any, dict[str, Any], Any]:
        """Run a prompt and return logits, activation cache, and token ids."""
        return self.runtime.run_with_cache(prompt)

    def tokenize(self, text: str) -> list[int] | None:
        """Return token ids for text, or None if the backend has no real tokenizer."""
        return self.runtime.tokenize(text)

    def run_with_hooks(self, prompt: str, fwd_hooks: list) -> Any:
        """Run a prompt with mid-forward-pass hooks. Returns logits."""
        return self.runtime.run_with_hooks(prompt, fwd_hooks)


def _select_torch_device(device_preference: list[str]) -> str:
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailableError(
            "PyTorch is required to select a runtime device. Install project dependencies first."
        ) from exc

    availability = {
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    for device in device_preference:
        if availability[device]:
            return device
    return "cpu"


class DummyRuntime:
    """Deterministic fake runtime for tests and pipeline smoke checks."""

    def __init__(self, config: ModelConfig, device: str):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PyTorch is required even for the dummy backend."
            ) from exc

        self.torch = torch
        self.config = config
        self.device = device
        self.n_layers = config.dummy_n_layers

    def _encode(self, prompt: str):
        token_values = [max(1, ord(char) % 255) for char in prompt]
        return self.torch.tensor([token_values], dtype=self.torch.long, device=self.device)

    def run_with_cache(self, prompt: str):
        tokens = self._encode(prompt)
        token_float = tokens.to(self.torch.float32).unsqueeze(-1)
        d_model = self.config.dummy_d_model
        offsets = self.torch.arange(d_model, device=self.device, dtype=self.torch.float32).view(
            1, 1, d_model
        )

        cache: dict[str, Any] = {}
        for layer in range(self.n_layers):
            layer_scale = float(layer + 1)
            resid = token_float * layer_scale / 255.0 + offsets / max(1, d_model)
            mlp_pre = resid[..., : d_model // 2]
            mlp_post = self.torch.tanh(mlp_pre)
            head_result = resid.unsqueeze(2)
            attn_pattern = self.torch.softmax(
                self.torch.ones(
                    (1, 1, tokens.shape[1], tokens.shape[1]),
                    device=self.device,
                    dtype=self.torch.float32,
                ),
                dim=-1,
            )
            cache[f"blocks.{layer}.hook_resid_pre"] = resid
            cache[f"blocks.{layer}.hook_resid_mid"] = resid + 0.1
            cache[f"blocks.{layer}.hook_resid_post"] = resid + 0.2
            cache[f"blocks.{layer}.mlp.hook_pre"] = mlp_pre
            cache[f"blocks.{layer}.mlp.hook_post"] = mlp_post
            cache[f"blocks.{layer}.attn.hook_result"] = head_result
            cache[f"blocks.{layer}.attn.hook_pattern"] = attn_pattern

        vocab_size = 32
        logits = self.torch.zeros((1, tokens.shape[1], vocab_size), device=self.device)
        logits.scatter_(-1, (tokens % vocab_size).unsqueeze(-1), 1.0)
        return logits, cache, tokens

    def tokenize(self, text: str) -> list[int] | None:
        return None

    def run_with_hooks(self, prompt: str, fwd_hooks: list) -> Any:
        logits, _, _ = self.run_with_cache(prompt)
        return logits


class TransformerLensRuntime:
    """Adapter around HookedTransformer."""

    def __init__(self, config: ModelConfig, device: str):
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise DependencyUnavailableError(
                "TransformerLens and PyTorch are required for the transformer_lens backend."
            ) from exc

        dtype = getattr(torch, config.dtype)
        load_kwargs: dict[str, Any] = {
            "device": device,
            "dtype": dtype,
            "default_prepend_bos": config.default_prepend_bos,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.cache_dir:
            load_kwargs["cache_dir"] = config.cache_dir
        if config.first_n_layers is not None:
            load_kwargs["first_n_layers"] = config.first_n_layers
        if config.quantization.mode == "4bit":
            load_kwargs["load_in_4bit"] = True
        elif config.quantization.mode == "8bit":
            load_kwargs["load_in_8bit"] = True

        self.model = HookedTransformer.from_pretrained(config.model_name, **load_kwargs)
        self.device = device
        self.n_layers = int(self.model.cfg.n_layers)
        self.default_prepend_bos = config.default_prepend_bos

    def run_with_cache(self, prompt: str):
        logits, cache = self.model.run_with_cache(
            prompt,
            prepend_bos=self.default_prepend_bos,
            return_cache_object=False,
        )
        tokens = self.model.to_tokens(prompt, prepend_bos=self.default_prepend_bos)
        return logits, cache, tokens

    def tokenize(self, text: str) -> list[int]:
        tokens = self.model.to_tokens(text, prepend_bos=self.default_prepend_bos)
        return tokens[0].tolist()

    def run_with_hooks(self, prompt: str, fwd_hooks: list) -> Any:
        return self.model.run_with_hooks(
            prompt,
            fwd_hooks=fwd_hooks,
            prepend_bos=self.default_prepend_bos,
        )


def _build_dummy_runtime(config: ModelConfig) -> LoadedModel:
    device = _select_torch_device(config.device_preference)
    runtime = DummyRuntime(config, device=device)
    return LoadedModel(config=config, runtime=runtime, device=device, n_layers=runtime.n_layers)


def _build_transformer_lens_runtime(config: ModelConfig) -> LoadedModel:
    device = _select_torch_device(config.device_preference)
    runtime = TransformerLensRuntime(config, device=device)
    return LoadedModel(config=config, runtime=runtime, device=device, n_layers=runtime.n_layers)


register_backend("dummy", _build_dummy_runtime)
register_backend("transformer_lens", _build_transformer_lens_runtime)


def load_model(config: ModelConfig) -> LoadedModel:
    """Load a model runtime from config."""
    factory = get_backend(config.backend)
    return factory(config)
