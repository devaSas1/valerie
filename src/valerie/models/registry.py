"""Backend registry for Valerie model runtimes."""

from __future__ import annotations

from collections.abc import Callable

MODEL_BACKENDS: dict[str, Callable] = {}


def register_backend(name: str, factory: Callable) -> None:
    MODEL_BACKENDS[name] = factory


def get_backend(name: str) -> Callable:
    try:
        return MODEL_BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_BACKENDS))
        raise KeyError(f"unknown model backend '{name}'. Available backends: {available}") from exc

