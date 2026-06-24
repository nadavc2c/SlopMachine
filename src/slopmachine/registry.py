"""Thin accessors over the model registry (src/slopmachine/config/models.yaml).

Capability -> resolved ModelSpec. This is the only place the rest of the code
goes to find out *which* model to use, so swapping models never touches code.
"""

from __future__ import annotations

from typing import Optional

from .config import ModelSpec, Registry, load_registry

_cache: Optional[Registry] = None


def registry(reload: bool = False) -> Registry:
    global _cache
    if _cache is None or reload:
        _cache = load_registry()
    return _cache


def get_model(capability: str, key: Optional[str] = None) -> tuple[str, ModelSpec]:
    """Resolve (model_key, spec) for a capability; key=None uses the default."""
    return registry().capability(capability).resolve(key)


def list_capabilities() -> list[str]:
    return sorted(registry().capabilities)


def list_models(capability: str) -> dict[str, ModelSpec]:
    return registry().capability(capability).models
