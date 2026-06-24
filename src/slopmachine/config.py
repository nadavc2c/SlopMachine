"""Config schemas and loaders for SlopMachine.

The model *registry* (``config/models.yaml``) is the single source of truth for
which model backs each capability. Code never hardcodes model ids — it asks the
registry. Style presets live in ``config/styles/*.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class SlopError(Exception):
    """A user-facing error (bad model/capability/style). The CLI prints it cleanly."""


def _project_root() -> Path:
    # src/slopmachine/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    env = os.environ.get("SLOP_CONFIG_DIR")
    return Path(env) if env else _project_root() / "config"


def outputs_dir() -> Path:
    env = os.environ.get("SLOP_OUTPUTS_DIR")
    d = Path(env) if env else _project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_cache_dir() -> Path:
    env = os.environ.get("SLOP_MODELS_DIR")
    d = Path(env) if env else _project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_hf_cache() -> Path:
    """Keep downloaded weights inside the repo (containment) unless HF_HOME is set.

    Must run before huggingface_hub / diffusers are imported. Respects a user-set
    HF_HOME so the shared cache can still be opted into.
    """
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(models_cache_dir())
    return Path(os.environ["HF_HOME"])


class ModelSpec(BaseModel):
    """One model (or adapter) entry in the registry."""

    repo_id: str
    precision: str = "bf16"          # bf16 | fp16 | fp32
    variant: Optional[str] = None     # e.g. "fp16" weight variant in the repo
    min_vram_gb: float = 8.0          # rough footprint; drives the load plan
    license: str = "unknown"
    tier: Optional[str] = None        # informational: "fast" | "quality" | ...
    pipeline: Optional[str] = None    # explicit diffusers pipeline class; else AutoPipeline
    subfolder: Optional[str] = None   # for adapters
    weight_name: Optional[str] = None # for adapters
    base_capability: Optional[str] = None  # adapter applies on top of this capability
    gen_defaults: dict = Field(default_factory=dict)
    notes: str = ""


class Capability(BaseModel):
    """A capability (e.g. "image") and its candidate models."""

    default: str
    models: dict[str, ModelSpec]

    def resolve(self, key: Optional[str] = None) -> tuple[str, ModelSpec]:
        chosen = key or self.default
        if chosen not in self.models:
            raise SlopError(
                f"Model '{chosen}' not found; available: {', '.join(self.models)}"
            )
        return chosen, self.models[chosen]


class Registry(BaseModel):
    capabilities: dict[str, Capability]

    def capability(self, name: str) -> Capability:
        if name not in self.capabilities:
            raise SlopError(
                f"Capability '{name}' not found; available: {', '.join(self.capabilities)}"
            )
        return self.capabilities[name]


class StylePreset(BaseModel):
    name: str
    positive: str = "{prompt}"
    negative: str = ""
    gen_defaults: dict = Field(default_factory=dict)

    def apply(self, prompt: str) -> tuple[str, str]:
        """Return (positive_prompt, negative_prompt) for the given user prompt."""
        return self.positive.format(prompt=prompt), self.negative


def load_registry(path: Optional[Path] = None) -> Registry:
    path = path or (config_dir() / "models.yaml")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Registry(**data)


def list_styles() -> list[str]:
    d = config_dir() / "styles"
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []


def load_style(name: str) -> StylePreset:
    path = config_dir() / "styles" / f"{name}.yaml"
    if not path.exists():
        raise SlopError(
            f"Style '{name}' not found at {path}. Available: {', '.join(list_styles())}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.setdefault("name", name)
    return StylePreset(**data)
