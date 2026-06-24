"""Config schemas and loaders for SlopMachine.

The model *registry* (``src/slopmachine/config/models.yaml``) is the single source of truth for
which model backs each capability. Code never hardcodes model ids — it asks the
registry. Style presets live in ``src/slopmachine/config/styles/*.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class SlopError(Exception):
    """A user-facing error (bad model/capability/style). The CLI prints it cleanly."""


def _package_dir() -> Path:
    """The installed slopmachine package dir — resolves whether editable or wheel-installed,
    so the bundled config/ (registry + styles) is found without needing the source repo."""
    import importlib.resources

    return Path(str(importlib.resources.files("slopmachine")))


def config_dir() -> Path:
    """Read-only registry + style presets, shipped as package data. Override with SLOP_CONFIG_DIR."""
    env = os.environ.get("SLOP_CONFIG_DIR")
    return Path(env) if env else _package_dir() / "config"


def outputs_dir() -> Path:
    """Where generated artifacts land. Defaults to ./outputs in the working dir; SLOP_OUTPUTS_DIR overrides."""
    env = os.environ.get("SLOP_OUTPUTS_DIR")
    d = Path(env) if env else Path.cwd() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_cache_dir() -> Path:
    """Local model cache (./models in the working dir). Override SLOP_MODELS_DIR, or set HF_HOME directly."""
    env = os.environ.get("SLOP_MODELS_DIR")
    d = Path(env) if env else Path.cwd() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_hf_cache() -> Path:
    """Keep downloaded weights in the local models/ dir (working dir or SLOP_MODELS_DIR) unless HF_HOME is set.

    Must run before huggingface_hub / diffusers are imported. Respects a user-set HF_HOME so the
    shared cache can still be opted into.
    """
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(models_cache_dir())
    return Path(os.environ["HF_HOME"])


# --- Providers / cloud opt-in gate ------------------------------------------------------------
# Remote providers cost money. They are OFF by default: resolve_provider() refuses them unless the
# human has set SLOP_ALLOW_CLOUD *and* a token is present — so no code path can spend by accident.
_REMOTE_PROVIDERS = ("hf-inference", "google-genai")
_PROVIDER_TOKENS = {
    "hf-inference": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "google-genai": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def cloud_allowed() -> bool:
    """True only if the human opted into paid cloud calls via SLOP_ALLOW_CLOUD."""
    return os.environ.get("SLOP_ALLOW_CLOUD", "").strip().lower() in ("1", "true", "yes", "on")


def provider_token(provider: str) -> Optional[str]:
    """Return the first set token env value for a remote provider, or None."""
    for env in _PROVIDER_TOKENS.get(provider, ()):
        value = os.environ.get(env)
        if value:
            return value
    return None


def resolve_provider(spec, cli_provider: Optional[str] = None) -> str:
    """Pick the provider (cli > SLOP_PROVIDER > spec.provider > 'local') and ENFORCE the cloud gate.

    Single choke point: a remote/paid provider is refused unless SLOP_ALLOW_CLOUD is set AND a token
    is present, so nothing spends money without an explicit human opt-in.
    """
    provider = (
        cli_provider or os.environ.get("SLOP_PROVIDER") or getattr(spec, "provider", None) or "local"
    ).strip()
    if provider == "local":
        return provider
    if provider not in _REMOTE_PROVIDERS:
        raise SlopError(f"Unknown provider '{provider}'. Known: local, {', '.join(_REMOTE_PROVIDERS)}.")
    if not cloud_allowed():
        raise SlopError(
            f"Provider '{provider}' is a paid cloud backend, disabled by default to prevent accidental "
            f"token spend. Opt in with SLOP_ALLOW_CLOUD=1 (and set {' or '.join(_PROVIDER_TOKENS[provider])})."
        )
    if not provider_token(provider):
        raise SlopError(
            f"Provider '{provider}' needs an API token; set one of: {', '.join(_PROVIDER_TOKENS[provider])}."
        )
    return provider


class ModelSpec(BaseModel):
    """One model (or adapter) entry in the registry."""

    repo_id: str
    precision: str = "bf16"          # bf16 | fp16 | fp32
    variant: Optional[str] = None     # e.g. "fp16" weight variant in the repo
    min_vram_gb: float = 8.0          # rough footprint; drives the load plan
    license: str = "unknown"
    tier: Optional[str] = None        # informational: "fast" | "quality" | ...
    provider: str = "local"           # "local" (diffusers) | "hf-inference" | "google-genai"
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


def load_assets() -> dict:
    """The on-demand asset catalog (dance clips, etc.) from config/assets.yaml. Empty dict if absent."""
    path = config_dir() / "assets.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
