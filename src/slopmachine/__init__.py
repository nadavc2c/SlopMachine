"""SlopMachine — modular, CLI-driven local generative-AI toolkit.

A thin, model-agnostic orchestration layer over HuggingFace `diffusers`,
wrapped in a `slop` CLI. Models are resolved at runtime from a registry
(`src/slopmachine/config/models.yaml`), never hardcoded here.
"""

__version__ = "0.1.0"
