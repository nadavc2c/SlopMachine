"""Hardware detection + a memory "load plan" so VRAM-fitting logic lives in ONE place.

All techniques used here are stock torch + diffusers (no extra deps): dtype choice,
``enable_model_cpu_offload`` / ``enable_sequential_cpu_offload``, VAE tiling, and
torch-native fp8 layerwise casting. Quantization libraries are intentionally avoided
unless a specific model proves to need them.

torch is imported lazily so the CLI (e.g. ``slop models list``) starts without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _torch():
    import torch  # lazy
    return torch


@dataclass
class LoadPlan:
    dtype: object          # torch.dtype
    device: str            # "cuda" | "cpu"
    offload: str           # "none" | "model" | "sequential"
    fp8_layerwise: bool
    vae_tiling: bool

    def describe(self) -> str:
        bits = [f"dtype={str(self.dtype).replace('torch.', '')}", f"device={self.device}"]
        if self.offload != "none":
            bits.append(f"offload={self.offload}")
        if self.fp8_layerwise:
            bits.append("fp8-layerwise")
        if self.vae_tiling:
            bits.append("vae-tiling")
        return ", ".join(bits)


def cuda_available() -> bool:
    return _torch().cuda.is_available()


def device() -> str:
    return "cuda" if cuda_available() else "cpu"


def gpu_name() -> str:
    t = _torch()
    return t.cuda.get_device_name(0) if t.cuda.is_available() else "cpu"


def capability() -> Optional[tuple[int, int]]:
    t = _torch()
    return t.cuda.get_device_capability(0) if t.cuda.is_available() else None


def total_vram_gb() -> float:
    t = _torch()
    if not t.cuda.is_available():
        return 0.0
    return t.cuda.get_device_properties(0).total_memory / (1024**3)


def dtype_for(precision: str):
    t = _torch()
    return {"fp16": t.float16, "bf16": t.bfloat16, "fp32": t.float32}.get(precision, t.bfloat16)


def plan_for(spec, headroom_gb: float = 2.0) -> LoadPlan:
    """Pick a load strategy for a model spec given the current GPU.

    - Comfortable fit  -> full GPU, no offload.
    - Tight fit        -> model CPU offload + fp8 layerwise casting + VAE tiling.
    - Very tight fit   -> sequential CPU offload (slow but fits).
    """
    t = _torch()
    if not t.cuda.is_available():
        return LoadPlan(t.float32, "cpu", "none", False, False)

    vram = total_vram_gb()
    dtype = dtype_for(spec.precision)
    needs = float(getattr(spec, "min_vram_gb", 8.0))

    if vram >= needs + headroom_gb:
        return LoadPlan(dtype, "cuda", "none", False, True)
    if vram >= needs * 0.5:
        return LoadPlan(dtype, "cuda", "model", True, True)
    return LoadPlan(dtype, "cuda", "sequential", True, True)
