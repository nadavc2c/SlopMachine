"""Hardware detection + a memory "load plan" so VRAM-fitting logic lives in ONE place.

All techniques used here are stock torch + diffusers (no extra deps): dtype choice,
``enable_model_cpu_offload`` / ``enable_sequential_cpu_offload``, and VAE tiling.
Quantization libraries are intentionally avoided unless a specific model proves to need them.
(No fp8: layerwise fp8 casting is broken on Blackwell sm_120 — CUBLAS_STATUS_INTERNAL_ERROR — see CLAUDE.md.)

torch is imported lazily so the CLI (e.g. ``slop models list``) starts without it.
"""

from dataclasses import dataclass


def _torch():
    import torch  # lazy
    return torch


@dataclass
class LoadPlan:
    dtype: object          # torch.dtype
    device: str            # "cuda" | "cpu" | "mps"
    offload: str           # "none" | "model" | "sequential"
    vae_tiling: bool

    def describe(self) -> str:
        bits = [f"dtype={str(self.dtype).replace('torch.', '')}", f"device={self.device}"]
        if self.offload != "none":
            bits.append(f"offload={self.offload}")
        if self.vae_tiling:
            bits.append("vae-tiling")
        return ", ".join(bits)


def cuda_available() -> bool:
    return _torch().cuda.is_available()


def _mps_available(t) -> bool:
    backends = getattr(t, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    return bool(mps and mps.is_available())


def device() -> str:
    """Best available compute device: CUDA (tuned default) -> Apple MPS -> CPU."""
    t = _torch()
    if t.cuda.is_available():
        return "cuda"
    if _mps_available(t):
        return "mps"
    return "cpu"


def gpu_name() -> str:
    t = _torch()
    if t.cuda.is_available():
        return t.cuda.get_device_name(0)
    if _mps_available(t):
        return "Apple MPS"
    return "cpu"


def capability() -> tuple[int, int] | None:
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
    - Tight fit        -> model CPU offload + VAE tiling.
    - Very tight fit   -> sequential CPU offload (slow but fits).

    No fp8: layerwise fp8 casting raises CUBLAS_STATUS_INTERNAL_ERROR on Blackwell sm_120 (CLAUDE.md);
    big models fit with bf16 + offload + VAE tiling instead.
    """
    t = _torch()
    dev = device()
    if dev == "cpu":
        return LoadPlan(t.float32, "cpu", "none", False)
    if dev == "mps":
        # Apple MPS: fp16, full on-device (the CUDA-specific offload tricks don't apply); VAE tiling.
        return LoadPlan(t.float16, "mps", "none", True)

    # CUDA (the tuned default).
    vram = total_vram_gb()
    dtype = dtype_for(spec.precision)
    needs = float(getattr(spec, "min_vram_gb", 8.0))

    if vram >= needs + headroom_gb:
        return LoadPlan(dtype, "cuda", "none", True)
    if vram >= needs * 0.5:
        return LoadPlan(dtype, "cuda", "model", True)
    return LoadPlan(dtype, "cuda", "sequential", True)
