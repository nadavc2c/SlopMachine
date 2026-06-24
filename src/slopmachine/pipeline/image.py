"""Image generation stage (text->image, with optional identity preservation).

Model-agnostic: it loads whatever the registry resolves to via diffusers'
``AutoPipelineForText2Image`` and only relies on the generic pipeline contract.
Call kwargs are filtered against the actual pipeline signature, so the same code
works across SDXL (takes negative_prompt) and FLUX (does not), etc.

torch/diffusers are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from .. import config, hardware
from ..registry import get_model
from .base import Stage


def _accepted_kwargs(pipe, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the pipeline's __call__ doesn't accept (keeps it model-agnostic)."""
    params = inspect.signature(pipe.__call__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


class ImageStage(Stage):
    name = "image"

    def __init__(self, model_key: Optional[str] = None, provider: Optional[str] = None):
        self.model_key, self.spec = get_model("image", model_key)
        # Resolve + GATE the provider up front: raises SlopError if a paid cloud backend is selected
        # without the SLOP_ALLOW_CLOUD opt-in (and a token). Default is the free local backend.
        self.provider = config.resolve_provider(self.spec, provider)
        self._pipe = None
        self._plan = None
        self._identity_loaded = False

    def load(self):
        if self._pipe is not None:
            return self._pipe
        from diffusers import AutoPipelineForText2Image

        plan = hardware.plan_for(self.spec)
        kwargs: dict[str, Any] = {"torch_dtype": plan.dtype}
        if self.spec.variant:
            kwargs["variant"] = self.spec.variant
        if self.spec.pipeline:
            # Explicit pipeline class from the registry (e.g. QwenImagePipeline) when AutoPipeline can't map it.
            import diffusers

            pipe = getattr(diffusers, self.spec.pipeline).from_pretrained(self.spec.repo_id, **kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(self.spec.repo_id, **kwargs)

        if plan.offload == "model":
            pipe.enable_model_cpu_offload()
        elif plan.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(plan.device)

        if plan.vae_tiling:
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                try:
                    vae.enable_tiling()  # current diffusers API (pipe.enable_vae_tiling is deprecated)
                except Exception:
                    pass

        self._pipe, self._plan = pipe, plan
        return pipe

    def load_identity(self, adapter_key: Optional[str] = None):
        """Load an identity adapter (e.g. IP-Adapter) onto the pipeline."""
        pipe = self.load()
        _, adapter = get_model("identity", adapter_key)
        pipe.load_ip_adapter(
            adapter.repo_id,
            subfolder=adapter.subfolder,
            weight_name=adapter.weight_name,
        )
        self._identity_loaded = True

    def run(
        self,
        prompt: str,
        negative_prompt: str = "",
        identity_image: Optional[str] = None,
        identity_scale: float = 0.7,
        seed: Optional[int] = None,
        out: Optional[Path] = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        out_path = Path(out) if out else None

        # Remote/paid provider (already gated in __init__): call the backend, skip diffusers entirely.
        if self.provider != "local":
            from .backends import generate_image

            image = generate_image(
                self.provider, self.spec, prompt, negative_prompt=negative_prompt,
                **{k: v for k, v in gen_kwargs.items() if v is not None},
            )
            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(out_path)
            return {"image": image, "path": out_path, "model": self.model_key, "provider": self.provider}

        # Local diffusers path.
        import torch
        from diffusers.utils import load_image

        pipe = self.load()

        params = dict(self.spec.gen_defaults)
        params.update({k: v for k, v in gen_kwargs.items() if v is not None})

        full = dict(params)
        full["prompt"] = prompt
        if negative_prompt:
            full["negative_prompt"] = negative_prompt
        if seed is not None:
            full["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        if identity_image:
            if not self._identity_loaded:
                self.load_identity()
            full["ip_adapter_image"] = load_image(identity_image)
            pipe.set_ip_adapter_scale(identity_scale)

        image = pipe(**_accepted_kwargs(pipe, full)).images[0]

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
        return {"image": image, "path": out_path, "model": self.model_key, "provider": "local"}
