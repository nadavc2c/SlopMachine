"""Pose-driven character animation stage — the "AI dance" flagship (motion transfer).

Wraps whatever video-animation pipeline the registry resolves to (default:
diffusers' ``WanAnimatePipeline`` / Wan2.2-Animate). Given a reference person image
plus a *preprocessed* pose-skeleton video and face video, it renders that person
performing the driving motion.

Model-agnostic: the pipeline class is named in the registry (``spec.pipeline``) and
looked up on ``diffusers``; call kwargs are filtered against the pipeline signature.
The driving-video -> pose/face preprocessing is NOT done here — it is ``PoseStage``'s
job (the diffusers pipeline expects already-preprocessed inputs).

torch/diffusers are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .. import hardware
from ..registry import get_model
from .base import Stage
from .image import _accepted_kwargs  # same model-agnostic kwarg filtering as the image stage


class AnimateStage(Stage):
    name = "animation"

    def __init__(self, model_key: Optional[str] = None):
        self.model_key, self.spec = get_model("animation", model_key)
        self._pipe = None
        self._plan = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        import diffusers
        import torch

        plan = hardware.plan_for(self.spec)

        # Pipeline class is registry-driven (no hardcoded model class in code).
        pipe_cls = getattr(diffusers, self.spec.pipeline) if self.spec.pipeline else diffusers.DiffusionPipeline
        pipe = pipe_cls.from_pretrained(self.spec.repo_id, torch_dtype=plan.dtype)

        # Wan's VAE is numerically happier in fp32 (the sanctioned replacement for the
        # deprecated upcast_vae). Cheap — the VAE is tiny (~0.5GB).
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            try:
                vae.to(torch.float32)
            except Exception:
                pass

        # OFFICIAL diffusers low-VRAM recipe for big video models (docs: optimization/memory):
        # GROUP OFFLOADING in bf16. CRITICAL: do NOT call pipe.to("cuda") afterwards — that pulls the
        # full ~32B transformer onto the GPU and OOMs; the group-offload hooks manage device placement
        # and stream layers from CPU. fp8 layerwise casting is intentionally skipped: it fails on
        # Blackwell sm_120 + torch cu128 (CUBLAS_STATUS_INTERNAL_ERROR -> slow unfused fallback).
        if plan.offload != "none":
            from diffusers.hooks.group_offloading import apply_group_offloading

            onload, offload_dev = torch.device("cuda"), torch.device("cpu")
            # Heavy denoiser(s): leaf-level streaming with CUDA-stream prefetch.
            for attr in ("transformer", "transformer_2"):
                mod = getattr(pipe, attr, None)
                if mod is not None and hasattr(mod, "enable_group_offload"):
                    mod.enable_group_offload(
                        onload_device=onload, offload_device=offload_dev,
                        offload_type="leaf_level", use_stream=True,
                        # On-the-fly pinning: pre-pinning the whole ~64GB transformer needs ~2x model
                        # size in pinned host RAM (we have 64GB) and OOMs. Required on this box.
                        low_cpu_mem_usage=True,
                    )
            # Text encoder: block-level group offloading.
            text_encoder = getattr(pipe, "text_encoder", None)
            if text_encoder is not None:
                apply_group_offloading(
                    text_encoder, onload_device=onload, offload_device=offload_dev,
                    offload_type="block_level", num_blocks_per_group=4,
                )
            # Small components stay resident on the GPU (cheap): VAE + image encoder.
            for attr in ("vae", "image_encoder"):
                comp = getattr(pipe, attr, None)
                if comp is not None:
                    try:
                        comp.to(onload)
                    except Exception:
                        pass
        else:
            pipe.to(plan.device)
        # Note: AutoencoderKLWan does not support VAE tiling/slicing (diffusers memory docs) and is
        # tiny (~0.5GB), so no tiling is applied.

        self._pipe, self._plan = pipe, plan
        return pipe

    @staticmethod
    def _fit_resolution(image, pipe, max_area: int) -> tuple[Any, int, int]:
        """Pick H/W from the reference aspect ratio, snapped to the model's grid.

        Mirrors the aspect_ratio_resize helper in the official Wan-Animate docs.
        """
        aspect = image.height / image.width
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = max(mod, round((max_area * aspect) ** 0.5) // mod * mod)
        width = max(mod, round((max_area / aspect) ** 0.5) // mod * mod)
        return image.resize((width, height)), height, width

    def run(
        self,
        reference_image: Any,
        pose_video: Any,
        face_video: Any,
        prompt: str = "",
        negative_prompt: str = "",
        fps: Optional[int] = None,
        seed: Optional[int] = None,
        out: Optional[Path] = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """Render the animation.

        ``reference_image`` / ``pose_video`` / ``face_video`` may be paths (loaded here)
        or already-loaded objects (e.g. threaded in from ImageStage / PoseStage by the runner).
        """
        import torch
        from diffusers.utils import export_to_video, load_image, load_video

        pipe = self.load()

        params = dict(self.spec.gen_defaults)
        params.update({k: v for k, v in gen_kwargs.items() if v is not None})

        # Non-pipeline knobs we consume ourselves (so they aren't passed to __call__).
        max_area = int(params.pop("max_area", 480 * 832))
        out_fps = int(fps if fps is not None else params.pop("fps", 30))
        params.pop("fps", None)

        image = load_image(reference_image) if isinstance(reference_image, str) else reference_image
        image, height, width = self._fit_resolution(image, pipe, max_area)

        full: dict[str, Any] = dict(params)
        full.update(
            {
                "image": image,
                "pose_video": load_video(pose_video) if isinstance(pose_video, str) else pose_video,
                "face_video": load_video(face_video) if isinstance(face_video, str) else face_video,
                "height": height,
                "width": width,
                # Wan-Animate needs a (text) prompt even with CFG off; keep it neutral by default.
                "prompt": prompt or "a person",
            }
        )
        if negative_prompt:
            full["negative_prompt"] = negative_prompt
        if seed is not None:
            full["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        frames = pipe(**_accepted_kwargs(pipe, full)).frames[0]

        out_path = Path(out) if out else None
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(out_path), fps=out_fps)
        return {"video": frames, "path": out_path, "model": self.model_key, "fps": out_fps}
