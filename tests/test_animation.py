"""Unit tests for the animation (AI-dance) capability.

All no-GPU / no-weights: they exercise registry wiring, the VRAM load plan, the
resolution math, and the lean pose-asset download patterns with fakes/monkeypatch.
"""

from types import SimpleNamespace

import slopmachine.hardware as hardware
from slopmachine.registry import get_model


# --- faked torch (same approach as test_hardware.py) ----------------------------------

class _FakeCuda:
    def __init__(self, gb):
        self._gb = gb

    def is_available(self):
        return True

    def get_device_properties(self, _idx):
        return SimpleNamespace(total_memory=int(self._gb * 1024**3))


def _fake_torch(gb):
    return SimpleNamespace(
        float16="float16", bfloat16="bfloat16", float32="float32", cuda=_FakeCuda(gb)
    )


# --- registry wiring -------------------------------------------------------------------

def test_animation_capability_resolves():
    key, spec = get_model("animation")
    assert key == "wan-animate"
    assert spec.pipeline == "WanAnimatePipeline"
    assert "Wan2.2-Animate" in spec.repo_id
    assert spec.gen_defaults.get("mode") == "animate"


def test_pose_capability_resolves():
    key, spec = get_model("pose")
    assert key == "wan-vitpose"
    assert spec.subfolder == "process_checkpoint"


def test_stages_instantiate():
    from slopmachine.pipeline.animation import AnimateStage
    from slopmachine.pipeline.pose import PoseStage

    assert AnimateStage().name == "animation"
    assert PoseStage().name == "pose"


# --- VRAM plan -------------------------------------------------------------------------

def test_animation_plan_uses_offload_and_fp8_on_16gb(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(16))
    _, spec = get_model("animation")  # min_vram_gb == 16
    plan = hardware.plan_for(spec)
    assert plan.offload == "model"
    assert plan.fp8_layerwise is True
    assert plan.vae_tiling is True


# --- resolution math (mirrors the official aspect_ratio_resize) ------------------------

def test_fit_resolution_is_grid_aligned():
    from PIL import Image

    from slopmachine.pipeline.animation import AnimateStage

    pipe = SimpleNamespace(
        vae_scale_factor_spatial=8,
        transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
    )
    img = Image.new("RGB", (1000, 500))
    resized, height, width = AnimateStage._fit_resolution(img, pipe, 480 * 832)
    mod = 8 * 2  # vae_scale_factor_spatial * patch_size[1]
    assert height % mod == 0 and width % mod == 0
    assert height > 0 and width > 0
    assert resized.size == (width, height)


# --- lean pose-asset download (det + pose2d only, never SAM2/FLUX/whole repo) -----------

def test_ensure_pose_assets_fetches_only_det_and_pose(monkeypatch):
    import slopmachine.pipeline.pose as pose_mod

    captured = {}

    def fake_snapshot_download(repo_id, allow_patterns=None):
        captured["repo_id"] = repo_id
        captured["allow_patterns"] = allow_patterns
        return "/fake/cache"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    _, spec = get_model("pose")
    result = pose_mod.ensure_pose_assets(spec)

    assert captured["repo_id"] == spec.repo_id
    assert captured["allow_patterns"] == [
        "process_checkpoint/det/*",
        "process_checkpoint/pose2d/*",
    ]
    assert str(result).endswith("process_checkpoint")
