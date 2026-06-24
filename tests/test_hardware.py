"""Unit-test the VRAM load-plan logic with a faked torch (no GPU required)."""

from types import SimpleNamespace

import slopmachine.hardware as hardware


class _FakeCuda:
    def __init__(self, gb):
        self._gb = gb

    def is_available(self):
        return True

    def get_device_properties(self, _idx):
        return SimpleNamespace(total_memory=int(self._gb * 1024**3))


def _fake_torch(gb):
    return SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        cuda=_FakeCuda(gb),
    )


def _spec(min_vram_gb, precision="fp16"):
    return SimpleNamespace(min_vram_gb=min_vram_gb, precision=precision)


def test_comfortable_fit_no_offload(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(16))
    plan = hardware.plan_for(_spec(8))
    assert plan.offload == "none"
    assert plan.device == "cuda"
    assert plan.vae_tiling is True


def test_tight_fit_uses_offload(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(16))
    plan = hardware.plan_for(_spec(16))
    assert plan.offload == "model"
    assert plan.vae_tiling is True


def test_very_tight_fit_sequential(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(16))
    plan = hardware.plan_for(_spec(40))
    assert plan.offload == "sequential"
