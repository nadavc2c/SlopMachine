"""ImageStage PAG gating + anti-slop negative baseline (no GPU / no network).

ImageStage.__init__ only resolves the registry spec + provider (no model load), so these are fast.
"""

import slopmachine.config as config
from slopmachine.pipeline.image import ImageStage


def test_pag_on_by_default_for_sdxl(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    assert ImageStage(model_key="sdxl").pag is True  # sdxl row sets supports_pag: true


def test_pag_disabled_by_flag(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    assert ImageStage(model_key="sdxl", pag=False).pag is False


def test_pag_off_for_mmdit_model(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    # flux-dev has no supports_pag flag -> diffusers PAG is auto-skipped even with pag=True.
    assert ImageStage(model_key="flux-dev", pag=True).pag is False


def test_pag_scale_is_stored():
    assert ImageStage(model_key="sdxl", pag_scale=4.5).pag_scale == 4.5


def test_default_negative_covers_anatomy():
    neg = config.DEFAULT_NEGATIVE.lower()
    assert "extra fingers" in neg
    assert "bad anatomy" in neg
    assert "extra limbs" in neg
