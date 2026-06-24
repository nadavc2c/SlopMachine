import pytest

from slopmachine.config import SlopError
from slopmachine.registry import get_model, list_capabilities, list_models, registry


def test_registry_parses():
    reg = registry(reload=True)
    assert "image" in reg.capabilities
    assert "identity" in reg.capabilities


def test_default_image_model():
    key, spec = get_model("image")
    assert key == "sdxl"
    assert spec.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"


def test_explicit_model():
    key, spec = get_model("image", "flux-dev")
    assert key == "flux-dev"
    assert "FLUX" in spec.repo_id


def test_unknown_model_raises():
    with pytest.raises(SlopError):
        get_model("image", "does-not-exist")


def test_unknown_capability_raises():
    with pytest.raises(SlopError):
        get_model("bogus")


def test_list_helpers():
    assert "image" in list_capabilities()
    assert "sdxl" in list_models("image")
