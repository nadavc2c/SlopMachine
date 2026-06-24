"""Provider resolution + cloud opt-in gate + device generalization (no GPU / no network).

The cloud gate is safety-critical (no $ spend without an explicit human opt-in), so it is tested at
the config level AND end-to-end at the ImageStage level.
"""

from types import SimpleNamespace

import pytest

import slopmachine.config as config
import slopmachine.hardware as hardware
from slopmachine.registry import get_model


def _spec(provider="local"):
    return SimpleNamespace(provider=provider)


# --- cloud opt-in gate -----------------------------------------------------------------

def test_default_provider_is_local(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.delenv("SLOP_ALLOW_CLOUD", raising=False)
    assert config.resolve_provider(_spec("local")) == "local"


def test_remote_refused_without_allow(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.delenv("SLOP_ALLOW_CLOUD", raising=False)
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(config.SlopError):
        config.resolve_provider(_spec("hf-inference"))


def test_remote_refused_without_token(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.setenv("SLOP_ALLOW_CLOUD", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(config.SlopError):
        config.resolve_provider(_spec("hf-inference"))


def test_remote_allowed_with_gate_and_token(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.setenv("SLOP_ALLOW_CLOUD", "1")
    monkeypatch.setenv("HF_TOKEN", "x")
    assert config.resolve_provider(_spec("hf-inference")) == "hf-inference"


def test_cli_provider_overrides_remote_spec_to_local(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.delenv("SLOP_ALLOW_CLOUD", raising=False)
    assert config.resolve_provider(_spec("hf-inference"), "local") == "local"


def test_slop_provider_env_selects_cloud(monkeypatch):
    monkeypatch.setenv("SLOP_PROVIDER", "google-genai")
    monkeypatch.setenv("SLOP_ALLOW_CLOUD", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert config.resolve_provider(_spec("local")) == "google-genai"


def test_unknown_provider_errors(monkeypatch):
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    with pytest.raises(config.SlopError):
        config.resolve_provider(_spec("nonsense"))


def test_imagestage_remote_row_is_gated(monkeypatch):
    """Selecting a remote model row must refuse construction without the cloud opt-in (no spend)."""
    monkeypatch.delenv("SLOP_PROVIDER", raising=False)
    monkeypatch.delenv("SLOP_ALLOW_CLOUD", raising=False)
    from slopmachine.pipeline.image import ImageStage

    with pytest.raises(config.SlopError):
        ImageStage(model_key="hf-flux")


# --- registry remote rows --------------------------------------------------------------

def test_registry_remote_rows_have_provider():
    assert get_model("image", "hf-flux")[1].provider == "hf-inference"
    assert get_model("image", "google")[1].provider == "google-genai"
    assert get_model("image", "sdxl")[1].provider == "local"  # default stays local


# --- device generalization (cuda -> mps -> cpu) ----------------------------------------

def _fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        float16="float16", bfloat16="bfloat16", float32="float32",
    )


def test_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(cuda=True, mps=True))
    assert hardware.device() == "cuda"


def test_device_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(cuda=False, mps=True))
    assert hardware.device() == "mps"


def test_device_cpu_fallback(monkeypatch):
    monkeypatch.setattr(hardware, "_torch", lambda: _fake_torch(cuda=False, mps=False))
    assert hardware.device() == "cpu"


# --- remote backend dispatch (mocked SDK; no network) ----------------------------------

def test_backend_dispatch_hf_inference(monkeypatch):
    import huggingface_hub

    import slopmachine.pipeline.backends as backends

    seen = {}

    class _FakeClient:
        def __init__(self, api_key=None):
            seen["api_key"] = api_key

        def text_to_image(self, prompt, model=None, negative_prompt=None):
            seen.update(prompt=prompt, model=model)
            return "FAKE_IMAGE"

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _FakeClient)
    monkeypatch.setenv("HF_TOKEN", "tok")
    out = backends.generate_image("hf-inference", SimpleNamespace(repo_id="some/model"), "a cat")
    assert out == "FAKE_IMAGE"
    assert seen["model"] == "some/model" and seen["api_key"] == "tok" and seen["prompt"] == "a cat"


def test_backend_unknown_provider_errors():
    import slopmachine.pipeline.backends as backends

    with pytest.raises(config.SlopError):
        backends.generate_image("nope", SimpleNamespace(repo_id="x"), "p")
