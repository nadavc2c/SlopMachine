"""Remote (token-based) image backends for `slop image`.

The LOCAL diffusers backend lives in ``pipeline/image.py``; this module holds only the remote,
paid providers. They are reached ONLY after ``config.resolve_provider()`` has enforced the cloud
opt-in gate (``SLOP_ALLOW_CLOUD``) and confirmed a token — so no call here can spend money
unexpectedly. Each backend returns a ``PIL.Image``. Heavy/optional SDK imports are deferred.
"""

from typing import Any

from .. import config


def generate_image(provider: str, spec, prompt: str, negative_prompt: str = "", **kwargs: Any):
    """Generate one image via a remote provider. Returns a PIL.Image."""
    if provider == "hf-inference":
        return _hf_inference(spec, prompt, negative_prompt)
    if provider == "google-genai":
        return _google_genai(spec, prompt)
    raise config.SlopError(f"No remote image backend for provider '{provider}'.")


def _hf_inference(spec, prompt: str, negative_prompt: str):
    """HuggingFace Inference Providers — official, token-based, unifies many partners."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=config.provider_token("hf-inference"))
    return client.text_to_image(prompt, model=spec.repo_id, negative_prompt=(negative_prompt or None))


def _google_genai(spec, prompt: str):
    """Google Gen AI SDK — current Gemini image model (e.g. gemini-3.x-flash-image / "Nano Banana"),
    NOT the deprecated Imagen path. Model id comes from the registry (``spec.repo_id``).

    Uses the SDK's current image API; the exact surface is verified against the installed
    ``google-genai`` version at build time (this module is only imported when the provider is chosen).
    """
    import base64
    import io

    from google import genai  # from the opt-in `cloud` extra
    from PIL import Image

    client = genai.Client(api_key=config.provider_token("google-genai"))
    result = client.interactions.create(model=spec.repo_id, input=prompt)
    return Image.open(io.BytesIO(base64.b64decode(result.output_image.data)))
