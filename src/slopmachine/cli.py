"""`slop` CLI — the operator interface for SlopMachine.

Commands are intentionally thin: they resolve a model from the registry, hand off
to a Stage, and print where the output landed. Heavy imports (torch/diffusers) are
deferred into the command bodies so `slop models list` / `--help` stay instant.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import typer

from . import config

# Keep downloaded weights inside the repo before anything touches HF/diffusers.
config.configure_hf_cache()

app = typer.Typer(
    help="slop — the Swiss-army-knife of generative content, built to be driven by AI agents.",
    no_args_is_help=True,
    add_completion=False,
    epilog=(
        "Examples:\n\n"
        '  slop capabilities --json\n\n'
        '  slop image "a neon cat" --style cyberpunk -o outputs/cat.png\n\n'
        "  slop dance --reference me.jpg --driving dance.mp4 -o outputs/dance.mp4\n\n"
        "  slop models show animation\n\n"
        "Every command self-documents: `slop <command> --help`."
    ),
)

# capability (registry) -> the slop command that uses it (shown by `slop capabilities`).
_CAP_COMMAND = {
    "image": "image",
    "identity": "image --identity",
    "animation": "dance",
    "pose": "dance (internal)",
}
models_app = typer.Typer(help="Inspect and fetch models from the registry.", no_args_is_help=True)
app.add_typer(models_app, name="models")
assets_app = typer.Typer(help="Download supporting assets (pose models, dance clips).", no_args_is_help=True)
app.add_typer(assets_app, name="assets")


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "image"


def _safe(fn, *args, **kwargs):
    """Run a resolver; turn SlopError into a clean CLI message instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except config.SlopError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def info(as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output.")):
    """Show device / VRAM / cache location + which backends are available."""
    from . import hardware

    dev = hardware.device()
    cap = hardware.capability() if dev == "cuda" else None
    tokens = {
        "local": True,
        "hf-inference": bool(config.provider_token("hf-inference")),
        "google-genai": bool(config.provider_token("google-genai")),
    }
    data = {
        "device": dev,
        "gpu": hardware.gpu_name(),
        "vram_gb": round(hardware.total_vram_gb(), 1) if dev == "cuda" else None,
        "capability": f"sm_{cap[0]}{cap[1]}" if cap else None,
        "hf_cache": str(config.configure_hf_cache()),
        "cloud_allowed": config.cloud_allowed(),
        "default_provider": os.environ.get("SLOP_PROVIDER", "local"),
        "tokens_present": tokens,
    }
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    typer.echo(f"Device:     {dev}" + ("" if dev == "cpu" else f"  ({data['gpu']})"))
    if data["vram_gb"] is not None:
        typer.echo(f"VRAM:       {data['vram_gb']:.1f} GB")
    if data["capability"]:
        typer.echo(f"Capability: {data['capability']}")
    typer.echo(f"HF cache:   {data['hf_cache']}")
    typer.echo(
        "Cloud:      "
        + ("ALLOWED (SLOP_ALLOW_CLOUD set)" if data["cloud_allowed"]
           else "off (default; set SLOP_ALLOW_CLOUD=1 to enable paid remote providers)")
    )
    typer.echo(f"Provider:   {data['default_provider']} (default)")
    have = [b for b in ("hf-inference", "google-genai") if tokens[b]]
    typer.echo("Tokens:     " + (", ".join(have) if have else "none (local only)"))


@app.command()
def styles(as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output.")):
    """List available style presets."""
    names = config.list_styles()
    if as_json:
        typer.echo(json.dumps(names, indent=2))
        return
    for name in names:
        typer.echo(f"  {name}")


@app.command()
def capabilities(as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output.")):
    """List every capability and the command that uses it — start here to see what slop can do."""
    from .registry import registry

    reg = registry()
    data = {
        cap: {"default": c.default, "models": list(c.models), "command": _CAP_COMMAND.get(cap, cap)}
        for cap, c in reg.capabilities.items()
    }
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    for cap, d in data.items():
        typer.echo(f"{cap:<11} -> slop {d['command']:<18} (default: {d['default']}; {len(d['models'])} model(s))")


@app.command()
def image(
    prompt: str = typer.Argument(..., help="Text prompt."),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Style preset (see `slop styles`)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Registry model key (default: registry default)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Backend: local | hf-inference | google-genai. Cloud is OFF by default — needs SLOP_ALLOW_CLOUD=1 + a token; default is local (free)."),
    identity: Optional[Path] = typer.Option(None, "--identity", "-i", exists=True, dir_okay=False, help="Reference face image for identity preservation."),
    identity_scale: float = typer.Option(0.6, help="Identity strength 0-1. Higher = closer to the reference but weaker prompt/scene; lower it if the scene isn't showing."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Inference steps."),
    guidance: Optional[float] = typer.Option(None, "--guidance", "-g", help="Guidance scale."),
    width: Optional[int] = typer.Option(None, "--width"),
    height: Optional[int] = typer.Option(None, "--height"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output PNG path."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON result (suppresses progress)."),
):
    """Generate an image from a text prompt."""
    from .pipeline.image import ImageStage

    final_prompt, negative, style_defaults = prompt, "", {}
    if style:
        preset = _safe(config.load_style, style)
        final_prompt, negative = preset.apply(prompt)
        style_defaults = preset.gen_defaults

    stage = _safe(ImageStage, model_key=model, provider=provider)

    gen = dict(style_defaults)
    for key, value in {"num_inference_steps": steps, "guidance_scale": guidance, "width": width, "height": height}.items():
        if value is not None:
            gen[key] = value

    if out is None:
        out = config.outputs_dir() / f"{stage.model_key}_{_slug(prompt)}.png"

    if not as_json:
        typer.echo(f"Model:  {stage.model_key} ({stage.spec.repo_id})")
        typer.echo(f"Plan:   {hardware_plan(stage)}")
        typer.echo("Generating (first run downloads weights)...")
    result = stage.run(
        prompt=final_prompt,
        negative_prompt=negative,
        identity_image=str(identity) if identity else None,
        identity_scale=identity_scale,
        seed=seed,
        out=out,
        **gen,
    )
    if as_json:
        typer.echo(json.dumps({
            "path": str(result.get("path")) if result.get("path") else None,
            "model": result.get("model"),
            "provider": getattr(stage.spec, "provider", "local"),
            "seed": seed,
            **{k: v for k, v in gen.items() if k in ("num_inference_steps", "guidance_scale", "width", "height")},
        }, indent=2))
    else:
        typer.echo(f"Saved:  {result['path']}")


@app.command()
def dance(
    reference: Path = typer.Option(..., "--reference", "-r", exists=True, dir_okay=False, help="Reference person image."),
    driving: Path = typer.Option(..., "--driving", "-d", exists=True, dir_okay=False, help="Driving dance video (the motion to transfer)."),
    prompt: str = typer.Option("", "--prompt", "-p", help="Optional scene/style prompt."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Animation model key (default: registry default)."),
    steps: Optional[int] = typer.Option(None, "--steps", help="Inference steps (fewer = faster)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility."),
    fps: Optional[int] = typer.Option(None, "--fps", help="Target FPS (default: driving video / preset)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output MP4 path."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON result (suppresses progress)."),
):
    """AI dance: animate a person from a reference photo with a driving dance video.

    Two stages: pose-skeleton + face extraction from the driving video (PoseStage),
    then pose-driven motion transfer (AnimateStage / Wan2.2-Animate).
    """
    from . import hardware
    from .pipeline.animation import AnimateStage
    from .pipeline.pose import PoseStage

    pose = _safe(PoseStage)
    anim = _safe(AnimateStage, model_key=model)

    if out is None:
        out = config.outputs_dir() / f"dance_{_slug(reference.stem)}.mp4"

    if not as_json:
        _offloaded = _safe(lambda s: hardware.plan_for(s).offload != "none", anim.spec)
        _strategy = (
            "bf16, group offload (transformer leaf-level + stream), fp32 VAE  [official 16GB Wan recipe]"
            if _offloaded
            else "bf16, full GPU, fp32 VAE"
        )
        typer.echo(f"Model:  {anim.model_key} ({anim.spec.repo_id})")
        typer.echo(f"Plan:   {_strategy}")
        typer.echo("Step 1/2: extracting pose skeleton + face from the driving video...")
    pose_out = pose.run(driving_video=str(driving), reference_image=str(reference), fps=fps)
    if not as_json:
        typer.echo(f"          {pose_out['num_frames']} frames @ {pose_out['fps']} fps")
        typer.echo("Step 2/2: animating (first run loads ~77GB weights with offload; minutes per clip)...")

    gen = {"num_inference_steps": steps} if steps is not None else {}
    result = anim.run(
        reference_image=pose_out["reference_image"],
        pose_video=pose_out["pose_video"],
        face_video=pose_out["face_video"],
        prompt=prompt,
        fps=pose_out["fps"],
        seed=seed,
        out=out,
        **gen,
    )
    if as_json:
        typer.echo(json.dumps({
            "path": str(result.get("path")) if result.get("path") else None,
            "model": result.get("model"),
            "provider": getattr(anim.spec, "provider", "local"),
            "fps": result.get("fps"),
            "num_frames": pose_out.get("num_frames"),
            "seed": seed,
        }, indent=2))
    else:
        typer.echo(f"Saved:  {result['path']}")


def hardware_plan(stage) -> str:
    from . import hardware

    try:
        return hardware.plan_for(stage.spec).describe()
    except Exception:
        return "n/a"


@models_app.command("list")
def models_list(as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output.")):
    """List capabilities and their candidate models (* = default)."""
    from .registry import registry

    reg = registry()
    if as_json:
        data = {
            cap: {
                "default": c.default,
                "models": {
                    k: {"repo_id": s.repo_id, "license": s.license, "min_vram_gb": s.min_vram_gb, "tier": s.tier}
                    for k, s in c.models.items()
                },
            }
            for cap, c in reg.capabilities.items()
        }
        typer.echo(json.dumps(data, indent=2))
        return
    for cap, c in reg.capabilities.items():
        typer.echo(f"\n{cap}  (default: {c.default})")
        for key, spec in c.models.items():
            star = "*" if key == c.default else " "
            typer.echo(f"  {star} {key:<18} {spec.repo_id}  [{spec.license}, ~{spec.min_vram_gb:.0f}GB]")


@models_app.command("show")
def models_show(
    capability: str = typer.Argument(..., help="Capability name, e.g. image."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Show full spec for a model."""
    from .registry import get_model

    key, spec = _safe(get_model, capability, model)
    typer.echo(f"{capability} / {key}")
    for field, value in spec.model_dump().items():
        if value not in (None, "", {}):
            typer.echo(f"  {field}: {value}")


@models_app.command("download")
def models_download(
    capability: str = typer.Argument(..., help="Capability name, e.g. image."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Pre-fetch a model's weights into the repo-local cache."""
    from huggingface_hub import snapshot_download

    from .registry import get_model

    key, spec = _safe(get_model, capability, model)
    typer.echo(f"Downloading {spec.repo_id} ({key}) into {config.configure_hf_cache()} ...")
    # Entries that target a subfolder (adapters / asset bundles) fetch only that — never the whole repo.
    if spec.subfolder:
        path = snapshot_download(spec.repo_id, allow_patterns=[f"{spec.subfolder}/*"])
    else:
        path = snapshot_download(spec.repo_id)
    typer.echo(f"Cached at: {path}")


@assets_app.command("download")
def assets_download(
    kind: str = typer.Argument(..., help="Asset kind to fetch. Known: 'pose'."),
):
    """Pre-fetch supporting assets into the repo-local cache."""
    if kind == "pose":
        from .pipeline.pose import ensure_pose_assets
        from .registry import get_model

        key, spec = _safe(get_model, "pose", None)
        typer.echo(f"Downloading pose models ({key}) from {spec.repo_id}/{spec.subfolder} ...")
        path = _safe(ensure_pose_assets, spec)
        typer.echo(f"Cached at: {path}")
    else:
        typer.secho(f"Unknown asset kind: '{kind}' (known: pose)", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
