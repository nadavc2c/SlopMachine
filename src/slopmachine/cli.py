"""`slop` CLI — the operator interface for SlopMachine.

Commands are intentionally thin: they resolve a model from the registry, hand off
to a Stage, and print where the output landed. Heavy imports (torch/diffusers) are
deferred into the command bodies so `slop models list` / `--help` stay instant.
"""

import json
import os
import re
from pathlib import Path
from typing import Annotated

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
def info(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False):
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
def styles(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False):
    """List available style presets."""
    names = config.list_styles()
    if as_json:
        typer.echo(json.dumps(names, indent=2))
        return
    for name in names:
        typer.echo(f"  {name}")


@app.command()
def capabilities(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False):
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
    prompt: Annotated[str, typer.Argument(help="Text prompt.")],
    style: Annotated[str | None, typer.Option("--style", "-s", help="Style preset (see `slop styles`).")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Registry model key (default: registry default).")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Backend: local | hf-inference | google-genai. Cloud is OFF by default — needs SLOP_ALLOW_CLOUD=1 + a token; default is local (free).")] = None,
    identity: Annotated[Path | None, typer.Option("--identity", "-i", exists=True, dir_okay=False, help="Reference face image for identity preservation.")] = None,
    identity_scale: Annotated[float, typer.Option(help="Identity strength 0-1. Higher = closer to the reference but weaker prompt/scene; lower it if the scene isn't showing.")] = 0.6,
    steps: Annotated[int | None, typer.Option("--steps", help="Inference steps.")] = None,
    guidance: Annotated[float | None, typer.Option("--guidance", "-g", help="Guidance scale.")] = None,
    width: Annotated[int | None, typer.Option("--width")] = None,
    height: Annotated[int | None, typer.Option("--height")] = None,
    negative: Annotated[str | None, typer.Option("--negative", help="Negative-prompt terms (used by models that support it; guidance-distilled models like FLUX/Qwen ignore it).")] = None,
    best_of: Annotated[int, typer.Option("--best-of", help="Generate N candidates for the agent to view and pick the best (agent-judged).")] = 1,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed for reproducibility.")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output PNG path.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON result (suppresses progress).")] = False,
):
    """Generate an image from a text prompt."""
    from .pipeline.image import ImageStage

    final_prompt, style_defaults = prompt, {}
    neg_parts = []
    if style:
        preset = _safe(config.load_style, style)
        final_prompt, style_neg = preset.apply(prompt)
        if style_neg:
            neg_parts.append(style_neg)
        style_defaults = preset.gen_defaults
    if negative:
        neg_parts.append(negative)
    negative_prompt = ", ".join(neg_parts)

    stage = _safe(ImageStage, model_key=model, provider=provider)

    gen = dict(style_defaults)
    for key, value in {"num_inference_steps": steps, "guidance_scale": guidance, "width": width, "height": height}.items():
        if value is not None:
            gen[key] = value

    if out is None:
        out = config.outputs_dir() / f"{stage.model_key}_{_slug(prompt)}.png"

    if best_of and best_of > 1:
        _best_of(stage, n=best_of, prompt=final_prompt, negative_prompt=negative_prompt,
                 identity=identity, identity_scale=identity_scale, seed=seed, out=out, gen=gen, as_json=as_json)
        return

    if not as_json:
        typer.echo(f"Model:  {stage.model_key} ({stage.spec.repo_id})")
        typer.echo(f"Plan:   {hardware_plan(stage)}")
        typer.echo("Generating (first run downloads weights)...")
    result = stage.run(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
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
            "provider": result.get("provider", "local"),
            "seed": seed,
            **{k: v for k, v in gen.items() if k in ("num_inference_steps", "guidance_scale", "width", "height")},
        }, indent=2))
    else:
        typer.echo(f"Saved:  {result['path']}")


@app.command()
def dance(
    reference: Annotated[Path, typer.Option("--reference", "-r", exists=True, dir_okay=False, help="Reference person image.")],
    driving: Annotated[Path, typer.Option("--driving", "-d", exists=True, dir_okay=False, help="Driving dance video (the motion to transfer).")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Optional scene/style prompt.")] = "",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Animation model key (default: registry default).")] = None,
    steps: Annotated[int | None, typer.Option("--steps", help="Inference steps (fewer = faster).")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed for reproducibility.")] = None,
    fps: Annotated[int | None, typer.Option("--fps", help="Target FPS (default: driving video / preset).")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output MP4 path.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON result (suppresses progress).")] = False,
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


def _best_of(stage, *, n, prompt, negative_prompt, identity, identity_scale, seed, out, gen, as_json):
    """Generate n candidates for the agent to view and pick the cleanest (agent-judged).

    The agent (Claude) is a frontier VLM judge: it reasons about anatomy, composition and appeal in a
    way a fixed reward model can't, and needs no extra model or download.
    """
    if not as_json:
        typer.echo(f"Model:  {stage.model_key} ({stage.spec.repo_id})")
        typer.echo(f"Best-of {n}: generating candidates for the agent to judge...")
    cands = []
    for i in range(n):
        s = (seed + i) if seed is not None else None
        cpath = out.with_name(f"{out.stem}_cand{i + 1}{out.suffix}")
        stage.run(
            prompt=prompt, negative_prompt=negative_prompt,
            identity_image=str(identity) if identity else None,
            identity_scale=identity_scale, seed=s, out=cpath, **gen,
        )
        cands.append({"path": str(cpath), "seed": s})
        if not as_json:
            typer.echo(f"  [{i + 1}/{n}] {cpath}")
    if as_json:
        typer.echo(json.dumps({"candidates": cands, "best_of": n}, indent=2))
    else:
        typer.echo(f"\n{n} candidates ready - review them and copy the cleanest to your output path.")


@models_app.command("list")
def models_list(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False):
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
    capability: Annotated[str, typer.Argument(help="Capability name, e.g. image.")],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
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
    capability: Annotated[str, typer.Argument(help="Capability name, e.g. image.")],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
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


@assets_app.command("list")
def assets_list(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False):
    """List supporting assets: pose models + the curated dance-clip catalog."""
    from .registry import registry

    reg = registry()
    pose_models = list(reg.capabilities["pose"].models) if "pose" in reg.capabilities else []
    dances = config.load_assets().get("dances", {}) or {}
    if as_json:
        typer.echo(json.dumps({"pose": pose_models, "dances": dances}, indent=2))
        return
    typer.echo(f"pose models: {', '.join(pose_models) or '(none)'}")
    if dances:
        for cid, d in dances.items():
            typer.echo(f"dance/{cid}: {d.get('style', '?')} ~{d.get('duration_s', '?')}s [{d.get('license', '?')}]")
    else:
        typer.echo("dance clips: none shipped yet (CC0 first-party catalog is a build step; see config/assets.yaml).")


@assets_app.command("download")
def assets_download(
    kind: Annotated[str, typer.Argument(help="Asset kind: 'pose' or 'dance'.")],
    name: Annotated[str | None, typer.Argument(help="For 'dance': the clip id (see `slop assets list`).")] = None,
):
    """Pre-fetch supporting assets into the local cache."""
    if kind == "pose":
        from .pipeline.pose import ensure_pose_assets
        from .registry import get_model

        key, spec = _safe(get_model, "pose", None)
        typer.echo(f"Downloading pose models ({key}) from {spec.repo_id}/{spec.subfolder} ...")
        path = _safe(ensure_pose_assets, spec)
        typer.echo(f"Cached at: {path}")
    elif kind == "dance":
        dances = config.load_assets().get("dances", {}) or {}
        if not name:
            typer.secho("Usage: slop assets download dance <id>  (see `slop assets list`).", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        entry = dances.get(name)
        if not entry:
            avail = ", ".join(dances) or "(none shipped yet)"
            typer.secho(f"Unknown dance clip '{name}'. Available: {avail}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        from huggingface_hub import hf_hub_download

        config.configure_hf_cache()
        typer.echo(f"Downloading dance clip '{name}' [{entry.get('license', '?')}] ...")
        path = hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"], repo_type="dataset")
        typer.echo(f"Cached at: {path}")
    else:
        typer.secho(f"Unknown asset kind: '{kind}' (known: pose, dance)", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
