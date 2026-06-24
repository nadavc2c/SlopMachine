# CLAUDE.md — SlopMachine

`slop` — the Swiss-army-knife of generative content, built from the ground up to be driven by AI
agents. A modular, CLI-driven local generative-AI toolkit (image · AI-dance/animation · → video ·
music) for an NVIDIA RTX 5080 (16 GB, Blackwell `sm_120`). A thin orchestration layer over HuggingFace
`diffusers`. The companion Agent Skill is `.claude/skills/slop/` (the agent-facing manual + setup over
the CLI). Agent-first CLI: discoverable `--help`, `--json` output, clean errors, no interactive prompts.

## Core values

**simple · portable · safe · secure · stable · well-maintained · best-practice per OFFICIAL docs only ·
NO "hacks" that abuse the tech.** Portable = NVIDIA is first-class but `slop` runs on CUDA/MPS/CPU and
on token-based remote providers. Safe/secure = cloud never spends without an explicit human opt-in
(`SLOP_ALLOW_CLOUD=1`); tokens come from env, never stored/committed. The Golden Rules below
operationalize these.

## Golden rules (non-negotiable)

1. **Read the official docs FIRST — every step.** Web-search the latest official source before
   asserting versions, APIs, or support matrices. Never reason from training memory. Never use
   deprecated patterns. (firecrawl only when a page won't load.)
2. **uv only.** `uv add` / `uv run`. Never `pip`/`conda`, never install into global/system Python.
3. **Stay contained.** Everything lives in this repo + its `.venv`. Model weights go to the
   repo-local `models/` cache (via `HF_HOME`). **Never install system-wide packages or drivers
   without asking first.**
4. **No ComfyUI.** `diffusers` is the backbone.
5. **Model-agnostic.** Never hardcode model ids in code. `src/slopmachine/config/models.yaml` is the single source
   of truth; refresh to current-best via the `/slop-models` skill. Models come and go — keep it fresh.
6. **Personal/experimental use** → choose the highest-quality model regardless of license (note
   restrictions in the registry; don't avoid models over them).
7. **Security & dependency trust.** Prefer **only official, well-maintained, widely-trusted
   libraries** (stable + safe) while still using modern best-in-class tooling (Astral `uv`).
   Supply-chain hygiene: pin/lock via `uv.lock`; keep the core lean and add extras opt-in
   (`[project.optional-dependencies]`); avoid abandoned/research-only one-offs; favour
   diffusers-native / big-corp-backed paths over fragile forks. **No sketchy workarounds** — if the
   clean path is blocked, stop and flag it rather than hacking around it. When upstream code is
   genuinely needed but unpackaged, vendor the official source under `vendor/` with a NOTICE
   (license + source + exact modifications), kept as close to verbatim as possible.

## Environment

- Windows 11 native · RTX 5080 16 GB (`sm_120`) · driver CUDA 13.3.
- Python **3.14** (uv-managed, see `.python-version`) · `torch` from the **cu128** index.
- Verify GPU: `uv run slop info` (expects `sm_120`, ~16 GB).

## Commands

```
uv run slop info [--json]              # GPU / VRAM / sm_ / cache location
uv run slop capabilities [--json]      # every capability + its default model (start here)
uv run slop styles [--json]            # list style presets
uv run slop models list [--json]       # capabilities + candidate models (* = default)
uv run slop models show image          # full spec for a model
uv run slop models download image      # pre-fetch weights into repo-local models/
uv run slop image "a neon cat" --style cyberpunk -o outputs/cat.png
uv run slop image "portrait" --identity me.jpg --seed 42   # "image of me"
# AI dance — needs the opt-in 'dance' extra at RUNTIME (a bare `uv run` would uninstall it):
uv run slop assets download pose       # ONNX pose models for dance
uv run --extra dance slop dance --reference me.jpg --driving dance.mp4 -o outputs/dance.mp4
```

## Architecture (`src/slopmachine/`)

- `cli.py` — Typer CLI (thin; heavy imports deferred into commands).
- `config.py` — pydantic schemas + loaders; repo-local `outputs/`, `models/`, HF-cache containment.
- `registry.py` — capability → `ModelSpec` (reads `src/slopmachine/config/models.yaml`). The only model-lookup path.
- `hardware.py` — VRAM detection + `LoadPlan` (dtype / offload / VAE-tiling; NO fp8 on sm_120). **All
  VRAM-fitting logic lives here**; stages ask it for a plan.
- `pipeline/base.py` — `Stage` interface (`run(**inputs) -> dict`). The modular unit.
- `pipeline/image.py` — image stage: `AutoPipelineForText2Image` (or an explicit registry `pipeline`
  class, e.g. `Flux2KleinPipeline` / `QwenImagePipeline`), model-agnostic kwarg filtering, optional
  identity adapter (IP-Adapter).
- `pipeline/pose.py` — `PoseStage`: driving video → pose-skeleton + face videos (ONNX YOLO+ViTPose on
  CPU), reusing the vendored official Wan preprocessing.
- `pipeline/animation.py` — `AnimateStage`: `WanAnimatePipeline` (AI dance). 16GB fit via the OFFICIAL
  diffusers group-offloading recipe (bf16, NO fp8, NO `pipe.to(cuda)`).
- `vendor/wan_animate_preprocess/` — vendored official Wan-Animate preprocessing (Apache-2.0; see NOTICE).
- `src/slopmachine/config/models.yaml` — the registry. `src/slopmachine/config/styles/*.yaml` — prompt presets (anime/cyberpunk/casino).

## Dependency policy — lean core, opt-in extras

- **Core only:** torch · diffusers · transformers · accelerate · huggingface_hub · safetensors + CLI deps.
- **Fitting big models into 16 GB uses STOCK torch/diffusers** (CPU/group offload, VAE tiling) — zero
  extra deps. For **video/animation** the official path is diffusers **group offloading** (transformer
  leaf-level + `use_stream` + `low_cpu_mem_usage`, text-encoder block-level), in `pipeline/animation.py`.
- **fp8 layerwise casting is BROKEN on this stack** (Blackwell `sm_120`): fp8 matmul raises
  `CUBLAS_STATUS_INTERNAL_ERROR` and falls back to a slow unfused path. Do NOT use fp8 here — use bf16 +
  offload + VAE tiling. (fp8 has been removed from `hardware.py`'s plan entirely.)
- **The `dance` extra** (`uv sync --extra dance`: onnxruntime · opencv-python-headless · imageio[-ffmpeg]
  · matplotlib · ftfy) is opt-in and needed at RUNTIME by `slop dance` — run it as
  `uv run --extra dance slop dance ...` (a bare `uv run` re-syncs to core and uninstalls it). CPU
  onnxruntime on purpose: the GPU wheel lacks Blackwell `sm_120` kernels; pose extraction is light.
- **Quantization libs** (bitsandbytes/gguf/torchao — note GGUF loading is currently broken in diffusers
  for Wan, issue #12009), **attention backends** (flash-attn/xformers), **Triton** — opt-in *per-model,
  only when measurably needed*. Never add upfront.

## Adding a new capability (the modular path)

1. Add a `Stage` subclass in `pipeline/` implementing `run()`.
2. Add the capability + model row(s) to `src/slopmachine/config/models.yaml`.
3. Add a `slop <cap>` command in `cli.py` that resolves the model and calls the stage.
   (e.g. future SVG: `TraceStage` + `pipeline/trace.py` + `slop trace`.)

## Roadmap

- **Foundation + image (done):** `slop image` (+ identity / "image of me").
- **AI dance (done):** `slop dance` — pose-skeleton extraction → Wan2.2-Animate motion transfer,
  verified ~12.5 GB on the RTX 5080. Companion Agent Skill: `.claude/skills/slop/`.
- **Portable backends + install-anywhere (done):** local CUDA/MPS/CPU + opt-in token-gated remote
  providers (cloud OFF by default); `config/` shipped as package data via `importlib.resources`.
- **Bleeding-edge models + agent-judged quality (in progress):** keep the registry on the current best
  open models (which solve anatomy/hands natively) + `--best-of N` for the agent to pick the cleanest.
  A periodic-update discipline keeps every model/dep/standard fresh (see `docs/PERIODIC-UPDATE.md`).
- **Character consistency + recipe workflows (sticker packs):** a saved **character** (reference image +
  locked style/seed via IP-Adapter → optional per-character LoRA) run through a library of **recipes**
  (poses / costumes / actions) to emit uniform sticker-style packs — workflows as human-readable data
  (`slop sticker-pack --character X`), curated/extended by Claude, not a ComfyUI node graph.
- **★ North star:** a fully autonomous, Claude-driven **TikTok slop machine** — brief → visuals → music
  → captions → ffmpeg assembly (9:16, burned-in captions) → post-ready clip.
- **Next:** image/text → video; music (ACE-Step / Stable Audio); faster dance (distilled few-step);
  curated CC0 driving-clip catalog; CLI-fied common ComfyUI workflows (upscale / inpaint / …).
