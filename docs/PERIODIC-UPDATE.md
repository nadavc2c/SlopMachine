# Periodic-update policy — keeping SlopMachine on the bleeding edge

SlopMachine must always run on the **latest STABLE, official, widely-used** libraries, models, and
agent standards — never 2023-era stale picks, never deprecated patterns, and (just as importantly)
**never beta/rc/preview/nightly**. "Newest + best + official" is the bar; "fragile" is what a stale,
single-author, abandoned thing *is* — not what the current best is.

This is operationalized three ways: (1) a greppable in-code research tag on every pick, (2) this
rationale doc + a running revalidation log, and (3) a monthly refresh routine (the `slop-update` skill).

## Golden rules this enforces

- **Read the official docs FIRST, every time** (PyPI / official docs / HF model cards / official GitHub
  releases). Never assert a version, API, or "best model" from training memory. Record the source URL +
  the date you verified it. (firecrawl only as a fallback when a page won't load.)
- **Latest STABLE only** — no beta/rc/preview/nightly. Stable + official + widely-used.
- **Model-agnostic** — `src/slopmachine/config/models.yaml` is the single source of truth; code never
  hardcodes model ids. Models come and go — keep it fresh.

## Hard target-stack constraints (any pick MUST satisfy these)

- Windows 11 native · Python **3.14** (cp314) · NVIDIA **RTX 5080, Blackwell sm_120**, 16 GB.
- `torch` from the **cu130** index (the current stable CUDA default; cu128 is frozen and no longer
  receives new torch builds — that is exactly how we got stranded on torch 2.11.0).
- diffusers-native (NO ComfyUI / A1111). 16 GB fit via **bf16 + CPU offload + VAE tiling**.
- **No fp8 on sm_120** — layerwise fp8 casting raises `CUBLAS_STATUS_INTERNAL_ERROR` (documented in CLAUDE.md).
- Recency without runnability is worthless: if the newest thing doesn't support this stack, pick the best one that does and say why.

## In-code research tags: `PERIODIC-UPDATE-RESEARCH`

Tag **every** version pin, model id, and stack assumption with a greppable comment carrying the
last-verified date and the official source, so the monthly routine can enumerate and prioritize the stalest:

```
# PERIODIC-UPDATE-RESEARCH: <subject> | last-verified=YYYY-MM-DD | latest-stable=<ver/id>
#   source=<official-url>
#   notes=<why this pick + any stack caveat>
```

Apply to, at minimum: `pyproject.toml` (each ML/tooling pin + the torch cu130 index URL),
`src/slopmachine/config/models.yaml` (every model row), `hardware.py` (the sm_120 / cu130 assumptions),
`pipeline/*.py` (pipeline classes + offload recipes), and the agent-facing files (`SKILL.md`,
`plugin.json`, `marketplace.json`, `llms.txt`). Find them all with: `rg "PERIODIC-UPDATE-RESEARCH"`.

## Monthly refresh routine

Run the **`slop-update`** skill (or do it by hand following its steps) once a month. In short: enumerate
the tags oldest-first; re-check every library against live PyPI/release notes (special attention to the
torch CUDA index, which PyTorch rotates); re-verify every model id via `/slop-models` (exists, ungated,
diffusers-native on the installed diffusers, stable, fits 16 GB) and surface any newer current-best;
re-check template/mascot media; re-check the **Agent-Skill** standard (agentskills.io) and the
**Claude-Code-plugin** standard (code.claude.com) against their live docs; bump the tag dates + lockfile;
verify on-device (`uv run slop info`, a smoke `slop image`); and append an entry to the log below.

Explicitly **ignore**: blog "open weights" claims with no HF repo, and paper-only models with no
diffusers-loadable weights (adopt them only once real, ungated, diffusers-native weights ship).

## Revalidation log

### 2026-06-24 — full-codebase revalidation (8-agent audit vs live official docs)

- **Infra/tooling/Python: current.** diffusers 0.38.0, transformers 5.12.1, accelerate 1.14.0,
  huggingface-hub 1.20.1, safetensors 0.8.0, pydantic 2.13.4, pytest 9.1.1, uv/uv_build, Python 3.14 —
  all at latest stable.
- **torch: STRANDED on 2.11.0** via the frozen cu128 index → switch to **cu130**, `torch>=2.12.1`.
- **Image models: swapped off 2023/2024.** New default **FLUX.2-klein-4B** (Apache, ungated, ~24 GB,
  4-step); max-quality tier **Qwen-Image-2512** (Apache, ungated, ~58 GB); FLUX.1-dev → prior-gen; SDXL →
  labeled legacy/low-VRAM fallback. (Animation Wan2.2-Animate-14B + pose ViTPose verified still current.)
- **Dropped 2023-era anti-"slop" band-aids:** PAG (no FLUX/Qwen pipeline), the always-on anti-anatomy
  negative baseline (ignored by guidance-distilled models), and the PickScore-2023 reward judge. Kept
  `--best-of N` as agent-judged. Removed the broken fp8 path on sm_120.
- **Also:** `typer>=0.26.7`; manifest fixes (marketplace `$schema` + top-level description); dead
  `runner.py`; minor 3.14 idioms.
