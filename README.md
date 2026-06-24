# SlopMachine

**`slop` — the Swiss-army-knife of generative content, built from the ground up to be driven by AI agents.**

A modular, CLI-driven toolkit for running the best open generative-AI models **locally** on an NVIDIA
Blackwell GPU (built/tested on an RTX 5080, 16 GB). One `slop` CLI, many capabilities — text→image
(with styles + identity) and the flagship **AI dance** (animate a person from a photo + a driving
video). Backed by HuggingFace `diffusers` — **no ComfyUI**.

Design goals: a CLI an **agent** can drive (discoverable `--help`, `--json` output, clean errors,
no prompts), always-current models (a registry you refresh, not ids hardcoded in code), and modular
stages you can extend without touching the rest.

## Requirements

- Windows (native), NVIDIA Blackwell GPU (`sm_120`) with a recent driver (CUDA 12.8+); ~16 GB VRAM.
- [`uv`](https://docs.astral.sh/uv/) — manages Python (3.14), the venv, and all dependencies.

## Quickstart

```bash
uv sync                                   # core deps; creates .venv (Python 3.14)
uv run slop info                          # confirm GPU / VRAM / sm_120
uv run slop capabilities                  # what can it do? (default model per capability)

# image
uv run slop image "a neon cat on a rooftop" --style cyberpunk -o outputs/cat.png

# AI dance (needs the opt-in 'dance' extra at runtime)
uv sync --extra dance
uv run --extra dance slop dance --reference me.jpg --driving dance.mp4 -o outputs/dance.mp4
```

First run of any model downloads its weights into the repo-local `models/` cache (kept out of git).

## Commands

| Command | What it does |
| --- | --- |
| `uv run slop info [--json]` | GPU, VRAM, capability, cache location |
| `uv run slop capabilities [--json]` | List every capability + its default model (start here) |
| `uv run slop styles [--json]` | List style presets (anime, cyberpunk, casino) |
| `uv run slop image "PROMPT" [opts]` | Text → image (`--style`, `--identity face.jpg`, `--seed`, …) |
| `uv run --extra dance slop dance -r me.jpg -d clip.mp4 -o out.mp4` | **AI dance**: animate a person with a driving video |
| `uv run slop models list/show/download [--json]` | Inspect / pre-fetch registry models |
| `uv run slop assets download pose` | Fetch the ONNX pose models used by `slop dance` |

Every command self-documents — `uv run slop <command> --help`.

## AI dance (the flagship)

Reference photo of a person **+** a driving dance video **→** that person performing the dance —
the open analog of Viggle / Kling / ElevenLabs AI-dance. Two stages: pose-skeleton + face extraction
(ONNX YOLO + ViTPose, CPU) → pose-driven motion transfer (diffusers-native **Wan2.2-Animate**). Fits
16 GB via the official group-offloading recipe (verified ~12.5 GB peak); it streams the model from
CPU, so expect **minutes per clip**. On out-of-memory, lower `--fps`/`--steps` or shorten the clip.

## Use it as an Agent Skill / Claude plugin

`.claude/skills/slop/` is a thin Agent Skill — the agent-facing manual (command map + examples +
setup) over the `slop` CLI, conforming to the [agentskills.io](https://agentskills.io) standard. It
*also* ships as a **Claude Code plugin** (`.claude/skills/slop/.claude-plugin/plugin.json`) listed in
a repo-root marketplace (`.claude-plugin/marketplace.json`).

**The CLI is the engine; the skill is its manual.** Because `slop` needs the Python project + a GPU +
on-demand weights, the primary path is to **clone the repo** — the skill then auto-loads and
`scripts/bootstrap.{ps1,sh}` sets everything up:

```bash
git clone https://github.com/nadavc2c/SlopMachine && cd SlopMachine
uv sync                                   # add --extra dance for video/animation
# In Claude Code the `slop` skill auto-loads from .claude/skills/slop/
```

**Install just the skill/plugin (for discovery) — all from GitHub, no npm publish needed:**

```bash
# Claude Code plugin marketplace:
/plugin marketplace add nadavc2c/SlopMachine
/plugin install slop@slopmachine

# Cross-agent (Cursor, Codex, Copilot, … — agentskills.io standard):
npx skills add nadavc2c/SlopMachine        # or: gh skill install (GitHub CLI's native command)
```

> The skill drives `uv run slop ...`, so it needs the repo present (its bootstrap locates it). To make
> `slop` runnable *anywhere* without the repo, publish the CLI to PyPI (`uv tool install slopmachine` /
> `uvx slop`) — a planned follow-up. (There is no "uv install skill"; uv installs the CLI, not skills.)

The skill ships **no weights** — they download on demand. (Refresh models with the **`slop-models`** skill.)

## Models are not hardcoded

`config/models.yaml` is the single source of truth for which model backs each capability. Swap a
model by editing one row — no code changes. Run the **`slop-models`** skill to research and refresh
the registry to the current-best models. Seeded entries are a runnable starting point; models come and go.

## Styles

Prompt presets live in `config/styles/*.yaml` (anime, cyberpunk, casino "3D glossy slot-art").
Add your own by dropping in a new YAML file.

## Containment

Everything stays in this folder: the `.venv` (uv), generated outputs (`outputs/`), and model weights
(`models/`, via `HF_HOME`). Nothing is installed system-wide. Heavy deps for video/animation are an
opt-in extra (`uv sync --extra dance`), keeping the core lean.

## Roadmap

- **Foundation + image** (done): `slop image` with styles + identity.
- **AI dance** (done): `slop dance` — pose-skeleton extraction → Wan2.2-Animate motion transfer, on 16 GB.
- **Next:** text/image → video; faster dance (distilled few-step); a curated CC0 driving-clip catalog;
  then CLI-fied versions of the most common ComfyUI workflows (upscale, inpaint, controlnet, …); music.

See `CLAUDE.md` for architecture and contributor conventions.
