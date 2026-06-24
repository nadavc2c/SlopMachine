---
name: slop
description: >-
  Drive the `slop` CLI — a local, GPU-accelerated generative-content toolkit (the Swiss-army-knife
  of generative media, built to be operated by AI agents). USE THIS SKILL whenever the user wants to
  create, generate, edit, stylize, or animate visual/audio content locally: make/generate an image,
  apply a style (anime / cyberpunk / casino), the "AI dance" use case (animate a person from a photo +
  a driving dance video via pose-skeleton motion transfer — the open analog of Viggle/Kling), or find,
  inspect, and download the current-best open models. Prefer actually running `uv run slop ...` over
  explaining how. Trigger even if the user doesn't say "slop" — e.g. "make me a picture", "animate
  this photo", "make him dance", "best image model".
license: See LICENSE / model cards (per-model; recorded in the registry).
compatibility: >-
  Requires the SlopMachine repo, Astral `uv`, Python 3.14, and an NVIDIA GPU (CUDA 12.8+, ~16GB VRAM
  for video/dance). CPU-only works for some commands but is slow. Run scripts/bootstrap on first use.
allowed-tools: Bash(uv run:*) Bash(uv sync:*) Bash(pwsh:*) Bash(bash:*)
shell: powershell
metadata:
  project: SlopMachine
  cli: slop
  version: "0.1.0"
---

# slop — generative content, by CLI

`slop` is a thin, model-agnostic CLI over HuggingFace `diffusers`. Each capability is a subcommand;
the registry (`src/slopmachine/config/models.yaml`) decides which model backs each one, so commands never change when
models do. **You operate `slop`; you don't reimplement it.** Run commands with `uv run slop ...` so
everything stays inside the project's `uv`-managed `.venv` (never global Python).

## How to invoke (read first)

Run from the **SlopMachine repo root**. Most commands are core:

```
uv run slop <command> [args]
```

The one exception is **`slop dance`**, which needs the optional pose/video deps at runtime — always
run it with `--extra dance` (a bare `uv run` re-syncs to the lean core and would uninstall them):

```
uv run --extra dance slop dance --reference me.jpg --driving clip.mp4 -o outputs/dance.mp4
```

## First, set up (once)

```
pwsh  ${CLAUDE_SKILL_DIR}/scripts/bootstrap.ps1   # Windows
bash  ${CLAUDE_SKILL_DIR}/scripts/bootstrap.sh    # macOS/Linux
```

Bootstrap finds the repo, checks `uv`, runs `uv sync --extra dance`, and does a GPU health check
(`slop info`). Heavy model weights are **not** bundled — they download on first use into the
repo-local cache. Re-run `uv sync --extra dance` if `slop dance` ever reports a missing module.

## Command surface (the map)

| Command | What it does | Detail |
|---|---|---|
| `slop info [--json]` | GPU / VRAM / capability / cache location | — |
| `slop capabilities [--json]` | List every capability + its default model (start here) | — |
| `slop styles [--json]` | List style presets (anime, cyberpunk, casino, …) | [references/image.md](references/image.md) |
| `slop image "<prompt>" [--style S] [--identity me.jpg] -o out.png` | Text → image (+ optional identity) | [references/image.md](references/image.md) |
| `slop dance --reference me.jpg --driving clip.mp4 -o out.mp4` | **AI dance**: animate a person with a driving video (pose-skeleton motion transfer) | [references/dance.md](references/dance.md) |
| `slop models list/show/download [--json]` | Inspect / pre-fetch the registry's models | [references/models.md](references/models.md) |
| `slop assets download <kind>` | Fetch supporting assets (e.g. `pose` ONNX models) | [references/dance.md](references/dance.md) |

**Discoverability:** every command self-documents — run `uv run slop <command> --help` for its exact
flags and examples before composing a call. `--json` (where shown) gives machine-readable output.

## How to operate

1. **Orient:** `uv run slop capabilities` (what can it do? which model is default?).
2. **Pick the command** from the table; read its `references/*.md` for examples + gotchas, or run
   `uv run slop <cmd> --help`.
3. **Ensure weights:** the command auto-downloads on first use, or pre-fetch with
   `uv run slop models download <capability>` (and `uv run slop assets download pose` for dance).
4. **Run it**, always passing `-o <path>` so the output location is explicit, and surface the
   `Saved: <path>` line to the user.

## Operating notes (set expectations, don't treat as a hang)

- **First run of any model downloads large weights** (image ~7GB; the dance model ~77GB) — minutes,
  one time, into the repo cache.
- **`slop dance` is slow on 16GB** (CPU↔GPU layer streaming): expect **minutes per clip**. It is not
  hung. Keep clips short and resolution modest; raise `--steps` only for final quality.
- **On CUDA out-of-memory**, retry with fewer frames (`--fps` lower), a shorter clip, or fewer
  `--steps`. The defaults are tuned to fit 16GB; larger settings may not.
- Errors are single-line `Error: ...` with a non-zero exit code — read them; they're actionable.

## Extending

New capability = a `Stage` in `src/slopmachine/pipeline/` + a row in `src/slopmachine/config/models.yaml` + a
`slop <verb>` command (see CLAUDE.md "Adding a new capability"). To refresh a model to the current
best, use the `slop-models` skill.
