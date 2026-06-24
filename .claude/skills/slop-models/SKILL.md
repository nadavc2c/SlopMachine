---
description: Pick or refresh the current best open model for a SlopMachine capability (image, video, animation, music) and update config/models.yaml. Use when the user wants to update models, find the latest/strongest model for a task, or refresh the registry.
when_to_use: Trigger on "update the models", "latest/best model for <X>", "refresh the registry", "what should we use for <capability>", or right after adding a new capability.
argument-hint: "[capability]"
---

# /slop-models — choose & refresh the best model per capability

Keep `config/models.yaml` pointed at the current strongest **and runnable** model for each
capability on THIS box: RTX 5080, 16 GB, `sm_120`, Windows native, Python 3.14, `diffusers`
backbone (no ComfyUI). Models churn monthly — this is how we stay current without hardcoding
model ids in code.

## Rules
- **Read the official model card / docs FIRST (web search). Never pick from memory.** Confirm the
  repo id, pipeline class, license, and VRAM footprint from the actual HuggingFace card / official
  repo as of today.
- Prefer official, well-maintained, big-corp-backed repos. Avoid ComfyUI-only models unless they
  also load via `diffusers` or an official Python repo.
- Use is personal/experimental → **license is not a blocker**; record it in the registry but choose
  on quality + runnability.

## Procedure (for a capability, e.g. `image`)
1. **Research** current best: HF Hub (trending / most-liked for the task), official vendor blogs and
   repos, recent head-to-head comparisons. Read the model card.
2. **Check it runs here:**
   - Loadable via `diffusers` (AutoPipeline or a documented pipeline class) or an official Python repo?
   - Fits 16 GB? If it exceeds ~14 GB at bf16, lean on `hardware.py` (model CPU offload, fp8 layerwise
     casting, VAE tiling) or note a quant option. **Flag honestly if it can't realistically run.**
   - Blackwell/torch-compatible? No Linux-only custom CUDA kernels (or note the workaround).
3. **Update `config/models.yaml`:** add/edit a row under the capability's `models:`; set `default:` if
   it becomes the new default. Fields (see `ModelSpec` in `src/slopmachine/config.py`): `repo_id`,
   `precision`, `variant`, `min_vram_gb`, `license`, `tier`, `pipeline` (only if not AutoPipeline),
   `subfolder`/`weight_name` (adapters), `gen_defaults`, `notes`.
4. **Verify:** `uv run slop models show <capability>`, then a smoke generation
   (`uv run slop image "..."`) watching VRAM in `nvidia-smi`. Don't call it done if it OOMs or errors.
5. **Record WHY** in `notes` (date, source, what it beats) so the next refresh has context.

## Capabilities & starting candidates (VERIFY before use — likely stale)
- **image:** SDXL (fast tier), FLUX-class (quality). **identity:** IP-Adapter / InstantID / PuLID.
- **video** (M2): Wan-class i2v/t2v.
- **animation** (M3): Wan-Animate / UniAnimate-class + pose extraction (rtmlib / YOLO-pose).
- **music** (M4): ACE-Step / Stable Audio class.

This list is a research starting point, **not** the answer. Confirm everything against current docs.
