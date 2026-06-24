# slop dance — AI dance (pose-skeleton motion transfer)

Animate a person from a single reference photo using the motion of a driving video — the open analog
of Viggle / Kling / ElevenLabs AI-dance. Engine: diffusers-native **Wan2.2-Animate** (`animation`
capability); driving signal from **PoseStage** (ONNX YOLO + ViTPose, CPU).

## Command

```
uv run --extra dance slop dance --reference <person.jpg> --driving <dance.mp4> -o outputs/dance.mp4
```

Run `uv run --extra dance slop dance --help` for the authoritative flag list. Key flags:
- `--reference, -r`  reference person image (a clear, full-body photo works best for dance).
- `--driving, -d`    driving video — the motion to copy (one person, full-body, clear framing).
- `--prompt, -p`     optional scene/style text (CFG is off by default, so it's a light touch).
- `--steps`          inference steps; fewer = faster + lower quality (default tuned for 16GB).
- `--fps`            target fps; lower = fewer frames = less VRAM + faster.
- `--seed`           reproducibility. `-o` output mp4 path.

## How it works (two stages, auto-chained)

1. **PoseStage** turns `--driving` into a pose-skeleton video + a face video (the driving signal).
2. **AnimateStage** (Wan2.2-Animate) renders the reference person following that motion.

## One-time assets

Both auto-download on first run, or pre-fetch:
```
uv run slop assets download pose         # small ONNX detector + pose models
uv run slop models download animation    # ~77GB Wan2.2-Animate (first run only)
```

## 16 GB GPU reality — set expectations, don't treat as a hang

- Verified to run at **~12.5 GB** via the official group-offload recipe, but it **streams** the model
  from CPU, so it's **minutes per clip**. This is expected, not a hang.
- On **CUDA out-of-memory**: lower `--fps` (fewer frames), shorten the clip, or lower `--steps`.
- Defaults fit 16 GB; large resolution / long clips may OOM.

## Chain: "generate me, then make me dance"

```
uv run slop image "full-body studio photo of a person, plain background, standing" -o outputs/me.png
uv run --extra dance slop dance --reference outputs/me.png --driving dance.mp4 -o outputs/me_dancing.mp4
```
