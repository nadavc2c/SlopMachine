# slop image — text → image (+ styles, + identity)

```
uv run slop image "<prompt>" [--style anime|cyberpunk|casino] [--identity face.jpg] -o outputs/x.png
```

Run `uv run slop image --help` for the full flag list. Key flags:
- `--style, -s`        prompt preset (`uv run slop styles` to list: anime, cyberpunk, casino).
- `--identity, -i`     reference face for identity preservation (IP-Adapter).
- `--identity-scale`   0–1; higher = closer to the face but weaker scene control (default 0.6).
- `--model, -m`        registry model key (default = registry default; see `slop models list`).
- `--pag / --no-pag`   Perturbed-Attention Guidance: structural anti-"slop", ON by default on the SDXL
                       tier (no-op on FLUX/Qwen). `--no-pag` = fastest; `--pag-scale` to tune (~3).
- `--best-of N`        generate N candidates, then judge. `--judge agent` (default) writes the N
                       candidates for YOU (the agent) to view and keep the cleanest — the agent-as-judge loop.
- `--negative`         extra negative-prompt terms (an anti-anatomy baseline is always applied, SDXL tier).
- `--steps`, `--guidance, -g`, `--width`, `--height`, `--seed`, `-o`.

## Examples

```
uv run slop image "a neon cat on a rooftop" --style cyberpunk -o outputs/cat.png
uv run slop image "portrait, studio light" --identity me.jpg --seed 42 -o outputs/me.png
uv run slop image "a fox-girl mascot, slot machine" --style casino --best-of 3 --seed 10 -o outputs/m.png
# best-of writes outputs/m_cand1..3.png; you (the agent) view them and copy the cleanest to outputs/m.png
```

## Notes
- First run downloads the image model (~7 GB) into the repo-local cache. Comfortable on 16 GB.
- `--identity` uses a global IP-Adapter: strong identity, weaker scene control (a face-only adapter
  is a planned upgrade). Lower `--identity-scale` if the scene/prompt isn't showing.
- **Anti-"slop":** PAG + an anti-anatomy negative baseline are ON by default (SDXL tier) for cleaner
  structure. For stubborn artifacts: raise `--best-of` and pick the cleanest, escalate to a stronger
  `--model` (e.g. flux-dev), or (planned) `slop fix` for targeted hand/face repair.
