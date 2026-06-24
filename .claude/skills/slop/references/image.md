# slop image — text → image (+ styles, + identity)

```
uv run slop image "<prompt>" [--style anime|cyberpunk|casino] [--identity face.jpg] -o outputs/x.png
```

Run `uv run slop image --help` for the full flag list. Key flags:
- `--style, -s`        prompt preset (`uv run slop styles` to list: anime, cyberpunk, casino).
- `--identity, -i`     reference face for identity preservation (IP-Adapter).
- `--identity-scale`   0–1; higher = closer to the face but weaker scene control (default 0.6).
- `--model, -m`        registry model key (default = registry default; see `slop models list`).
- `--steps`, `--guidance, -g`, `--width`, `--height`, `--seed`, `-o`.

## Examples

```
uv run slop image "a neon cat on a rooftop" --style cyberpunk -o outputs/cat.png
uv run slop image "portrait, studio light" --identity me.jpg --seed 42 -o outputs/me.png
```

## Notes
- First run downloads the image model (~7 GB) into the repo-local cache. Comfortable on 16 GB.
- `--identity` uses a global IP-Adapter: strong identity, weaker scene control (a face-only adapter
  is a planned upgrade). Lower `--identity-scale` if the scene/prompt isn't showing.
