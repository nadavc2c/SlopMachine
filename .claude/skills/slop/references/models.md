# slop models / assets — the model registry

`src/slopmachine/config/models.yaml` is the single source of truth: each capability maps to a model. Code never
hardcodes model ids — it asks the registry — so swapping a model is a one-line registry edit.

```
uv run slop capabilities [--json]         # what slop can do + the default model per capability
uv run slop models list [--json]          # all capabilities + candidate models (* = default)
uv run slop models show <capability>      # full spec (add --model <key> for a non-default)
uv run slop models download <capability>  # pre-fetch weights into the repo-local cache
uv run slop assets download pose          # supporting ONNX models used by `slop dance`
```

## Notes
- Everything caches in the repo-local `models/` dir (HF_HOME containment) — nothing global.
- Entries with a `subfolder` (adapters / asset bundles) fetch only that subfolder, never a whole repo.
- To refresh a capability to the current-best model, use the **`slop-models`** skill — it researches
  official sources (HF trending, leaderboards, model cards), checks the 16 GB fit, and updates the
  registry row.
