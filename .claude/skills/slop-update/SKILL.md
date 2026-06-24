---
name: slop-update
description: Monthly bleeding-edge refresh for SlopMachine — re-verify every library, dependency, model, template asset, and agent standard (Agent-Skill + Claude-Code-plugin) against the LATEST STABLE official docs, update the in-code PERIODIC-UPDATE-RESEARCH tags, and log it. Use when the user asks to "update everything", "refresh to latest", "monthly update", or "bring deps/models up to date".
when_to_use: Trigger on "update all libraries/deps/models", "refresh to the latest", "monthly update", "are we still bleeding-edge?".
---

# /slop-update — monthly bleeding-edge refresh

Keep SlopMachine on the **latest STABLE, official** libraries, models, and agent standards. Read the
policy + hard constraints first: **`docs/PERIODIC-UPDATE.md`**. Core rule: **verify everything against
live official docs — never assert from memory — and use latest STABLE only (no beta/rc/preview).**

## Procedure

1. **Enumerate the tags.** `rg "PERIODIC-UPDATE-RESEARCH"` across the repo; sort by oldest `last-verified`.
2. **Libraries / dependencies** (`pyproject.toml`): for each pin, fetch the live PyPI JSON / official
   release notes; compare floor vs latest stable. **torch especially:** confirm the chosen CUDA index
   (currently **cu130**) still carries the latest cp314 Blackwell-`sm_120` Windows wheel — PyTorch rotates
   indexes out (that is how 2.11→2.12 stranded us). Verify each upgrade still satisfies the target stack
   in `docs/PERIODIC-UPDATE.md`.
3. **Models** (`src/slopmachine/config/models.yaml`): use the **`slop-models`** skill to re-verify every
   `repo_id` still exists, is ungated/loadable, is diffusers-native on the installed diffusers, is STABLE
   (not preview), and fits 16 GB; surface any newer current-best (e.g. a newer FLUX/Qwen/Wan, or real
   ungated weights for a model that was paper-only).
4. **Template / mascot media** (`docs/`, any catalog assets, `config/styles/*`): verify present,
   license-clean, and still load.
5. **Agent-Skill standard:** re-check `.claude/skills/*/SKILL.md` frontmatter + structure against the
   live agentskills.io specification.
6. **Claude-Code-plugin standard:** re-check `.claude-plugin/marketplace.json` + `plugin.json` against the
   live code.claude.com plugin/marketplace docs (+ SchemaStore schemas); re-check `llms.txt` vs llmstxt.org.
7. **Apply + tag:** make the clearly-current upgrades; bump each `last-verified` date + source URL; `uv lock`
   + `uv sync`.
8. **Verify on-device:** `uv run slop info` (sm_120, ~16 GB) + a smoke `uv run slop image` (and
   `uv run --extra dance slop dance` if touched).
9. **Log it:** append a dated entry to the "Revalidation log" in `docs/PERIODIC-UPDATE.md` — what was
   current, what was bumped, what is newly stale, and any item deliberately NOT adopted (paper-only /
   closed-API / blog-claimed) with the reason.

## Ignore (do not chase)

Blog "open weights" claims with no HF repo; paper-only models with no diffusers-loadable weights;
closed cloud-API-only releases. Adopt only real, ungated, diffusers-native, stable weights.
