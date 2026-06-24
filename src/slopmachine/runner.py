"""Minimal sequential pipeline runner — the chaining primitive.

Runs stages in order, threading each stage's output dict into the next stage's
inputs. This is the seam that future multi-stage pipelines (e.g. pose -> animate,
or image -> trace) plug into. YAML-defined DAGs can build on this later.
"""

from __future__ import annotations

from typing import Any

from .pipeline.base import Stage


def run_pipeline(stages: list[Stage], **initial: Any) -> dict[str, Any]:
    state: dict[str, Any] = dict(initial)
    for stage in stages:
        state.update(stage.run(**state))
    return state
