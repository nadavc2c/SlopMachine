"""The Stage interface — the modular unit the whole toolkit is built from.

Every generative step (image, video, pose, animate, music, trace, ...) is a Stage
with a uniform ``run(**inputs) -> dict`` contract. New capabilities = new Stage
subclasses; pipelines are just stages chained by the runner. Adding an SVG-tracing
step later means dropping in a TraceStage — no changes to existing stages.
"""

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute the stage and return a dict of named outputs.

        Outputs from one stage become inputs to the next when chained by the runner,
        so keep output keys descriptive (e.g. ``image``, ``video``, ``pose``).
        """
        ...
