import pytest
import typer

from slopmachine.cli import _safe, _slug
from slopmachine.config import SlopError


def test_slug_basic():
    assert _slug("A Neon Cat!! on Mars") == "a-neon-cat-on-mars"


def test_slug_empty_fallback():
    assert _slug("!!!") == "image"


def test_safe_passthrough():
    assert _safe(lambda x: x + 1, 1) == 2


def test_safe_converts_sloperror_to_exit():
    def boom():
        raise SlopError("nope")

    with pytest.raises(typer.Exit):
        _safe(boom)
