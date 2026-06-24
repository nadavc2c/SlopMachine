import pytest

from slopmachine.config import SlopError, list_styles, load_style


def test_styles_present():
    styles = list_styles()
    assert {"anime", "cyberpunk", "casino"} <= set(styles)


def test_style_apply_formats_prompt():
    preset = load_style("casino")
    positive, negative = preset.apply("a fox mascot")
    assert "a fox mascot" in positive
    assert negative  # casino has a non-empty negative prompt


def test_unknown_style_raises():
    with pytest.raises(SlopError):
        load_style("does-not-exist")
