import pytest

from megspeech.objective.auxiliary.order import HeadOrder


def test_position() -> None:
    """A term's head sits after the heads of the earlier terms with nonzero weight."""
    weights = {"semantic": 0.5, "next_word": 0.3, "prev_word": 0.3, "speech": 0.0}
    assert [HeadOrder.position(weights, name) for name in ("semantic", "next_word", "prev_word")] == [0, 1, 2]
    assert HeadOrder.position({"next_word": 0.3, "prev_word": 0.3}, "prev_word") == 1
    with pytest.raises(ValueError, match="no head was trained"):
        HeadOrder.position({"next_word": 0.3}, "prev_word")
