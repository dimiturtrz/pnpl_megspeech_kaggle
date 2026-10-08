import pytest

from core.rules import Rule

RULES: tuple[Rule[int], ...] = (
    Rule(lambda n: n < 0, lambda n: f"{n} is negative"),
    Rule(lambda n: n % 2 == 1, lambda n: f"{n} is odd"),
)


def test_enforce() -> None:
    Rule.enforce(4, RULES)
    Rule.enforce(4, ())
    with pytest.raises(ValueError, match="-3 is negative"):
        Rule.enforce(-3, RULES)
    with pytest.raises(ValueError, match="3 is odd"):
        Rule.enforce(3, RULES)
