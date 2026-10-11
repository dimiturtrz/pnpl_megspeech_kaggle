from dataclasses import dataclass
from enum import StrEnum

import pytest

from core.coercion import Coercion, Items, Kept, Parsed, Scalar, Truth


class Colour(StrEnum):
    RED = "red"


@dataclass(frozen=True)
class Tagged:
    tag: str

    @classmethod
    def parse(cls, text: str) -> "Tagged":
        return cls(text.upper())


def test_read() -> None:
    assert Coercion.read(tuple[Tagged, ...], ["a", "b"]) == (Tagged("A"), Tagged("B"))
    assert Coercion.read(bool, "False") is False
    assert Coercion.read(Colour, "red") is Colour.RED
    value = object()
    assert Coercion.read(int, value) is value


def test_items_applies() -> None:
    assert Items().applies(tuple[int, ...], ["1"])
    assert Items().applies(tuple[int, ...], {"a": 1})
    assert not Items().applies(tuple[int, ...], "12")
    assert not Items().applies(tuple[int, ...], 5)
    assert not Items().applies(list[int], ["1"])


def test_items_convert() -> None:
    assert Items().convert(tuple[int, ...], ["1", 2]) == (1, 2)
    assert Items().convert(tuple[tuple[int, ...], ...], [["1"]]) == ((1,),)


def test_parsed_applies() -> None:
    assert Parsed().applies(Tagged, "a")
    assert not Parsed().applies(Tagged, 3)
    assert not Parsed().applies(int, "3")


def test_parsed_convert() -> None:
    assert Parsed().convert(Tagged, "a") == Tagged("A")


def test_truth_applies() -> None:
    assert Truth().applies(bool, "no")
    assert not Truth().applies(bool, 1)
    assert not Truth().applies(int, "1")


def test_truth_convert() -> None:
    assert Truth().convert(bool, "FALSE") is False
    assert Truth().convert(bool, "true") is True
    with pytest.raises(ValueError, match="neither"):
        Truth().convert(bool, "no")


def test_scalar_applies() -> None:
    assert Scalar().applies(float, 1)
    assert Scalar().applies(Colour, "red")
    assert not Scalar().applies(int, True)
    assert not Scalar().applies(int, None)
    assert not Scalar().applies(int | None, "4")


def test_scalar_convert() -> None:
    assert Scalar().convert(Colour, "red") is Colour.RED
    assert Scalar().convert(str, 3) == "3"


def test_kept_applies() -> None:
    assert Kept().applies(object, None)


def test_kept_convert() -> None:
    value = object()
    assert Kept().convert(int, value) is value
