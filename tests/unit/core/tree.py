import argparse
import math
from dataclasses import dataclass, field
from enum import StrEnum

import pytest

from core.tree import Tree


class Shape(StrEnum):
    ROUND = "round"
    SQUARE = "square"


@dataclass(frozen=True)
class Inner:
    rate: float = 0.5
    shape: Shape = Shape.ROUND


@dataclass(frozen=True)
class Outer:
    inner: Inner = Inner()
    other: Inner = Inner(rate=2.0)
    names: tuple[str, ...] = ()
    steps: int = 3
    fast: bool = True
    limit: float = math.inf
    sizes: tuple[int, ...] = field(default=(1, 2))


def tree() -> Tree[Outer]:
    return Tree(Outer)


def parsed(*argv: str) -> Outer:
    parser = argparse.ArgumentParser()
    tree().arguments(parser)
    return tree().parse(parser.parse_args(argv))


def test_leaves() -> None:
    leaves = {path: value for path, _, value in tree().leaves()}
    assert leaves["inner.rate"] == 0.5
    assert leaves["other.rate"] == 2.0
    assert leaves["steps"] == 3
    assert len(leaves) == 9


def test_arguments() -> None:
    parser = argparse.ArgumentParser()
    tree().arguments(parser)
    assert vars(parser.parse_args([]))["other.rate"] == 2.0
    assert vars(parser.parse_args(["--no-fast", "--names", "a", "b"]))["names"] == ["a", "b"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--inner.shape", "oval"])


def test_parse() -> None:
    assert parsed() == Outer()
    out = parsed("--inner.shape", "square", "--names", "a", "b", "--no-fast", "--sizes", "4", "--other.rate", "1")
    assert out == Outer(Inner(shape=Shape.SQUARE), Inner(rate=1.0), ("a", "b"), fast=False, sizes=(4,))


def test_nest() -> None:
    assert Tree.nest({"a.b": 1, "a.c": 2, "d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}


def test_load() -> None:
    assert tree().load({"inner": {"rate": 1, "shape": "square"}, "names": ["x"]}) == Outer(
        Inner(1.0, Shape.SQUARE), names=("x",)
    )
    assert tree().load({"limit": math.inf}).limit == math.inf


def test_lift() -> None:
    assert tree().lift({"steps": 5, "inner": {"rate": 1}}) == {"steps": 5, "inner": {"rate": 1}}
    with pytest.raises(KeyError, match="'rate' names 2 leaves"):
        tree().lift({"rate": 1.0})
    with pytest.raises(KeyError, match="names 0 leaves"):
        tree().lift({"absent": 1})


def test_build() -> None:
    assert tree().build(Inner(), {"rate": 3}) == Inner(rate=3.0)
    with pytest.raises(AttributeError):
        tree().build(Inner(), {"absent": 1})


def test_coerce() -> None:
    assert Tree.coerce(tuple[str, ...], ["a"]) == ("a",)
    assert Tree.coerce(Shape, "square") is Shape.SQUARE
    assert Tree.coerce(float, 1) == 1.0
    assert isinstance(Tree.coerce(float, 1), float)
    assert Tree.coerce(int, "4") == 4
    assert Tree.coerce(tuple[int, ...], ["1", "2"]) == (1, 2)
    assert Tree.coerce(float, "inf") == math.inf
    assert Tree.coerce(bool, False) is False
    assert Tree.coerce(bool, "false") is False
    assert Tree.coerce(bool, "True") is True


def test_truth() -> None:
    assert Tree.truth("FALSE") is False
    assert Tree.truth("true") is True
    with pytest.raises(ValueError, match="neither"):
        Tree.truth("no")
