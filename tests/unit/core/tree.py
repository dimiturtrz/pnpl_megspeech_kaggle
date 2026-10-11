import argparse
import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import ClassVar

import pytest

from core.tree import Retired, Tree


class Shape(StrEnum):
    ROUND = "round"
    SQUARE = "square"


@dataclass(frozen=True)
class Inner:
    rate: float = 0.5
    shape: Shape = Shape.ROUND


@dataclass(frozen=True)
class Pair:
    name: str
    count: int = 1

    @classmethod
    def parse(cls, text: str) -> "Pair":
        name, _, count = text.partition(":")
        return cls(name, int(count) if count else 1)


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


@dataclass(frozen=True)
class Group:
    FLAT: ClassVar[bool] = True
    gain: float = 1.0


@dataclass(frozen=True)
class Split:
    group: Group = Group()
    inner: Inner = Inner()


def test_flags() -> None:
    assert tree().flags("inner.rate") == ("--inner.rate",)
    assert Tree(Split).flags("group.gain") == ("--group.gain", "--gain")
    assert Tree(Split).flags("inner.rate") == ("--inner.rate",)


def test_flat_branch_answers_to_both_flags() -> None:
    parser = argparse.ArgumentParser()
    Tree(Split).arguments(parser)
    assert Tree(Split).parse(parser.parse_args(["--gain", "2"])).group.gain == 2.0
    assert Tree(Split).parse(parser.parse_args(["--group.gain", "3"])).group.gain == 3.0


def test_flat_record_of_a_split_group_loads() -> None:
    assert Tree(Split).load({"gain": 2.0}, saved=True) == Split(Group(2.0))


@dataclass(frozen=True)
class Scoped:
    split: Split = Split()
    gain: float = 0.0


def test_stray_resolves_inside_its_branch_first() -> None:
    assert Tree(Scoped).load({"split": {"gain": 2.0}, "gain": 1.0}) == Scoped(Split(Group(2.0)), 1.0)
    with pytest.raises(KeyError, match="'gain' names 2 leaves"):
        Tree(Scoped).lift({"split": {"inner": {"gain": 2.0}}})


def test_flat() -> None:
    assert Tree.flat({"group": {"gain": 2.0}, "steps": 3}) == {"gain": 2.0, "steps": 3}
    assert Tree.flat({"gain": 2.0}) == {"gain": 2.0}


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
    assert Tree.coerce(Pair, "a:2") == Pair("a", 2)
    assert Tree.coerce(tuple[Pair, ...], ["a", "b:3"]) == (Pair("a", 1), Pair("b", 3))
    assert Tree.coerce(Pair, Pair("c", 4)) == Pair("c", 4)
    for kind, value, expected in COERCED:
        out = Tree.coerce(kind, value)
        assert (type(out), out) == (type(expected), expected), (kind, value)
    for kind, value in KEPT:
        assert Tree.coerce(kind, value) is value, (kind, value)
    for kind, value in REFUSED:
        with pytest.raises(ValueError):
            Tree.coerce(kind, value)


class Name(str):
    __slots__ = ()

    @classmethod
    def parse(cls, text: str) -> "Name":
        return cls(text.upper())


class Level(IntEnum):
    LOW = 1
    HIGH = 2


COERCED: tuple[tuple[object, object, object], ...] = (
    (tuple[int, ...], ("1", 2), (1, 2)),
    (tuple[int, ...], [], ()),
    (tuple[str, int], ["1", 2], ("1", "2")),
    (tuple[tuple[int, ...], ...], [["1"], []], ((1,), ())),
    (tuple[str, ...], {"a": 1}, ("a",)),
    (Name, "ab", Name("AB")),
    (Name, 3, Name("3")),
    (bool, "TRUE", True),
    (bool, 1, True),
    (bool, 0.0, False),
    (int, 4.7, 4),
    (int, " 4 ", 4),
    (float, True, 1.0),
    (float, "1e3", 1000.0),
    (str, 3, "3"),
    (str, 1.5, "1.5"),
    (Level, 2, Level.HIGH),
)
"""(kind, value, what it reads as, type included), one per equivalence class of `coerce` and its boundaries."""

KEPT: tuple[tuple[object, object], ...] = (
    (tuple[int, ...], "12"),
    (tuple[int, ...], 5),
    (int, True),
    (bool, True),
    (bool, None),
    (int, None),
    (int, [1]),
    (str, "x"),
    (Shape, Shape.ROUND),
    (Pair, {"name": "a"}),
    (Pair, 3),
    (int | None, "4"),
    (dict[str, int], {"a": "1"}),
    (Path, "a/b"),
)
"""(kind, value) that `coerce` hands back as the very object it was given."""

REFUSED: tuple[tuple[object, object], ...] = ((bool, "yes"), (int, "4.5"), (Shape, "oval"), (Level, 3), (Level, "1"))
"""(kind, value) that `coerce` refuses with a `ValueError`."""


@dataclass(frozen=True)
class Knobs:
    RENAMED: ClassVar[dict[str, str]] = {"old": "base"}
    base: str = "a"
    gain: float = 1.0


@dataclass(frozen=True)
class Old:
    RETIRED: ClassVar[dict[str, Retired]] = {
        "gone": Retired(0),
        "mode": Retired("x", becomes={"y": {"steps": 9}}),
        "pair": Retired(False, "pair refused", {True: {}}, {"gone": 0}),
    }
    LEGACY: ClassVar[dict[str, object]] = {"steps": 7}
    knobs: Knobs = Knobs()
    steps: int = 3
    name: str = ""


def test_retired_default_dropped() -> None:
    assert Tree(Old).load({"gone": 0, "mode": "x"}) == Old()


def test_retired_non_default_refused() -> None:
    with pytest.raises(ValueError, match="gone=2 is retired"):
        Tree(Old).load({"gone": 2})


def test_retired_becomes_yields_to_the_record() -> None:
    assert Tree(Old).load({"mode": "y"}).steps == 9
    assert Tree(Old).load({"mode": "y", "steps": 4}).steps == 4


def test_retired_given_sibling() -> None:
    assert Tree(Old).load({"pair": True}) == Old()
    with pytest.raises(ValueError, match="pair refused"):
        Tree(Old).load({"pair": True, "gone": 0, "mode": "x", "name": "n", "steps": 1} | {"pair": 2})


def test_unknown_refused() -> None:
    with pytest.raises(KeyError, match="nowhere"):
        Tree(Old).load({"nowhere": 1})


def test_stray_in_branch_moves_to_its_leaf() -> None:
    assert Tree(Old).load({"knobs": {"gain": 2.0, "name": "n"}}) == Old(Knobs(gain=2.0), name="n")


def test_renamed_leaf() -> None:
    assert Tree(Old).load({"knobs": {"old": "b"}}).knobs.base == "b"
    assert Tree(Old).load({"old": "c"}).knobs.base == "c"


def test_legacy_only_when_saved() -> None:
    assert Tree(Old).load({}).steps == 3
    assert Tree(Old).load({}, saved=True).steps == 7
    assert Tree(Old).load({"steps": 5}, saved=True).steps == 5


def test_legacy_of_a_branch() -> None:
    @dataclass(frozen=True)
    class Holder:
        old: Old = Old()

    assert Tree(Holder).load({}, saved=True).old.steps == 7


def test_resolve() -> None:
    entry = Retired(0, becomes={7: {"steps": 1}}, given={"mate": True})
    table = {"mate": Retired(False)}
    assert entry.resolve("n", {"n": 0}, table) == {}
    assert entry.resolve("n", {"n": 7, "mate": True}, table) == {"steps": 1}
    with pytest.raises(ValueError, match="n=7 is retired"):
        entry.resolve("n", {"n": 7}, table)
    with pytest.raises(ValueError, match="n=1 is retired"):
        entry.resolve("n", {"n": 1, "mate": True}, table)


def test_resolve_reads_a_list_as_its_tuple() -> None:
    entry = Retired((3,))
    assert entry.resolve("k", {"k": [3]}, {}) == {}
    with pytest.raises(ValueError, match=r"k=\(3, 5\) is retired"):
        entry.resolve("k", {"k": [3, 5]}, {})
