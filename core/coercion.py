"""How a config leaf's declared type reads a JSON or command-line value: the first of `COERCIONS` that applies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from typing import ClassVar, cast, get_args, get_origin, override

PARSE = "parse"


class Coercion(ABC):
    """One way a leaf's declared type reads a JSON or command-line value."""

    @staticmethod
    def read(kind: object, value: object) -> object:
        """`value` as `kind`, by the first of `COERCIONS` that applies."""
        return next(rule for rule in COERCIONS if rule.applies(kind, value)).convert(kind, value)

    @abstractmethod
    def applies(self, kind: object, value: object) -> bool:
        """Whether this is how `kind` reads `value`."""

    @abstractmethod
    def convert(self, kind: object, value: object) -> object:
        """`value` as `kind`."""


class Items(Coercion):
    """A list as the tuple its leaf declares, each item read as the tuple's first argument."""

    @override
    def applies(self, kind: object, value: object) -> bool:
        return get_origin(kind) is tuple and isinstance(value, Iterable) and not isinstance(value, str)

    @override
    def convert(self, kind: object, value: object) -> object:
        return tuple(Coercion.read(get_args(kind)[0], item) for item in cast("Iterable[object]", value))


class Parsed(Coercion):
    """Text read by the kind's own `parse`."""

    @override
    def applies(self, kind: object, value: object) -> bool:
        return getattr(kind, PARSE, None) is not None and isinstance(value, str)

    @override
    def convert(self, kind: object, value: object) -> object:
        return getattr(kind, PARSE)(value)


class Truth(Coercion):
    """`true` or `false`, in any case, as a bool; `bool` itself reads every non-empty text as true."""

    TRUTHS: ClassVar[dict[str, bool]] = {"true": True, "false": False}

    @override
    def applies(self, kind: object, value: object) -> bool:
        return kind is bool and isinstance(value, str)

    @override
    def convert(self, kind: object, value: object) -> object:
        text = str(value).lower()
        if text not in self.TRUTHS:
            raise ValueError(f"{value!r} is neither true nor false")
        return self.TRUTHS[text]


class Scalar(Coercion):
    """A number or text as the number, text or enum (by value) its leaf declares, unless it already is one."""

    @override
    def applies(self, kind: object, value: object) -> bool:
        scalar = isinstance(kind, type) and issubclass(kind, StrEnum | int | float | str)
        return scalar and isinstance(value, str | int | float) and not isinstance(value, cast("type", kind))

    @override
    def convert(self, kind: object, value: object) -> object:
        return cast("type", kind)(value)


class Kept(Coercion):
    """Any other value, as given."""

    @override
    def applies(self, kind: object, value: object) -> bool:
        return True

    @override
    def convert(self, kind: object, value: object) -> object:
        return value


COERCIONS: tuple[Coercion, ...] = (Items(), Parsed(), Truth(), Scalar(), Kept())
"""Tried in order: the first that applies reads the value."""
