"""Nested frozen dataclasses as a command line: one flag per leaf, named by its dotted path.

    tree = Tree(RunConfig)
    tree.arguments(parser)                 # --optim.lr, --context.clock, --data.tasks, ...
    config = tree.parse(parser.parse_args())

A saved config reloads through `load`, which also accepts a flat record from before the tree: a key that is no
branch of the root goes to the one leaf of that name.
"""

import argparse
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import fields, is_dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SEPARATOR = "."


class Tree[T: DataclassInstance]:
    """The leaves of a config class whose fields are values or further config classes, each with its default."""

    def __init__(self, root: type[T], default: T | None = None) -> None:
        self.root = root
        self.default = root() if default is None else default

    def leaves(self, node: "DataclassInstance | None" = None, prefix: str = "") -> Iterator[tuple[str, object, object]]:
        """Each leaf's dotted path, declared type and value in `node` (the default tree when None)."""
        node = self.default if node is None else node
        hints = get_type_hints(type(node))
        for field in fields(node):
            value = getattr(node, field.name)
            if is_dataclass(value) and not isinstance(value, type):
                yield from self.leaves(value, f"{prefix}{field.name}{SEPARATOR}")
            else:
                yield f"{prefix}{field.name}", hints[field.name], value

    def arguments(self, parser: argparse.ArgumentParser) -> None:
        """One flag per leaf, defaulting to the tree's value; `coerce` types what the command line gives as text."""
        for path, kind, value in self.leaves():
            flag, dest, shown = f"--{path}", path, "default %(default)s"
            if kind is bool:
                parser.add_argument(flag, dest=dest, default=value, help=shown, action=argparse.BooleanOptionalAction)
            elif get_origin(kind) is tuple:
                parser.add_argument(flag, dest=dest, default=value, help=shown, nargs="*")
            elif isinstance(kind, type) and issubclass(kind, StrEnum):
                parser.add_argument(
                    flag, dest=dest, default=value, help=shown, choices=[member.value for member in kind]
                )
            else:
                parser.add_argument(flag, dest=dest, default=value, help=shown)

    def parse(self, namespace: argparse.Namespace) -> T:
        values = vars(namespace)
        return self.load(self.nest({path: self.coerce(kind, values[path]) for path, kind, _ in self.leaves()}))

    @staticmethod
    def nest(flat: Mapping[str, object]) -> dict[str, object]:
        """`{"a.b": 1}` as `{"a": {"b": 1}}`."""
        nested: dict[str, object] = {}
        groups: dict[str, dict[str, object]] = {}
        for path, value in flat.items():
            head, _, rest = path.partition(SEPARATOR)
            if rest:
                groups.setdefault(head, {})[rest] = value
            else:
                nested[head] = value
        return nested | {head: Tree.nest(group) for head, group in groups.items()}

    def load(self, values: Mapping[str, object]) -> T:
        """The tree with `values` (nested, or flat by leaf name) over its defaults; unknown keys are refused."""
        return self.build(self.default, self.lift(values))

    def lift(self, values: Mapping[str, object]) -> dict[str, object]:
        branches = {field.name for field in fields(self.default)}
        by_name: dict[str, list[str]] = {}
        for path, _, _ in self.leaves():
            by_name.setdefault(path.rsplit(SEPARATOR, 1)[-1], []).append(path)
        flat: dict[str, object] = {}
        for key, value in values.items():
            paths = [key] if key in branches else by_name.get(key, [])
            if len(paths) != 1:
                raise KeyError(f"{key!r} names {len(paths)} leaves of {self.root.__name__}")
            flat[paths[0]] = value
        return self.nest(flat)

    def build[N: DataclassInstance](self, node: N, values: Mapping[str, object]) -> N:
        hints = get_type_hints(type(node))
        changes: dict[str, object] = {}
        for name, value in values.items():
            current = getattr(node, name)
            nested = is_dataclass(current) and isinstance(value, Mapping)
            changes[name] = self.build(current, value) if nested else self.coerce(hints[name], value)
        return replace(node, **changes)

    @staticmethod
    def truth(text: str) -> bool:
        """`true` or `false`, in any case; `bool` itself reads every non-empty text as true."""
        truths = {"true": True, "false": False}
        if text.lower() not in truths:
            raise ValueError(f"{text!r} is neither true nor false")
        return truths[text.lower()]

    @classmethod
    def coerce(cls, kind: object, value: object) -> object:
        """A JSON or command-line value as the leaf's declared type: text parsed (by the kind's own `parse` if it has
        one), enums by value, lists as tuples."""
        if get_origin(kind) is tuple and isinstance(value, Iterable) and not isinstance(value, str):
            return tuple(cls.coerce(get_args(kind)[0], item) for item in value)
        parse = getattr(kind, "parse", None)
        if parse is not None and isinstance(value, str):
            return parse(value)
        if kind is bool and isinstance(value, str):
            return cls.truth(value)
        scalar = isinstance(kind, type) and issubclass(kind, StrEnum | int | float | str)
        if scalar and isinstance(value, str | int | float) and not isinstance(value, kind):
            return kind(value)
        return value
