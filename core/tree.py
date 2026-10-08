"""Nested frozen dataclasses as a command line: one flag per leaf, named by its dotted path.

    tree = Tree(RunConfig)
    tree.arguments(parser)                 # --optim.lr, --context.clock, --data.tasks, ...
    config = tree.parse(parser.parse_args())

A saved config reloads through `load`, which also accepts a flat record from before the tree: a key that is no
branch of the root goes to the one leaf of that name, and a key misplaced inside a branch goes there too. A config
class retires what it dropped through three optional class attributes: `RETIRED` (`Retired` per dropped field),
`RENAMED` (old leaf name to new) and `LEGACY` (what a record that predates a field implies, applied by a saved load).
"""

import argparse
import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SEPARATOR = "."
RETIRED = "RETIRED"
RENAMED = "RENAMED"
LEGACY = "LEGACY"


@dataclass(frozen=True)
class Retired:
    """A field its config class no longer has. A record's value equal to `default` is dropped; one in `becomes` is
    dropped for the fields it maps to (never over a field the record sets), provided each sibling in `given` holds its
    stated value (a sibling the record omits holds its own retired default); any other value is refused."""

    default: object
    reason: str = ""
    becomes: Mapping[object, Mapping[str, object]] = dataclasses.field(default_factory=dict)
    given: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def resolve(self, name: str, record: Mapping[str, object], table: Mapping[str, "Retired"]) -> Mapping[str, object]:
        """The fields a record's value of `name` stands for, or a refusal naming it; JSON's lists read as tuples."""
        value = record[name]
        value = tuple(value) if isinstance(value, list) else value
        if value == self.default:
            return {}
        held = all(record.get(key, table[key].default) == wanted for key, wanted in self.given.items())
        if value not in self.becomes or not held:
            raise ValueError(self.reason or f"{name}={value!r} is retired, and only its default {self.default!r} loads")
        return self.becomes[value]


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

    def load(self, values: Mapping[str, object], *, saved: bool = False) -> T:
        """The tree with `values` (nested, or flat by leaf name) over its defaults; unknown keys are refused. A `saved`
        record also takes each class's `LEGACY` fields it does not set."""
        lifted = self.lift(values)
        return self.build(self.default, self._aged(lifted) if saved else lifted)

    def _nodes(self, kind: type | None = None, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], type]]:
        """Each config class of the tree with its branch path from the root."""
        kind = self.root if kind is None else kind
        yield path, kind
        for name, hint in get_type_hints(kind).items():
            if is_dataclass(hint) and name in {field.name for field in fields(kind)}:
                yield from self._nodes(hint, (*path, name))

    def _aged(self, nested: Mapping[str, object]) -> dict[str, object]:
        """`nested` with, for each class that declares `LEGACY`, those fields it leaves unset."""
        aged = dict(nested)
        for path, kind in self._nodes():
            aged = self._defaulted(aged, path, getattr(kind, LEGACY, {}))
        return aged

    @staticmethod
    def _defaulted(
        node: Mapping[str, object], path: tuple[str, ...], legacy: Mapping[str, object]
    ) -> dict[str, object]:
        """`node` with `legacy` under the branch at `path`, wherever the record sets none of it."""
        if not path:
            return {**legacy, **node}
        child = node.get(path[0])
        return {**node, path[0]: Tree._defaulted(child if isinstance(child, Mapping) else {}, path[1:], legacy)}

    def _table(self, attribute: str) -> dict[str, object]:
        """The `attribute` tables of every class of the tree, merged."""
        return {name: value for _, kind in self._nodes() for name, value in getattr(kind, attribute, {}).items()}

    def lift(self, values: Mapping[str, object]) -> dict[str, object]:
        by_name: dict[str, list[str]] = {}
        for path, _, _ in self.leaves():
            by_name.setdefault(path.rsplit(SEPARATOR, 1)[-1], []).append(path)
        renames = self._table(RENAMED)
        flat, strays = self._place(self.root, "", values, renames)
        for key, value in strays.items():
            paths = by_name.get(str(renames.get(key, key)), [])
            if len(paths) != 1:
                raise KeyError(f"{key!r} names {len(paths)} leaves of {self.root.__name__}")
            flat[paths[0]] = value
        return self.nest(flat)

    def _place(
        self, kind: type, prefix: str, values: Mapping[str, object], renames: Mapping[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Sort a record into the leaves it names by path and the strays no class here has (found by leaf name
        anywhere), after `kind` retires its dropped fields."""
        hints = get_type_hints(kind)
        names = {field.name for field in fields(kind)}
        flat: dict[str, object] = {}
        strays: dict[str, object] = {}
        for key, value in self._retire(kind, values).items():
            if key in names and is_dataclass(hints[key]) and isinstance(value, Mapping):
                below, loose = self._place(hints[key], f"{prefix}{key}{SEPARATOR}", value, renames)
                flat |= below
                strays |= loose
            elif key in names and key not in renames:
                flat[f"{prefix}{key}"] = value
            else:
                strays[key] = value
        return flat, strays

    @staticmethod
    def _retire(kind: type, values: Mapping[str, object]) -> dict[str, object]:
        """`values` without the fields `kind` retired, plus what each stands for that the record does not set."""
        retired: Mapping[str, Retired] = getattr(kind, RETIRED, {})
        derived: dict[str, object] = {}
        for name in values.keys() & retired.keys():
            derived |= retired[name].resolve(name, values, retired)
        return derived | {key: value for key, value in values.items() if key not in retired}

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
