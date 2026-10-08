"""A config's validity as data: each rule names when a node is broken and what to say about it."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule[T]:
    """One constraint on a `T`: `broken` holds when the node violates it, `message` says how."""

    broken: Callable[[T], bool]
    message: Callable[[T], str]

    @staticmethod
    def enforce(node: T, rules: Iterable["Rule[T]"]) -> None:
        """Raise `ValueError` with the first broken rule's message, in the order given."""
        for rule in rules:
            if rule.broken(node):
                raise ValueError(rule.message(node))
