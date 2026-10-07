"""Which saved head belongs to which auxiliary term: a run saves one head per nonzero weight, in the order
`Auxiliary.of` builds the terms."""

from collections.abc import Mapping
from typing import ClassVar


class HeadOrder:
    WEIGHTS: ClassVar[tuple[str, ...]] = (
        "speech",
        "semantic",
        "lexical",
        "phonetic",
        "syntactic",
        "tagging",
        "next_word",
        "prev_word",
        "adversary",
    )
    """The `OptimConfig` weights in the order their terms are built."""

    @classmethod
    def position(cls, weights: Mapping[str, float], name: str) -> int:
        """Where the head of weight `name` sits among the heads a run with these `weights` saved."""
        if not weights.get(name):
            raise ValueError(f"{name} has weight 0: no head was trained for it")
        return sum(1 for earlier in cls.WEIGHTS[: cls.WEIGHTS.index(name)] if weights.get(earlier))
