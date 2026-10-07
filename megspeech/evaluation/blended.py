"""Members trained at different `DataConfig.READING`s, each group scored under its own config and blended per word.

The blend is the one `Ensemble` makes inside a group, taken across groups: the groups' log-probabilities averaged per
word, each group weighted by its member count, so every member counts once. The prior is averaged the same way, which
makes the blend's logit adjustment the average of the groups' own.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import override

import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel

Read = Callable[[LoadedModel], tuple[Split, Float[Tensor, "n v"]]]
WORDS = ("y", "token", "index", "onset")
REFUSED = "a blended ensemble reads its groups through run_context and kaggle_chosen only"


@dataclass(frozen=True, kw_only=True)
class Blended(LoadedModel):
    """`groups` blended as one model; the first group's reader, model and data stand for it where one is read."""

    groups: tuple[LoadedModel, ...]
    weights: tuple[float, ...]

    @classmethod
    def of(cls, groups: Sequence[LoadedModel]) -> "Blended":
        members = [len(group.names) for group in groups]
        weights = tuple(count / sum(members) for count in members)
        prior = torch.stack([w * group.log_prior for w, group in zip(weights, groups, strict=True)]).sum(0)
        seen = frozenset().union(*(group.evaluator.seen for group in groups))
        lead = groups[0]
        return cls(
            root=lead.root,
            data=lead.data,
            device=lead.device,
            reading=lead.reading,
            geometry=lead.geometry,
            partition=lead.partition,
            model=lead.model,
            evaluator=replace(lead.evaluator, seen=seen),
            log_prior=prior,
            names=tuple(name for group in groups for name in group.names),
            groups=tuple(groups),
            weights=weights,
        )

    @property
    @override
    def adapted_on(self) -> int:
        return max(group.adapted_on for group in self.groups)

    def blend(self, read: Read, what: str) -> tuple[Split, Float[Tensor, "n v"]]:
        """The first group's split and the weighted mean of every group's log-probabilities for it, after checking
        that every group read the same words in the same order."""
        reads = (read(group) for group in self.groups)
        lead, first = next(reads)
        parts = [first]
        for split, part in reads:
            same = all(torch.equal(getattr(lead, name), getattr(split, name)) for name in WORDS)
            if not same or part.shape != first.shape:
                raise ValueError(f"the groups of a blended ensemble read different words for {what}")
            parts.append(part)
        return lead, torch.stack([w * part for w, part in zip(self.weights, parts, strict=True)]).sum(0)

    @override
    def run_context(self, run: Run) -> tuple[Split, Float[Tensor, "n v"]]:
        return self.blend(lambda group: group.run_context(run), run.name)

    @override
    def kaggle_chosen(self, track: Track) -> tuple[Split, Float[Tensor, "n v"]]:
        return self.blend(lambda group: group.kaggle_chosen(track), f"the {track} Kaggle test")

    @override
    def context(self, split: Split) -> Float[Tensor, "n v"]:
        raise NotImplementedError(REFUSED)

    @override
    def splits(self, runs: list[Run]) -> dict[Run, tuple[pd.DataFrame, Split]]:
        raise NotImplementedError(REFUSED)

    @override
    def domains(self, held: dict[str, list[Run]]) -> tuple[dict[str, Split], pd.DataFrame]:
        raise NotImplementedError(REFUSED)

    @override
    def kaggle(self, track: Track) -> Split:
        raise NotImplementedError(REFUSED)
