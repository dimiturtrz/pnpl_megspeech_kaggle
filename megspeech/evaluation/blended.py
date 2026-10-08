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
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.scoring.score import Head

Read = Callable[[ExportedModel], tuple[Split, Float[Tensor, "n v"]]]
WORDS = ("y", "token", "index", "onset")
REFUSED = "a blended ensemble reads its groups through run_context and kaggle_chosen only"


@dataclass(frozen=True, kw_only=True)
class Blended(ExportedModel):
    """`groups` blended as one model; the first group's reader, model and data stand for it where one is read."""

    groups: tuple[ExportedModel, ...]
    weights: tuple[float, ...]

    @classmethod
    def of(cls, groups: Sequence[ExportedModel]) -> "Blended":
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

    @staticmethod
    def agree(lead: Split, split: Split) -> bool:
        """Whether `split` holds `lead`'s words in the same order."""
        return all(torch.equal(getattr(lead, name), getattr(split, name)) for name in WORDS)

    @override
    def domain_logits(self, runs: list[Run]) -> tuple[dict[str, Split], dict[str, dict[Head, Float[Tensor, "n v"]]]]:
        """Each group's domains under its own config, scored in the blend's partition; per domain and head the
        weighted mean of the groups' log-probabilities, after checking that every group read the same words."""
        reads = [replace(group, partition=self.partition).domain_logits(runs) for group in self.groups]
        lead, _ = reads[0]
        for domains, _ in reads[1:]:
            if domains.keys() != lead.keys() or not all(self.agree(lead[name], domains[name]) for name in lead):
                raise ValueError("the groups of a blended ensemble read different words for the scored domains")
        logits = {
            name: {
                head: torch.stack(
                    [w * logits[name][head] for w, (_, logits) in zip(self.weights, reads, strict=True)]
                ).sum(0)
                for head in Head
            }
            for name in lead
        }
        return lead, logits

    def blend(self, read: Read, what: str) -> tuple[Split, Float[Tensor, "n v"]]:
        """The first group's split and the weighted mean of every group's log-probabilities for it, after checking
        that every group read the same words in the same order."""
        reads = (read(group) for group in self.groups)
        lead, first = next(reads)
        parts = [first]
        for split, part in reads:
            if not self.agree(lead, split) or part.shape != first.shape:
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
