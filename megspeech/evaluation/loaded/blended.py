"""Members trained at different `DataConfig.READING`s, each group scored under its own config and blended per word.

The blend is the one `Ensemble` makes inside a group, taken across groups: the groups' log-probabilities averaged per
word, each group weighted by its member count, so every member counts once. A word is identified by its label, token
and onset, not by its row: groups that cut the same words into other windows or spans still blend, and groups whose
word sets differ are refused. The prior is averaged the same way, which
makes the blend's logit adjustment the average of the groups' own.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.scoring.score import Head

Read = Callable[[ExportedModel], tuple[Split, Float[Tensor, "n v"]]]
ABSENT = -1.0
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
    def keys(split: Split) -> Float[np.ndarray, "n k"]:
        """Each row's word key: its label, token and onset (`ABSENT` where no span holds it)."""
        onset = np.full(len(split.y), ABSENT)
        index, times = split.index.cpu().numpy(), split.onset.cpu().numpy()
        onset[index[index != PAD]] = times[index != PAD]
        return np.stack([split.y.cpu().numpy(), split.token.cpu().numpy(), onset], axis=1).astype(np.float64)

    @classmethod
    def order(cls, lead: Split, split: Split) -> Int[Tensor, " n"] | None:
        """The rows of `split` that stand in `lead`'s row order, matched on word key (equal keys pair in row order);
        None when the two hold different words."""
        mine, theirs = cls.keys(lead), cls.keys(split)
        if mine.shape != theirs.shape:
            return None
        first, second = np.lexsort(mine.T[::-1]), np.lexsort(theirs.T[::-1])
        if not np.array_equal(mine[first], theirs[second]):
            return None
        rows = np.empty_like(first)
        rows[first] = second
        return torch.from_numpy(rows)

    @classmethod
    def aligned(cls, lead: Split, split: Split, what: str) -> Int[Tensor, " n"]:
        """`order`, or a refusal naming what was read."""
        rows = cls.order(lead, split)
        if rows is None:
            raise ValueError(f"the groups of a blended ensemble read different words for {what}")
        return rows

    @classmethod
    def orders(cls, reads: list[dict[str, Split]]) -> list[dict[str, Int[Tensor, " n"]]]:
        """Per group and scored domain, its rows in the first group's word order; a refusal when the groups score
        different domains or read different words."""
        lead = reads[0]
        rows = [{name: torch.arange(len(split.y)) for name, split in lead.items()}]
        for domains in reads[1:]:
            if domains.keys() != lead.keys():
                raise ValueError("the groups of a blended ensemble read different words for the scored domains")
            rows.append({name: cls.aligned(lead[name], domains[name], f"the scored domain {name}") for name in lead})
        return rows

    @override
    def domain_logits(self, runs: list[Run]) -> tuple[dict[str, Split], dict[str, dict[Head, Float[Tensor, "n v"]]]]:
        """Each group's domains under its own config, scored in the blend's partition; per domain and head the
        weighted mean of the groups' log-probabilities, after checking that every group read the same words."""
        reads = [replace(group, partition=self.partition).domain_logits(runs) for group in self.groups]
        lead, lead_logits = reads[0]
        rows = self.orders([domains for domains, _ in reads])
        logits = {
            name: {
                head: torch.stack(
                    [
                        w * read[1][name][head][order[name]]
                        for w, read, order in zip(self.weights, reads, rows, strict=True)
                    ]
                ).sum(0)
                for head in Head
            }
            for name in lead_logits
        }
        return lead, logits

    def blend(self, read: Read, what: str) -> tuple[Split, Float[Tensor, "n v"]]:
        """The first group's split and the weighted mean of every group's log-probabilities for it, each group's rows
        put in the first group's order by word key."""
        reads = (read(group) for group in self.groups)
        lead, first = next(reads)
        parts = [first]
        for split, part in reads:
            parts.append(part[self.aligned(lead, split, what)])
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
