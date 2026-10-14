"""Where each scored row sits in its recording: the span (sentence) it belongs to and its place in that span.

A pool's neighbours are only a repeat of the same word when they come from another span; rows of one span overlap in
signal (1 s windows of words ~0.25 s apart), so same-span neighbours can be trivial. The Kaggle test gives the same
span grouping through its onsets, as the scoring already uses.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.studies.knnreadout import SCORED

SCRATCH = "zz-tns"


@dataclass(frozen=True)
class Layout:
    """Each scored row's span number and its position within that span, in the dump's row order."""

    span: Int[Tensor, " n"]
    place: Int[Tensor, " n"]

    @classmethod
    def of(cls, split: Split, novel: Int[Tensor, " n"]) -> "Layout":
        """The `novel` words of `split` placed by the span index."""
        index = split.index
        valid = index != PAD
        spans = torch.arange(len(index), device=index.device)[:, None].expand_as(index)
        places = torch.arange(index.shape[1], device=index.device)[None].expand_as(index)
        span = torch.full((len(split.y),), -1, device=index.device)
        place = torch.full_like(span, -1)
        span[index[valid]] = spans[valid]
        place[index[valid]] = places[valid]
        return cls(span[novel].cpu(), place[novel].cpu())

    @classmethod
    def alone(cls, n: int) -> "Layout":
        """`n` rows that each stand in a span of their own."""
        return cls(torch.arange(n), torch.zeros(n, dtype=torch.long))

    @staticmethod
    def path(root: DataRoot, base: str) -> Path:
        return root.models() / SCRATCH / f"{base}--layout.pt"

    @staticmethod
    def build(  # devtools-ignore: test-mirror
        root: DataRoot, scoring: Scoring, device: torch.device, base: str
    ) -> dict[Partition, dict[str, "Layout"]]:
        """The layout of every row `Collector.build` dumps for `base`, domain by domain."""
        loaded = Loader.of(Group(root, scoring, device), [base])
        runs = Run.discover(root.raw())
        out = {}
        for partition in SCORED:
            held = replace(loaded, partition=partition)
            domains, _ = held.domains(held.ruled(runs))
            counted = held.evaluator.eligible(domains)
            out[partition] = {n: Layout.of(s, s.novel(held.evaluator.seen)) for n, s in counted.items()}
        return out

    @staticmethod
    def save(layouts: dict[Partition, dict[str, "Layout"]], path: Path) -> None:  # devtools-ignore: test-mirror
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({p: {n: (v.span, v.place) for n, v in d.items()} for p, d in layouts.items()}, path)

    @staticmethod
    def load(path: Path) -> dict[Partition, dict[str, "Layout"]]:  # devtools-ignore: test-mirror
        saved = torch.load(path, weights_only=False)
        return {Partition(p): {n: Layout(*v) for n, v in d.items()} for p, d in saved.items()}
