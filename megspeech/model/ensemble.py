"""Average trained context models in log-probability space (`megspeech.submission.ensemble` scores and submits)."""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.input.subjects import Unseen
from megspeech.model.sentence import SentenceNet


class Ensemble(nn.Module):
    """Members' log-softmax averaged per head: a geometric mean of their predictive distributions."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def load(cls, paths: list[Path], unseen: Unseen = Unseen.IDENTITY) -> "Ensemble":
        """SentenceNet checkpoints, each reading a subject it has no matrix for through `unseen`."""
        return cls([SentenceNet.load(path, unseen) for path in paths])

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        outputs = [member(x, onset, slot, subject) for member in self.members]
        context, word = (
            torch.stack([out.float().log_softmax(-1) for out in head]).mean(0) for head in zip(*outputs, strict=True)
        )
        return context, word
