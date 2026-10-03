"""Average trained context models in log-probability space (`megspeech.submission.ensemble` scores and submits)."""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.onsets import ContextConfig
from megspeech.model.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.subjects import Unseen

SCALE_MEDIAN = "scale.median"
"""A SentenceNet state's per-channel scaler median: one entry per channel the model reads."""


class Ensemble(nn.Module):
    """Members' log-softmax averaged per head: a geometric mean of their predictive distributions."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def load(cls, paths: list[Path], unseen: Unseen = Unseen.IDENTITY) -> "Ensemble":
        """SentenceNet checkpoints, each reading a subject it has no matrix for through `unseen`; each state dict
        overwrites the placeholder scaler's statistics, shaped as the state's own: MEG channels, or a cached
        encoder's feature channels."""
        members: list[nn.Module] = []
        for path in paths:
            context = ContextConfig.load(path.parent)
            state = torch.load(path, map_location="cpu")
            placeholder = Scaler(torch.zeros(1, state[SCALE_MEDIAN].shape[0], 1))
            model = SentenceNet(placeholder, context=context, unseen=unseen)
            model.load_state_dict(state)
            members.append(model)
        return cls(members)

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        outputs = [member(x, onset, slot, subject) for member in self.members]
        context = torch.stack([out[0].float().log_softmax(-1) for out in outputs]).mean(0)
        word = torch.stack([out[1].float().log_softmax(-1) for out in outputs]).mean(0)
        return context, word
