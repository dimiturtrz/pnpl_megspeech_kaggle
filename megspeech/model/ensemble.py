"""Average trained context models in log-probability space (`megspeech.submission` scores and submits)."""

import json
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.config import METRICS
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.input.subjects import Unseen


class Ensemble(nn.Module):
    """Members' log-softmax averaged per head: a geometric mean of their predictive distributions."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def load(cls, paths: list[Path], unseen: Unseen = Unseen.IDENTITY) -> "Ensemble":
        """SentenceNet checkpoints, each reading a subject it has no matrix for through `unseen`."""
        return cls([Checkpoint.load(path, unseen) for path in paths])

    @staticmethod
    def trained(models: Path, name: str) -> list[Path]:
        """The trained runs behind `name`: those an ensemble's record lists, else the run itself."""
        record = models / name / METRICS
        saved: dict[str, list[str]] = json.loads(record.read_text()) if record.exists() else {}
        return [models / member for member in saved.get("members", [name])]

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
