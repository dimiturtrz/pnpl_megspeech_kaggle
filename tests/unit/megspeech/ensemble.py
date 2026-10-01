from pathlib import Path
from typing import override

import torch
from torch import nn

from megspeech.ensemble import Ensemble
from megspeech.model import SentenceNet
from megspeech.scaler import Scaler
from megspeech.wordnet import CHANNELS, WORDS

SAMPLES = 50


class Peaked(nn.Module):
    def __init__(self, word: int) -> None:
        super().__init__()
        self.word = word

    @override
    def forward(self, x: torch.Tensor, onset: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(*mask.shape, WORDS)
        logits[..., self.word] = 10
        return logits, logits


def test_forward() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    members: list[nn.Module] = [Peaked(3), Peaked(3), Peaked(7)]
    context, word = Ensemble(members).forward(torch.zeros(2, 1, 1), torch.zeros(1, 2), mask)
    assert torch.all(context.argmax(-1) == 3)
    assert torch.allclose(context, word)


def test_load(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    model = SentenceNet(Scaler(torch.randn(8, CHANNELS, SAMPLES, generator=generator)))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    loaded = Ensemble.load([tmp_path / "model.pt"] * 2)
    assert len(loaded.members) == 2
    restored = loaded.members[0].state_dict()
    assert all(torch.equal(restored[key], value) for key, value in model.state_dict().items())
