from pathlib import Path

import pytest
import torch

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.branch import Branch
from megspeech.model.config import ContextConfig
from megspeech.model.encoding import ENCODER
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import Unseen
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)
SLOT = torch.tensor([[0, 1, 3], [2, 3, 3]])


def saved(directory: Path) -> Path:
    directory.mkdir(parents=True)
    scale = Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR))
    torch.save(SentenceNet(scale, CONTEXT).state_dict(), directory / "model.pt")
    CONTEXT.save(directory)
    return directory


def branch(directory: Path) -> Branch:
    return Branch.load(saved(directory), CONTEXT, CHANNELS)


def test_load(tmp_path: Path) -> None:
    directory = saved(tmp_path / "stem")
    made = Branch.load(directory, CONTEXT, CHANNELS, Unseen.IDENTITY)
    state = torch.load(directory / "model.pt")
    assert made.name == "stem"
    assert all(
        torch.equal(made.state_dict()[key], state[key]) for key in made.state_dict() if key.split(".")[0] in ENCODER
    )
    with pytest.raises(ValueError, match="reads"):
        Branch.load(directory, CONTEXT, CHANNELS + 1)


def test_projected(tmp_path: Path) -> None:
    made = branch(tmp_path / "stem").eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    subject = torch.zeros(3, dtype=torch.long)
    out = made.projected(SpanInput(x, torch.zeros(SLOT.shape), SLOT, subject))
    assert out.shape == (2, 3, CONTEXT.width)
    assert torch.allclose(out[0, 0], made.project(made.embed(x, subject)[0]), atol=1e-6)


def test_many(tmp_path: Path) -> None:
    saved(tmp_path / "a")
    saved(tmp_path / "b")
    made = Branch.many([tmp_path / "a", tmp_path / "b"], CONTEXT, CHANNELS)
    assert [one.name for one in made] == ["a", "b"]
