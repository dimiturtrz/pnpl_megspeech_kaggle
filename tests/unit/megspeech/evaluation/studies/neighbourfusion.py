import json
from dataclasses import asdict
from pathlib import Path
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput, Split
from megspeech.data.config import METRICS
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies.neighbourfusion import HEADS, Member, Neighbours
from megspeech.model.config import ContextConfig
from megspeech.model.ensemble import Ensemble
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.objective.auxiliary.neighbour import NextWordTerm, PreviousWordTerm
from megspeech.training.auxiliary import AuxiliaryConfig
from megspeech.training.config import RunConfig
from megspeech.training.optim import OptimConfig

CPU = torch.device("cpu")
LOUD = 10.0


def real_net() -> SentenceNet:
    return SentenceNet(
        nn.Identity(), ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
    )


class Embedding(nn.Module):
    """Stands in for a `SentenceNet`: each position's embedding is its window's channel means."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x

    def heard(self, spans: SpanInput) -> tuple[None, Tensor, None, None]:
        return None, nn.functional.pad(spans.x.mean(-1), (0, 0, 0, 1))[spans.slot], None, None


def constant(label: int) -> nn.Linear:
    head = nn.Linear(CHANNELS, VOCAB_SIZE)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    head.bias.data[label] = LOUD
    return head


def member() -> Member:
    return Member(cast("SentenceNet", Embedding()), constant(1), constant(2))


def saved(directory: Path, optim: OptimConfig) -> None:
    config = RunConfig(optim=optim)
    directory.mkdir(parents=True)
    (directory / METRICS).write_text(json.dumps({"config": asdict(config)}, default=str))


def test_beside() -> None:
    """A word hears its previous word's next-head and its next word's previous-head, and nothing from a missing one."""
    present = torch.tensor([[True, True, True, False]])
    out = member().beside(torch.zeros(1, 4, CHANNELS), present)
    after = torch.log_softmax(constant(1).bias, dim=-1)
    before = torch.log_softmax(constant(2).bias, dim=-1)
    assert torch.allclose(out[0, 0], before)
    assert torch.allclose(out[0, 1], after + before)
    assert torch.allclose(out[0, 2], after)
    assert torch.allclose(out[0, 3], after)


def test_load(tmp_path: Path) -> None:
    net = real_net()
    directory = tmp_path / "run"
    saved(directory, OptimConfig(auxiliaries=AuxiliaryConfig(semantic=0.0, next_word=0.3, prev_word=0.2)))
    width = net.words.width
    heads = nn.ModuleList([NextWordTerm(weight=0.3).head(width), PreviousWordTerm(weight=0.2).head(width)])
    torch.save({f"heads.{k}": v for k, v in heads.state_dict().items()}, directory / HEADS)
    loaded = Member.load(net, directory, CPU)
    for mine, theirs in ((loaded.after, heads[0]), (loaded.before, heads[1])):
        assert all(torch.equal(a, b) for a, b in zip(mine.parameters(), theirs.parameters(), strict=True))
    assert loaded.net is net
    bare = tmp_path / "bare"
    saved(bare, OptimConfig(auxiliaries=AuxiliaryConfig(next_word=0.3)))
    with pytest.raises(ValueError, match="no neighbour heads"):
        Member.load(net, bare, CPU)


def test_of(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="wraps the ensemble"):
        Neighbours.of(nn.Identity(), [tmp_path], CPU)
    saved(tmp_path / "run", OptimConfig(auxiliaries=AuxiliaryConfig(next_word=0.3, prev_word=0.3)))
    net = real_net()
    heads = nn.ModuleList(
        [NextWordTerm(weight=0.3).head(net.words.width), PreviousWordTerm(weight=0.3).head(net.words.width)]
    )
    torch.save({f"heads.{k}": v for k, v in heads.state_dict().items()}, tmp_path / "run" / HEADS)
    ensemble = Ensemble([net])
    assert len(Neighbours.of(ensemble, [tmp_path / "run"], CPU).members) == 1


def test_scores() -> None:
    """Every word of a three-word span and a lone word is read in word order: the lone word hears no neighbour."""
    onset = np.array([0.0, 0.2, 0.4, 20.0])
    split = Split.from_arrays(np.zeros((4, CHANNELS, 5), dtype=np.float32), np.arange(4), Spans.by_gap(onset), CPU)
    out = Neighbours((member(), member())).scores(split)
    after = torch.log_softmax(constant(1).bias, dim=-1)
    before = torch.log_softmax(constant(2).bias, dim=-1)
    assert out.shape == (4, VOCAB_SIZE)
    assert torch.allclose(out[0], before)
    assert torch.allclose(out[1], after + before)
    assert torch.allclose(out[2], after)
    assert (out[3] == 0).all()
