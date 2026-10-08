from dataclasses import replace
from pathlib import Path
from typing import cast

import torch

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.objective.anchored import AnchoredHead
from megspeech.objective.anchors import Source
from megspeech.objective.scorer import DIM
from tests.unit.megspeech.model.sentence import CONTEXT, GENERATOR, SAMPLES

torch.manual_seed(0)


def test_load(tmp_path: Path) -> None:
    model = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), CONTEXT)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    CONTEXT.save(tmp_path)
    loaded = Checkpoint.load(tmp_path / "model.pt")
    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in model.state_dict().items())


def test_load_anchored(tmp_path: Path) -> None:
    anchors = torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR)
    anchored = replace(CONTEXT, anchor=Source.AUDIO)
    model = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), anchored, anchors)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    anchored.save(tmp_path)
    loaded = Checkpoint.load(tmp_path / "model.pt")
    assert torch.equal(cast("AnchoredHead", loaded.head).anchors, anchors)
    reduced = replace(anchored, anchor_dim=8)
    small = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), reduced, anchors[:, :8])
    torch.save(small.state_dict(), tmp_path / "small.pt")
    reduced.save(tmp_path)
    assert cast("AnchoredHead", Checkpoint.load(tmp_path / "small.pt").head).anchors.shape == (VOCAB_SIZE, 8)


def test_subspace_channels(tmp_path: Path) -> None:
    """A model reading a `Projection`'s components: its input width is its scaler's, saved and loaded back so."""
    components = 5
    context = replace(CONTEXT, subjects=2, whiten=True)
    sample = torch.randn(4, components, SAMPLES, generator=GENERATOR)
    model = SentenceNet(Scaler(sample), context, channels=components).eval()
    model.whiten.fit(model.scale(sample))
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, components, SAMPLES, generator=GENERATOR)
    onsets = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    context_logits, _ = model.forward(x, onsets, slot, torch.zeros(len(x), dtype=torch.long))
    assert context_logits.shape == (2, 3, VOCAB_SIZE)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    context.save(tmp_path)
    loaded = Checkpoint.load(tmp_path / "model.pt").eval()
    assert loaded.state_dict()["whiten.matrix"].shape == (components, components)
    again, _ = loaded.forward(x, onsets, slot, torch.zeros(len(x), dtype=torch.long))
    assert torch.allclose(again, context_logits)
