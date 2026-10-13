from pathlib import Path
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import SpanInput, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies.knnreadout import (
    FLOOR,
    KS,
    LAMBDAS,
    TEMPERATURES,
    Builder,
    Dump,
    Embedder,
    Mixing,
    Neighbours,
    Readout,
    Rows,
    Setting,
    Space,
    Store,
    Study,
)
from megspeech.model.sentence import SentenceNet

CPU = torch.device("cpu")
SHARP = 0.01
WIDTH = 3


class Slots(nn.Module):
    """Stands in for a `SentenceNet`: each position's refined vector is its batch row, repeated."""

    width = WIDTH

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x

    def heard(self, spans: SpanInput) -> tuple[Tensor, None, None, None]:
        return spans.slot.float()[..., None].expand(*spans.slot.shape, WIDTH), None, None, None


def split() -> Split:
    x = np.zeros((4, CHANNELS, 5), dtype=np.float32)
    return Split.from_arrays(x, np.arange(4), Spans.by_gap(np.array([0.0, 0.2, 0.4, 20.0])), CPU)


def axes(*labels: int) -> Tensor:
    """One unit vector per label, along its own axis."""
    return torch.eye(VOCAB_SIZE)[list(labels)]


def store() -> Store:
    """Two keys of label 0 along axis 0, one of label 1 along axis 1."""
    return Store(axes(0, 0, 1), torch.tensor([0, 0, 1]))


def dump() -> Dump:
    """Two folds; one domain whose two words lie on their own label's axis, the base indifferent between labels."""
    rows = Rows(torch.full((2, VOCAB_SIZE), -np.log(VOCAB_SIZE)), torch.tensor([0, 1]), (axes(0, 1), axes(0, 1)))
    stores = tuple(Store(axes(*range(VOCAB_SIZE)), torch.arange(VOCAB_SIZE)) for _ in range(2))
    partitions = {p: {"Sherlock1": rows} for p in (Partition.VALIDATION, Partition.TEST)}
    return Dump(stores, partitions, torch.full((VOCAB_SIZE,), -np.log(VOCAB_SIZE)), 0.0)


def setting(readout: Readout = Readout.KNN, lam: float = 1.0) -> Setting:
    return Setting(readout, Space.RAW, KS[0], TEMPERATURES[0], Mixing.LOGLINEAR, lam, 0.0)


def test_embed() -> None:
    """Every word of a three-word span and a lone word is read, in word order."""
    out = Embedder(cast("SentenceNet", Slots())).embed(split())
    assert torch.allclose(out, torch.arange(4.0)[:, None].expand(4, WIDTH))


def test_swept() -> None:
    assert Embedder(cast("SentenceNet", Slots())).swept(split()).shape == (4, WIDTH)


def test_unit() -> None:
    x = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
    mean = x.mean(dim=0)
    assert torch.allclose(Space.RAW.unit(x, mean).norm(dim=-1), torch.ones(2))
    assert torch.allclose(Space.CENTRED.unit(x, mean), torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))


def test_vote() -> None:
    """The nearest neighbour dominates a sharp vote; a label nobody votes for keeps only the floor."""
    near = Neighbours(torch.tensor([[1.0, 0.0]]), torch.tensor([[3, 4]]))
    sharp = near.vote(2, SHARP)
    assert sharp[0, 3] > 1 - 1e-3
    assert sharp[0, 0] == FLOOR
    assert torch.allclose(near.vote(1, SHARP)[0, 3], torch.tensor(1.0))


def test_neighbours() -> None:
    found = store().neighbours(axes(1), 2, Space.RAW)
    assert found.labels.tolist() == [[1, 0]]
    assert torch.allclose(found.sims, torch.tensor([[1.0, 0.0]]))


def test_prototypes() -> None:
    cosines = store().prototypes(axes(0), Space.RAW)
    assert cosines.shape == (1, VOCAB_SIZE)
    assert cosines[0].argmax() == 0
    assert cosines[0, 2] == 0


def test_mix() -> None:
    base = torch.log(torch.tensor([[0.5, 0.5]]))
    readout = torch.tensor([[1.0, 0.0]]).clamp(min=FLOOR)
    assert torch.allclose(Mixing.LINEAR.mix(base, readout, 0.5).exp(), torch.tensor([[0.75, 0.25]]), atol=1e-5)
    assert torch.allclose(Mixing.LOGLINEAR.mix(base, readout, 0.0), base)
    assert torch.allclose(Mixing.LOGLINEAR.mix(base, readout, 1.0), readout.log())


def test_path(tmp_path: Path) -> None:
    assert Dump.path(DataRoot(tmp_path), "m", "kaggle").name == "m-kaggle.pt"


def test_disjoint() -> None:
    Builder.disjoint([["a"], ["b"]], {"c"})
    with pytest.raises(ValueError, match="overlap"):
        Builder.disjoint([["a"], ["a"]], set())
    with pytest.raises(ValueError, match="overlap"):
        Builder.disjoint([["a"]], {"a"})


def test_reads() -> None:
    neighbours, cosines = Study(dump()).reads(Partition.VALIDATION, Space.RAW)["Sherlock1"]
    assert len(neighbours) == len(cosines) == 2
    assert neighbours[0].labels.shape == (2, min(max(KS), VOCAB_SIZE))


def test_readout() -> None:
    reads = Study(dump()).reads(Partition.VALIDATION, Space.RAW)["Sherlock1"]
    for readout in Readout:
        p = Study.readout(reads, readout, KS[0], SHARP)
        assert p.argmax(dim=-1).tolist() == [0, 1]


def test_graded() -> None:
    study = Study(dump())
    graded = study.graded(Partition.VALIDATION, {"Sherlock1": axes(0, 1)}, 0.0)
    hits, labels = graded.domains["Sherlock1"]
    assert hits.all()
    assert labels.tolist() == [0, 1]


def test_base() -> None:
    assert set(Study(dump()).base(Partition.TEST).domains) == {"Sherlock1"}


def test_settings() -> None:
    graded = Study(dump()).settings(Partition.VALIDATION)
    assert setting() in graded
    assert setting(Readout.PROTOTYPE, LAMBDAS[0]) in graded


def test_select() -> None:
    study = Study(dump())
    graded = study.settings(Partition.VALIDATION)
    assert study.select(graded, Readout.KNN, alone=True).lam == 1.0
    assert study.select(graded, Readout.PROTOTYPE, alone=False).lam < 1.0


def test_at() -> None:
    study = Study(dump())
    graded = study.at(Partition.TEST, setting())
    assert study.ruler.point(graded) == study.ruler.point(study.settings(Partition.TEST)[setting()])
