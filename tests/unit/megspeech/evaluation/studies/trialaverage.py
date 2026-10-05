from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.trialaverage import TRIALS, Averaging, Ceiling, Centroids, Encoder
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import SPOKEN, loaded, runs

WIDTH = 4


class Reader(nn.Module):
    """A word head reading the first embedding component back as the class."""

    @override
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(embeddings[:, 0].long(), VOCAB_SIZE).float() * 10


class Words(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = Reader()


class Embedding(nn.Module):
    """A word encoder embedding each window as its row index repeated."""

    def __init__(self) -> None:
        super().__init__()
        self.words = Words()

    def embed(self, x: torch.Tensor, subject: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.arange(len(x), dtype=torch.float32)[:, None].expand(-1, WIDTH)


class Holder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.members = nn.ModuleList([Embedding()])


def encoder(tmp_path: Path) -> Encoder:
    return Encoder(replace(loaded(tmp_path), model=Holder()))


def clustered() -> Averaging:
    """Ten trials of each of classes 0-2, noisy around the class's own axis."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(3).repeat_interleave(10)
    centre = torch.eye(WIDTH)[labels] * 3
    return Averaging(centre + torch.randn(30, WIDTH, generator=generator), labels, draws=5)


def test_empty() -> None:
    assert Centroids.empty(WIDTH).counts.sum() == 0


def test_add() -> None:
    found = Centroids.empty(WIDTH).add(torch.ones(2, WIDTH), torch.tensor([3, 3]))
    assert found.counts[3] == 2
    assert found.sums[3].tolist() == [2.0] * WIDTH


def test_scores() -> None:
    """An embedding is nearest its own class's mean; a class without training trials is never ranked."""
    found = Centroids.empty(WIDTH).add(torch.eye(WIDTH)[:2], torch.tensor([0, 1]))
    scores = found.scores(torch.eye(WIDTH)[:2])
    assert scores.argmax(dim=1).tolist() == [0, 1]
    assert scores[:, 2].isinf().all()


def test_counts() -> None:
    assert clustered().counts[:3].tolist() == [10, 10, 10]


def test_draw() -> None:
    """Each eligible class gives one average; a class with fewer than `k` trials gives none."""
    averaged, classes = clustered().draw(4, torch.Generator().manual_seed(0))
    assert classes.tolist() == [0, 1, 2]
    assert averaged.shape == (3, WIDTH)
    assert clustered().draw(11, torch.Generator().manual_seed(0))[1].numel() == 0


def test_rates() -> None:
    """A readout ranking the low classes on top hits every eligible class always; the others are NaN."""
    rates = clustered().rates(2, lambda e: -torch.arange(VOCAB_SIZE).float().expand(len(e), -1))
    assert rates[:3].tolist() == [1.0, 1.0, 1.0]
    assert rates[3:].isnan().all()


def test_curve() -> None:
    """Classes with fewer trials than `k` are NaN from that `k` on."""
    curve = clustered().curve(lambda e: torch.zeros(len(e), VOCAB_SIZE))
    assert curve.shape == (len(TRIALS), VOCAB_SIZE)
    assert curve[:, :3].isnan().tolist() == [[False] * 3] * 4 + [[True] * 3] * 2


def test_balanced() -> None:
    curve = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    assert Averaging.balanced(curve).tolist() == [1.0, 0.5]
    assert Averaging.balanced(curve, torch.tensor([False, True])).isnan()[0]


def test_network(tmp_path: Path) -> None:
    assert isinstance(encoder(tmp_path).network, Embedding)


def test_embed(tmp_path: Path) -> None:
    enc = encoder(tmp_path)
    domains, _ = enc.loaded.domains({RULER: runs(tmp_path)[:1]})
    assert enc.embed(domains[RULER]).shape == (len(SPOKEN), WIDTH)


def test_head(tmp_path: Path) -> None:
    out = encoder(tmp_path).head(torch.full((1, WIDTH), 7.0))
    assert out.argmax().item() == 7
    assert out.shape == (1, VOCAB_SIZE)


def test_centroids(tmp_path: Path) -> None:
    found = encoder(tmp_path).centroids(runs(tmp_path)[:1])
    assert found.counts.sum() == len(SPOKEN)


def test_domain(tmp_path: Path) -> None:
    enc = encoder(tmp_path)
    domains, _ = enc.loaded.domains({RULER: runs(tmp_path)[:1]})
    averaging, curves = Ceiling(enc, Centroids.empty(WIDTH), draws=2).domain(domains[RULER])
    assert set(curves) == {"centroid", "head", "head-tau0"}
    assert len(averaging.labels) == len(SPOKEN)


def test_lines() -> None:
    curves = {"head": torch.full((len(TRIALS), VOCAB_SIZE), float("nan"))}
    curves["head"][:, :3] = 0.5
    lines = Ceiling.lines("d", clustered(), curves)
    assert lines[0] == "d n=30 classes=3 fixed_classes=0"
    assert "k1=0.5000(c=3)" in lines[1]


def test_report(tmp_path: Path) -> None:
    enc = encoder(tmp_path)
    domains, _ = enc.loaded.domains({RULER: runs(tmp_path)[:1]})
    lines = Ceiling(enc, Centroids.empty(WIDTH), draws=2).report(domains)
    assert any(line.startswith(f"{RULER} head eligible") for line in lines)
