import pandas as pd
import pytest
import torch

from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.evaluation.studies.templates import (
    ABSENT,
    Durations,
    Noise,
    Readout,
    Reliability,
    Rsa,
    Scale,
    Sums,
    Templates,
    Whitener,
)

CPU = torch.device("cpu")
SHAPE = (4, 6)
REPS = 3
WORDS = 5


def windows(count: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """`count` windows per word of the first `WORDS` words: a word's own pattern plus unit noise."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(WORDS).repeat(count)
    pattern = torch.randn(VOCAB_SIZE, *SHAPE, generator=generator) * 3
    return pattern[labels] + torch.randn(len(labels), *SHAPE, generator=generator), labels


def filled(count: int = 20) -> Sums:
    sums = Sums.empty(2, REPS, SHAPE, CPU)
    x, y = windows(count)
    sums.add(0, x[: len(y) // 2], y[: len(y) // 2])
    sums.add(1, x[len(y) // 2 :], y[len(y) // 2 :])
    return sums


def test_apply() -> None:
    x = torch.randn(5, *SHAPE) * 2 + 3
    scaled = Scale(torch.full((SHAPE[0],), 3.0), torch.full((SHAPE[0],), 2.0)).apply(x)
    assert torch.allclose(scaled, (x - 3) / 2)


def test_means() -> None:
    scale = Scale(torch.zeros(SHAPE[0]), torch.full((SHAPE[0],), 2.0))
    means = scale.means(torch.ones(1, 2, *SHAPE) * 4, torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(means[0, 0], torch.ones(SHAPE))
    assert means[0, 1].abs().sum() == 0


def test_pooled() -> None:
    means = torch.stack([torch.ones(VOCAB_SIZE, *SHAPE), torch.full((VOCAB_SIZE, *SHAPE), 4.0)])
    counts = torch.stack([torch.full((VOCAB_SIZE,), 3), torch.full((VOCAB_SIZE,), 1)])
    pooled = Templates(means, counts).pooled()
    assert torch.allclose(pooled.means[0], torch.full((VOCAB_SIZE, *SHAPE), 1.75))
    assert pooled.counts[0, 0] == 4


def test_distance() -> None:
    means = torch.randn(1, VOCAB_SIZE, *SHAPE)
    distance = Templates(means, torch.ones(1, VOCAB_SIZE, dtype=torch.long)).distance()
    assert torch.allclose(distance.diagonal(), torch.zeros(VOCAB_SIZE), atol=1e-5)
    assert torch.allclose(distance, distance.T, atol=1e-5)


def test_empty() -> None:
    sums = Sums.empty(2, REPS, SHAPE, CPU)
    assert sums.halves.shape == (REPS, 2, 2 * VOCAB_SIZE, *SHAPE)


def test_add() -> None:
    sums = filled()
    assert sums.counts.sum() == 20 * WORDS
    assert sums.half_counts.sum(dim=(2,)).eq(sums.counts.sum(dim=1)).all()
    assert torch.allclose(sums.halves[0].sum(dim=1), sums.sums.sum(dim=1))


def test_scale() -> None:
    scale = filled().scale()
    assert scale.std.min() > 0
    assert scale.mean.shape == (SHAPE[0],)


def test_templates() -> None:
    sums = filled()
    templates = sums.templates(sums.scale())
    assert templates.counts[:, :WORDS].min() == 10
    assert templates.counts[:, WORDS:].sum() == 0


def test_split() -> None:
    sums = filled()
    first, second, valid = sums.split(sums.scale(), None)
    assert first.shape == (REPS, VOCAB_SIZE, *SHAPE)
    assert valid[:, :WORDS].all()
    assert not valid[:, WORDS:].any()
    assert sums.split(sums.scale(), 0)[2].shape == (REPS, VOCAB_SIZE)
    assert second.shape == first.shape


def test_of() -> None:
    sums = filled(40)
    reliability = Reliability.of(*sums.split(sums.scale(), None))
    assert reliability.same[:, :WORDS].min() > reliability.diff[:, :WORDS].max()
    assert reliability.same[:, WORDS:].isnan().all()


def test_spearman_brown() -> None:
    reliability = Reliability(torch.full((2, VOCAB_SIZE), 0.5), torch.zeros(2, VOCAB_SIZE))
    assert torch.allclose(reliability.spearman_brown(), torch.full((VOCAB_SIZE,), 2 / 3))


def test_summary() -> None:
    sums = filled(40)
    reliability = Reliability.of(*sums.split(sums.scale(), None))
    line = reliability.summary(sums.counts.sum(dim=0).long())
    assert "words=5" in line
    assert "trials/word min=40" in line


def test_apply_whitener() -> None:
    x = torch.randn(3, *SHAPE)
    white = Whitener(torch.eye(SHAPE[0]) * 2, torch.eye(SHAPE[1]))
    assert torch.allclose(white.apply(x), x * 2)


def test_spatial_only() -> None:
    white = Whitener(torch.eye(SHAPE[0]), torch.ones(SHAPE[1], SHAPE[1])).spatial_only()
    assert torch.equal(white.temporal, torch.eye(SHAPE[1]))


def test_add_noise() -> None:
    noise = Noise.empty(SHAPE, CPU)
    noise.add(torch.randn(8, *SHAPE))
    assert noise.seen[0] == 8 * SHAPE[1]
    assert noise.seen[1] == 8 * SHAPE[0]
    assert noise.spatial.trace() > 0


def test_empty_noise() -> None:
    noise = Noise.empty(SHAPE, CPU)
    assert noise.spatial.shape == (SHAPE[0], SHAPE[0])
    assert noise.temporal.shape == (SHAPE[1], SHAPE[1])


def test_shrunk() -> None:
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(40, 8, generator=generator, dtype=torch.float64)
    total, quartic = rows.T @ rows, rows.pow(2).sum(dim=1).pow(2).sum()
    cov = Noise.shrunk(total, quartic, torch.tensor(40.0, dtype=torch.float64))
    assert torch.allclose(cov, cov.T)
    assert cov.trace() == pytest.approx(float((total / 40).trace()), rel=1e-6)
    assert torch.linalg.eigvalsh(cov).min() > 0


def test_inverse_root() -> None:
    cov = torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64))
    root = Noise.inverse_root(cov)
    assert torch.allclose(root @ cov.float() @ root, torch.eye(2), atol=1e-5)


def test_whitener() -> None:
    noise = Noise.empty(SHAPE, CPU)
    noise.add(torch.randn(64, *SHAPE))
    white = noise.whitener()
    assert white.spatial.shape == (SHAPE[0], SHAPE[0])
    assert white.temporal.shape == (SHAPE[1], SHAPE[1])


def test_standardised() -> None:
    flat = Readout.standardised(torch.randn(3, 10) * 4 + 2)
    assert torch.allclose(flat.norm(dim=-1), torch.ones(3))
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(3), atol=1e-6)


def decoder(count: int = 30) -> tuple[Readout, Readout, torch.Tensor, torch.Tensor]:
    x, y = windows(count)
    sums = Sums.empty(1, REPS, SHAPE, CPU)
    sums.add(0, x, y)
    scale = sums.scale()
    templates = sums.templates(scale)
    noise = Noise.empty(SHAPE, CPU)
    noise.add(scale.apply(x) - templates.means[0][y])
    return Readout.correlation(templates), Readout.discriminant(templates, noise.whitener()), scale.apply(x), y


def test_correlation() -> None:
    read, _, x, y = decoder()
    assert (read.scores(x, 0).argmax(dim=1) == y).float().mean() > 0.95


def test_discriminant() -> None:
    _, read, x, y = decoder()
    assert (read.scores(x, 0).argmax(dim=1) == y).float().mean() > 0.95
    assert read.bias[0, 0] > 0


def test_of_readout() -> None:
    templates = Templates(torch.randn(1, VOCAB_SIZE, *SHAPE), torch.ones(1, VOCAB_SIZE, dtype=torch.long))
    read = Readout.of(templates, lambda x: x.flatten(1), discriminant=False)
    assert read.means.shape == (1, VOCAB_SIZE, SHAPE[0] * SHAPE[1])
    assert read.bias.sum() == 0


def test_scores() -> None:
    read, _, x, _ = decoder()
    scores = read.scores(x, 0)
    assert scores.shape == (len(x), VOCAB_SIZE)
    assert (scores[:, WORDS:] == ABSENT).all()


def test_add_durations() -> None:
    durations = Durations()
    durations.add(torch.tensor([0, 0, 1]), torch.tensor([1.0, 3.0, float("nan")]))
    assert durations.counts[:2].tolist() == [2.0, 0.0]


def test_mean() -> None:
    durations = Durations()
    durations.add(torch.tensor([0, 0]), torch.tensor([1.0, 3.0]))
    assert durations.mean()[0] == 2.0


def test_gap() -> None:
    assert Rsa.gap(torch.tensor([0.0, 1.0, 4.0]))[0].tolist() == [0.0, 1.0, 4.0]


def test_phonetic() -> None:
    table = Pronunciation(
        torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))).sort().values,
        torch.zeros(VOCAB_SIZE, 4, dtype=torch.long),
    )
    assert Rsa.phonetic(table).sum() == 0


def test_semantic() -> None:
    distance = Rsa.semantic(torch.eye(3))
    assert distance.diagonal().abs().max() < 1e-6
    assert distance[0, 1] == pytest.approx(1.0)


def test_ranks() -> None:
    assert Rsa.ranks(torch.tensor([3.0, 1.0, 1.0, 2.0])).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_correlate() -> None:
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(12, 3, generator=generator)
    target = torch.cdist(points, points)
    words = torch.ones(12, dtype=torch.bool)
    rsa = Rsa(permutations=200)
    r, p, pairs = rsa.correlate(target + 0.01 * torch.rand(12, 12), target, words)
    assert r > 0.9
    assert p < 0.05
    assert pairs == 66
    assert abs(rsa.correlate(torch.rand(12, 12), target, words)[0]) < 0.5
