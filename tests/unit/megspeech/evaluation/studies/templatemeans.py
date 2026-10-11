import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.templatemeans import Reliability, Scale, Sums, Templates

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


def test_scale_apply() -> None:
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


def test_sums_empty() -> None:
    sums = Sums.empty(2, REPS, SHAPE, CPU)
    assert sums.halves.shape == (REPS, 2, 2 * VOCAB_SIZE, *SHAPE)


def test_sums_add() -> None:
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


def test_reliability_of() -> None:
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


def test_standardised() -> None:
    flat = Templates.standardised(torch.randn(3, 10) * 4 + 2)
    assert torch.allclose(flat.norm(dim=-1), torch.ones(3))
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(3), atol=1e-6)
