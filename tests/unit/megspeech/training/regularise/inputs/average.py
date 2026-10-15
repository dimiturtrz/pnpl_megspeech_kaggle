import dataclasses

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Batch, Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.training.regularise.inputs.augment import Augment
from megspeech.training.regularise.inputs.average import GroupAverage
from megspeech.training.regularise.inputs.bgmix import Untouched

CPU = torch.device("cpu")
CHANNELS = 2
SAMPLES = 3


def pair(labels: list[int], subjects: list[int] | None = None) -> tuple[Split, Batch, torch.Tensor]:
    n = len(labels)
    torch.manual_seed(0)
    x = torch.randn(n, CHANNELS, SAMPLES)
    split = Split.from_arrays(x.numpy(), np.array(labels), Spans.by_gap(np.arange(n) * 10.0), CPU)
    if subjects is not None:
        split = dataclasses.replace(split, subject=torch.tensor(subjects))
    x = split.windows(torch.arange(n))
    index = torch.arange(n)[None]
    return split, Batch(x, torch.zeros(1, n), index, split.subject, index=index), x


def averager(split: Split, k: int) -> GroupAverage:
    made = GroupAverage.of(split, k, 1.0)
    assert isinstance(made, GroupAverage)
    return made


def test_of() -> None:
    split, _, _ = pair([1, 1])
    assert not isinstance(GroupAverage.of(split, 1, 1.0), GroupAverage)
    assert not isinstance(GroupAverage.of(split, 3, 0.0), GroupAverage)
    assert averager(split, 3).size.tolist() == [2, 2]


def test_apply() -> None:
    split, batch, x = pair([4, 4, 9])
    out = averager(split, 2).apply(x, batch, split)
    assert torch.allclose(out[0], x[:2].mean(0), atol=1e-5)
    assert torch.allclose(out[1], x[:2].mean(0), atol=1e-5)


def test_apply_single_occurrence_unchanged() -> None:
    split, batch, x = pair([4, 4, 9])
    assert torch.equal(averager(split, 5).apply(x, batch, split)[2], x[2])


def test_apply_uses_every_partner_when_group_is_small() -> None:
    split, batch, x = pair([4, 4, 4, 9])
    out = averager(split, 8).apply(x, batch, split)
    assert torch.allclose(out[:3], x[:3].mean(0).expand(3, -1, -1), atol=1e-5)


def test_apply_draws_k_windows_from_a_larger_group() -> None:
    split, batch, x = pair([4] * 6)
    out = averager(split, 3).apply(x, batch, split)
    pairs = torch.combinations(torch.arange(1, 6), 2).tolist()
    candidates = torch.stack([(x[0] + x[a] + x[b]) / 3 for a, b in pairs])
    assert torch.isclose(candidates, out[0], atol=1e-5).flatten(1).all(1).any()


def test_apply_never_mixes_subjects() -> None:
    split, batch, x = pair([4, 4, 4, 4], [0, 0, 1, 1])
    out = averager(split, 4).apply(x, batch, split)
    assert torch.allclose(out[:2], x[:2].mean(0).expand(2, -1, -1), atol=1e-5)
    assert torch.allclose(out[2:], x[2:].mean(0).expand(2, -1, -1), atol=1e-5)


def test_apply_chance_zero_is_identity() -> None:
    split, batch, x = pair([4, 4, 4])
    made = averager(split, 3)
    made.p = 0.0
    assert torch.equal(made.apply(x, batch, split), x)
    assert made.drained() == 0


def test_apply_leaves_the_split_untouched() -> None:
    split, batch, x = pair([4, 4, 4])
    before = split.windows(torch.arange(3)).clone()
    averager(split, 3).apply(x, batch, split)
    assert torch.equal(split.windows(torch.arange(3)), before)


def test_apply_ignores_padding_and_unlabelled_rows() -> None:
    split, batch, x = pair([4, 4, IGNORE])
    padded = torch.cat([x, torch.zeros(1, CHANNELS, SAMPLES)])
    batch = Batch(padded, batch.onset, torch.tensor([[0, 1, 2]]), batch.subject, index=batch.index)
    out = averager(split, 3).apply(padded, batch, split)
    assert torch.equal(out[2:], padded[2:])


def test_drained() -> None:
    split, batch, x = pair([4, 4])
    made = averager(split, 2)
    made.apply(x, batch, split)
    assert made.drained() == 2
    assert made.drained() == 0


def test_untouched_is_identity() -> None:
    split, batch, x = pair([4, 4])
    assert torch.equal(Untouched().apply(x, batch, split), x)


def test_augment_rejects_bad_average() -> None:
    assert Augment().average == 1
    with pytest.raises(ValueError, match="average"):
        Augment(average=0)
    with pytest.raises(ValueError, match="average"):
        Augment(average_chance=1.5)
