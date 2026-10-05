import dataclasses

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Batch, Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.training.regularise.bgmix import MIN_CELL, BGMix, BGMixKey, Untouched

CPU = torch.device("cpu")
CHANNELS = 2
SAMPLES = 3


def table(subjects: int = 1) -> torch.Tensor:
    cells = torch.arange(subjects * VOCAB_SIZE, dtype=torch.float32).view(subjects, VOCAB_SIZE, 1, 1)
    return cells.expand(subjects, VOCAB_SIZE, CHANNELS, SAMPLES).clone()


def pair(labels: list[int]) -> tuple[Split, Batch, torch.Tensor]:
    n = len(labels)
    x = torch.randn(n, CHANNELS, SAMPLES)
    split = Split.from_arrays(x.numpy(), np.array(labels), Spans.by_gap(np.arange(n) * 10.0), CPU)
    index = torch.arange(n)[None]
    batch = Batch(x, torch.zeros(1, n), index, index, torch.zeros(n, dtype=torch.long))
    return split, batch, x


def test_of() -> None:
    labels = [3] * MIN_CELL + [4] * 2
    split, _, _ = pair(labels)
    x = split.windows(torch.arange(len(labels)))
    mixer = BGMix.of(split, 1.0)
    assert isinstance(mixer, BGMix)
    assert torch.allclose(mixer.table[0, 3], x[:MIN_CELL].mean(0), atol=1e-5)
    assert torch.allclose(mixer.table[0, 4], x[MIN_CELL:].mean(0), atol=1e-5)
    assert isinstance(BGMix.of(split, 0.0), Untouched)


def test_averaged() -> None:
    sums = torch.zeros(2 * VOCAB_SIZE, 1, 1)
    counts = torch.zeros(2 * VOCAB_SIZE)
    sums[3], counts[3] = 40.0, 40.0
    sums[VOCAB_SIZE + 3], counts[VOCAB_SIZE + 3] = 10.0, 2.0
    averaged = BGMix.averaged(sums, counts, 2)
    assert averaged[0, 3].item() == 1.0
    assert abs(averaged[1, 3].item() - 50.0 / 42.0) < 1e-5


def test_rows() -> None:
    split, batch, _ = pair([1, IGNORE, 2])
    assert BGMix.rows(batch, split).tolist() == [1, IGNORE, 2]


def test_apply_p0_is_identity_and_untouched_is_too() -> None:
    split, batch, x = pair([1, 2, 3, 4])
    assert torch.equal(BGMix(0.0, table()).apply(x, batch, split), x)
    assert torch.equal(Untouched().apply(x, batch, split), x)


def test_apply_p1_is_template_plus_the_partners_residual() -> None:
    torch.manual_seed(0)
    split, batch, x = pair([1, 2, 3, 4])
    mixer = BGMix(1.0, table())
    out = mixer.apply(x, batch, split)
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for i in range(4):
        residual = out[i] - labels[i]
        partner = [j for j in range(4) if torch.allclose(x[j] - labels[j], residual, atol=1e-5)]
        assert partner
        assert partner != [i]
    assert mixer.drained() == 4


def test_apply_leaves_unlabelled_rows_and_never_partners_with_them() -> None:
    torch.manual_seed(0)
    split, batch, x = pair([1, IGNORE, 2, IGNORE])
    out = BGMix(1.0, table()).apply(x, batch, split)
    assert torch.equal(out[1], x[1])
    assert torch.equal(out[3], x[3])
    assert torch.allclose(out[0] - 1.0, x[2] - 2.0, atol=1e-5)
    assert torch.allclose(out[2] - 2.0, x[0] - 1.0, atol=1e-5)


def test_drained() -> None:
    split, batch, x = pair([1, 2, 3, 4])
    mixer = BGMix(1.0, table())
    mixer.apply(x, batch, split)
    assert mixer.drained() == 4
    assert mixer.drained() == 0
    assert Untouched().drained() == 0


def sourced(labels: list[int], sources: list[int]) -> tuple[Split, Batch, torch.Tensor]:
    split, batch, x = pair(labels)
    return dataclasses.replace(split, source=torch.tensor(sources)), batch, x


def test_key_of() -> None:
    split, _, _ = sourced([1, 2], [3, 4])
    assert BGMixKey.SUBJECT.of(split).tolist() == [0, 0]
    assert BGMixKey.SOURCE.of(split).tolist() == [3, 4]
    with pytest.raises(ValueError, match="source"):
        BGMixKey.SOURCE.of(pair([1])[0])


def test_of_source_keyed_templates_are_per_source() -> None:
    labels = [3] * (2 * MIN_CELL)
    sources = [0] * MIN_CELL + [1] * MIN_CELL
    split, _, _ = sourced(labels, sources)
    x = split.windows(torch.arange(len(labels)))
    mixer = BGMix.of(split, 1.0, BGMixKey.SOURCE)
    assert isinstance(mixer, BGMix)
    assert mixer.table.shape[0] == 2
    assert torch.allclose(mixer.table[0, 3], x[:MIN_CELL].mean(0), atol=1e-5)
    assert torch.allclose(mixer.table[1, 3], x[MIN_CELL:].mean(0), atol=1e-5)


def test_of_source_keyed_thin_cell_falls_back_to_pooled_label() -> None:
    labels = [3] * MIN_CELL + [3] * 2
    split, _, _ = sourced(labels, [0] * MIN_CELL + [1] * 2)
    x = split.windows(torch.arange(len(labels)))
    mixer = BGMix.of(split, 1.0, BGMixKey.SOURCE)
    assert isinstance(mixer, BGMix)
    assert torch.allclose(mixer.table[1, 3], x.mean(0), atol=1e-5)


def test_apply_source_keyed_uses_each_rows_own_source_template() -> None:
    torch.manual_seed(0)
    split, batch, x = sourced([1, 2], [0, 1])
    tab = table(2)
    tab[1] += 100.0
    out = BGMix(1.0, tab, BGMixKey.SOURCE).apply(x, batch, split)
    assert torch.allclose(out[0], tab[0, 1] + x[1] - tab[1, 2], atol=1e-4)
    assert torch.allclose(out[1], tab[1, 2] + x[0] - tab[0, 1], atol=1e-4)
