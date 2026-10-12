import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import DENSE, FEATURES, TRAIN, Batch, Features, SpanInput, SpanText, Split
from megspeech.data.batching.windows import Windows
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD, Spans

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


class Kaggle:
    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": [0, 1, 2],
                "subject": 0,
                "source": ["sentence", "sentence", "word"],
                "epoch": [0, 0, 0],
                "onset_s": [0.0, 0.3, 0.0],
            }
        )


def test_from_arrays() -> None:
    assert len(split().windows) == 12
    assert split().windows(torch.tensor([0])).shape == (1, CHANNELS, SAMPLES)


def test_of() -> None:
    spans = Spans.by_gap(np.array([0.0, 0.1, 9.0]))
    at = torch.tensor([0, 10, 20])
    windows = Windows(torch.zeros(CHANNELS, 100), at, at + 50)
    held = Split.of(windows, (np.array([0, 1, 2]),) * 2, spans, np.array([1.0, 1.0, 3.0]))
    assert held.y.tolist() == [0, 1, 2]
    assert held.index.shape[0] == 2
    assert held.weight.tolist() == [1.0, 1.0, 3.0]
    assert held.finish is None


def timed() -> Split:
    """Two spans read at 10 Hz: words at 0 and 0.1 s, the second's epoch running 2 s on; a lone word 6 s long."""
    spans = Spans.by_gap(np.array([0.0, 0.1, 9.0]))
    at = torch.tensor([0, 10, 20])
    windows = Windows(torch.zeros(CHANNELS, 100), at, at + 50, following=torch.tensor([10, 30, 80]))
    return Split.of(windows, (np.array([0, 1, 2]),) * 2, spans, np.ones(3)).finished(10)


def test_ended() -> None:
    """Each span's last word ends `TAIL` before its bound, counted from the span's first onset."""
    held = timed()
    assert Split.ended(held.windows, held.index, held.onset, 10).tolist() == pytest.approx([1.1, 5.0])
    assert Split.ended(held.windows, held.index, held.onset, 10, tail=0.0).tolist() == pytest.approx([2.1, 6.0])


def test_finished() -> None:
    """Without next onsets every end is 0: unknown."""
    held = timed()
    assert held.finish is not None
    assert held.finish.tolist() == pytest.approx([1.1, 5.0])
    plain = dataclasses.replace(held, windows=dataclasses.replace(held.windows, following=None)).finished()
    assert plain.finish is not None
    assert plain.finish.tolist() == [0.0, 0.0]


def test_ends() -> None:
    held = timed()
    ends = held.ends(torch.tensor([1]))
    assert ends is not None
    assert ends.tolist() == pytest.approx([5.0])
    assert dataclasses.replace(held, finish=None).ends(torch.tensor([0])) is None
    finish = held.batch(torch.tensor([1]), 4).finish
    assert finish is not None
    assert finish.tolist() == pytest.approx([5.0])


def test_sample() -> None:
    assert split().sample(5).shape == (5, CHANNELS, SAMPLES)
    assert split().sample(100).shape == (12, CHANNELS, SAMPLES)


def test_drawn() -> None:
    drawn = split().drawn(5)
    assert len(drawn) == len(set(drawn.tolist())) == 5
    assert split().drawn(100).sort().values.tolist() == list(range(12))


def test_labelled() -> None:
    held = dataclasses.replace(split(), y=torch.tensor([0, IGNORE, 2]))
    assert held.labelled().tolist() == [0, 2]


def test_lone() -> None:
    """The dense words share a span; each word 2 s after its neighbour is a span of its own."""
    assert split().lone().tolist() == [False] * DENSE + [True] * (12 - DENSE)


def test_log_prior() -> None:
    assert torch.allclose(split().log_prior().exp().sum(), torch.tensor(1.0))
    heavy = dataclasses.replace(split(), weight=(torch.arange(12) % 3 == 0).float() * 100)
    assert heavy.log_prior().argmax() == 0


def test_prior() -> None:
    prior = Split.prior(torch.tensor([0, 0, 1, IGNORE, IGNORE, IGNORE]), torch.ones(6))
    assert prior.argmax() == 0
    assert torch.allclose(prior.exp().sum(), torch.tensor(1.0))


def test_split_novel() -> None:
    held = split()
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(12 - DENSE) * 2.0]
    everything = frozenset(SpanText(Spans.by_gap(onset).index, held.y.numpy()).sequences())
    assert len(held.novel(everything)) == 0
    assert held.novel(frozenset()).tolist() == list(range(DENSE))
    unknown = dataclasses.replace(held, y=torch.where(torch.arange(12) == 1, IGNORE, held.y))
    assert unknown.novel(frozenset()).tolist() == [0, 2, 3]
    retold = dataclasses.replace(held, token=held.token + 100)
    assert retold.novel(everything).tolist() == list(range(DENSE))


def test_text() -> None:
    held = split()
    assert held.text.index.tolist() == held.index.tolist()
    assert held.text.token.tolist() == held.token.tolist()


def test_recurring() -> None:
    onset = np.array([0, 0.1, 0.2, 9, 9.1, 9.2, 20, 20.1, 30, 30.1])
    y = np.array([0, 1, 2, 0, 1, 2, 3, 4, 3, 4])
    twice = Split.from_arrays(np.zeros((10, CHANNELS, SAMPLES), np.float32), y, Spans.by_gap(onset), CPU)
    assert twice.recurring(1.0).weight.tolist() == [0.5] * 6 + [1.0] * 4
    assert twice.recurring(0.0).weight.tolist() == [1.0] * 10


def test_batch() -> None:
    batch = split().batch(torch.tensor([1, 0]), rows=6)
    assert batch.slot[:, :DENSE].tolist() == [[0, 6, 6, 6], [1, 2, 3, 4]]
    assert torch.equal(batch.x[:5], split().windows(torch.tensor([DENSE, 0, 1, 2, 3])))
    assert batch.onset.shape == batch.slot.shape
    overflow = split().batch(torch.tensor([0]), rows=2)
    assert overflow.slot[:, :DENSE].tolist() == [[0, 1, 2, 2]]
    assert overflow.index[:, :DENSE].tolist() == [[0, 1, PAD, PAD]]
    jittered = split().batch(torch.tensor([1, 0]), rows=6, jitter=1)
    assert jittered.x.shape == batch.x.shape
    assert torch.equal(jittered.slot, batch.slot)


def segmented(length: int = 75) -> Split:
    made = split()
    return dataclasses.replace(made, windows=dataclasses.replace(made.windows, segment=length))


def test_segmented() -> None:
    batch = segmented().segmented(torch.tensor([1, 0]))
    assert batch.x.shape == (2, CHANNELS, 75)
    assert batch.slot[:, :DENSE].tolist() == [[0, 2, 2, 2], [1, 1, 1, 1]]
    window = split().windows(torch.tensor([0]))[0]
    assert torch.allclose(torch.diff(batch.x[1, :, :SAMPLES], dim=-1), torch.diff(window, dim=-1), atol=1e-5)
    assert batch.subject.shape == (2,)
    assert batch.onset.shape == batch.slot.shape
    assert torch.equal(segmented().batch(torch.tensor([1, 0]), rows=6, jitter=3).x, batch.x)


def test_capacity() -> None:
    s = split()
    natural = s.weight[s.index[:, 0]]
    assert s.capacity(1, natural) == 8
    assert s.capacity(100, natural) == 192
    longest = (s.index != PAD).sum(dim=1) == (s.index != PAD).sum(dim=1).max()
    assert s.capacity(100, longest.float()) > s.capacity(100, natural)


def test_permuted() -> None:
    """Window starts are moved together, the labels and onsets stay."""
    original = split(6)
    moved = original.permuted(0)
    order = torch.randperm(6, generator=torch.Generator().manual_seed(0))
    assert sorted(moved.windows.at.tolist()) == sorted(original.windows.at.tolist())
    assert not torch.equal(moved.windows.at, original.windows.at)
    assert torch.equal(moved.windows.end - moved.windows.at, original.windows.end[order] - original.windows.at[order])
    assert torch.equal(moved.y, original.y)
    assert torch.equal(moved.onset, original.onset)
    assert torch.equal(original.permuted(0).windows.at, moved.windows.at)


def test_ordered() -> None:
    """Words are dealt within their own span only, PAD slots stay, onsets and word data stay."""
    original = split(12)
    moved = original.ordered(0)
    valid = original.index != PAD
    assert torch.equal(moved.index == PAD, original.index == PAD)
    assert torch.equal(moved.index[1:], original.index[1:])
    assert sorted(moved.index[0][valid[0]].tolist()) == sorted(original.index[0][valid[0]].tolist())
    assert any(not torch.equal(original.ordered(seed).index[0], original.index[0]) for seed in range(8))
    assert torch.equal(moved.onset, original.onset)
    assert torch.equal(moved.y, original.y)
    assert moved.windows is original.windows


def test_ordered_travels() -> None:
    """A word's label and window follow it: reading a word's slot through the new index gives its own data."""
    original = split(12)
    moved = original.ordered(3)
    words = moved.index[moved.index != PAD]
    assert torch.equal(moved.y[words], original.y[words])
    assert torch.equal(moved.windows.at[words], original.windows.at[words])


def test_ordered_seeded() -> None:
    original = split(12)
    assert torch.equal(original.ordered(1).index, original.ordered(1).index)


def test_ordered_interior_pad() -> None:
    """A PAD between words keeps its slot."""
    original = dataclasses.replace(split(12), index=torch.tensor([[0, PAD, 1, 2], [3, 4, PAD, PAD]]))
    moved = original.ordered(0)
    assert torch.equal(moved.index == PAD, original.index == PAD)
    assert sorted(moved.index[0][moved.index[0] != PAD].tolist()) == [0, 1, 2]
    assert sorted(moved.index[1][moved.index[1] != PAD].tolist()) == [3, 4]


def numbered() -> Split:
    return split().numbered(pd.Series(["b", "a"] * 6))


def test_numbered() -> None:
    held = numbered()
    assert held.run is not None
    assert held.run.tolist() == [0, 1] * 6


def test_split_runs() -> None:
    assert numbered().runs() == 2
    assert split().runs() == 0


def test_words_at() -> None:
    batch = split().batch(torch.tensor([1, 0]), rows=6)
    assert batch.words_at(12)[:, :DENSE].tolist() == [[DENSE, 12, 12, 12], [0, 1, 2, 3]]


def test_span_input_runs() -> None:
    """A span position reads its row's run, -1 where empty; every row is one run where unknown."""
    known = dataclasses.replace(spans(), run=torch.tensor([4, 5, 6]))
    assert known.runs.tolist() == [[4, 6, -1]]
    assert spans().runs.tolist() == [[0, 0, -1]]


def test_runs_of() -> None:
    assert split().runs_of(torch.tensor([0, 1])) is None
    of = numbered().runs_of(torch.tensor([3, 0]))
    assert of is not None
    assert of.tolist() == [1, 0]
    batch = numbered().batch(torch.tensor([1, 0]), rows=6)
    assert batch.run is not None
    assert batch.inputs().run is batch.run
    assert batch.run[:5].tolist() == [DENSE % 2, 0, 1, 0, 1]
    whole = dataclasses.replace(segmented(), run=numbered().run).segmented(torch.tensor([1, 0]))
    assert whole.run is not None
    assert whole.run.tolist() == [DENSE % 2, 0]


def test_inputs() -> None:
    batch = split().batch(torch.tensor([1, 0]), rows=6)
    spans = batch.inputs()
    assert spans.x is batch.x
    assert spans.onset is batch.onset
    assert spans.slot is batch.slot
    assert spans.subject is batch.subject
    other = torch.zeros_like(batch.x)
    assert batch.inputs(other).x is other


def spans() -> SpanInput:
    return SpanInput(
        torch.zeros(3, 2, 4), torch.ones(1, 3), torch.tensor([[0, 2, 3]]), torch.zeros(3, dtype=torch.long)
    )


def test_mask() -> None:
    assert spans().mask.tolist() == [[True, True, False]]


def test_reading() -> None:
    x = torch.ones(3, 2, 4)
    read = spans().reading(x)
    assert read.x is x
    assert torch.equal(read.slot, spans().slot)


def test_timed() -> None:
    onset = torch.zeros(1, 3)
    timed = spans().timed(onset)
    assert timed.onset is onset
    assert torch.equal(timed.x, spans().x)


def test_spanned() -> None:
    original = split(6)
    kept = original.spanned(torch.tensor([2, 0]))
    assert torch.equal(kept.index, original.index[[2, 0]])
    assert torch.equal(kept.onset, original.onset[[2, 0]])
    assert torch.equal(kept.y, original.y)


def test_heard() -> None:
    original = split(6)
    heard = original.heard()
    assert set(heard.tolist()) <= set(original.index[original.index != PAD].tolist())
    assert (original.y[heard] != IGNORE).all()
    assert torch.equal(original.spanned(torch.tensor([], dtype=torch.long)).heard(), heard[:0])


def test_kaggle_like() -> None:
    assert split().kaggle_like().tolist() == list(range(DENSE))
    unknown = dataclasses.replace(split(), y=torch.where(torch.arange(12) == 0, IGNORE, split().y))
    assert unknown.kaggle_like().tolist() == list(range(1, DENSE))


def test_from_kaggle() -> None:
    windows = Windows.stacked(torch.zeros(3, CHANNELS, SAMPLES))
    held = Split.from_kaggle(Kaggle(), windows)
    assert held.windows is windows
    assert held.index[:, :2].tolist() == [[0, 1], [2, -1]]
    assert held.onset[0, :2].tolist() == pytest.approx([0.0, 0.3])
    assert held.run is not None
    assert held.run.tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="2 Kaggle test windows for 3"):
        Split.from_kaggle(Kaggle(), Windows.stacked(torch.zeros(2, CHANNELS, SAMPLES)))


def test_target() -> None:
    slot = torch.tensor([[0, 1, 2]])
    batch = Batch(torch.zeros(2, 1, 1), torch.zeros(1, 3), slot, torch.zeros(2), index=torch.tensor([[4, 3, PAD]]))
    assert batch.target(torch.tensor([0, 1, 2, 7, 8])).tolist() == [[8, 7, IGNORE]]


TOKEN = np.array([5, 6, 7, 5, 6, 7, 9, 9])
INDEX = np.array([[0, 1, 2], [3, 4, 5], [6, 7, PAD]])


def test_sequences() -> None:
    assert SpanText(INDEX, TOKEN).sequences() == [(5, 6, 7), (5, 6, 7), (9, 9)]
    spans = Spans.by_gap(np.array([0.0, 0.2, 9.0]))
    assert set(SpanText(spans.index, np.array([4, 5, 6])).sequences()) == {(4, 5), (6,)}


def test_repeats() -> None:
    """A recited text heard twice weighs each of its words half at power 1; a span too short to recite keeps 1."""
    text = SpanText(INDEX, TOKEN)
    assert text.repeats(1.0).tolist() == [0.5] * 6 + [1.0, 1.0]
    assert text.repeats(0.0).tolist() == [1.0] * 8
    shifted = SpanText(np.array([[PAD, 0, 1, 2], [3, 4, 5, PAD]]), TOKEN[:6])
    assert shifted.repeats(1.0).tolist() == [0.5] * 6
    unspanned = SpanText(np.array([[0, 1, 2]]), TOKEN)
    assert unspanned.repeats(1.0).tolist() == [1.0] * 8


def test_span_text_novel() -> None:
    text = SpanText(INDEX, TOKEN)
    assert text.novel(frozenset()).tolist() == list(range(8))
    assert text.novel(frozenset({(5, 6, 7)})).tolist() == [6, 7]
    assert len(SpanText(np.array([[0, PAD]]), TOKEN).novel(frozenset())) == 0


FEATURE = torch.arange(6.0).reshape(3, 2)


def saved(tmp_path: Path) -> Features:
    features = Features(tmp_path / "f")
    features.save(TRAIN, FEATURE)
    return features


def windows(count: int = 3) -> Windows:
    return Windows.stacked(torch.ones(count, 4, 5))


def test_under(tmp_path: Path):
    root = DataRoot(tmp_path)
    assert Features.under(root, "x").directory == root.kind("models") / FEATURES / "x"
    assert Features.under(root, "").directory is None
    joined = Features.under(root, "x+y")
    assert joined.directory == root.kind("models") / FEATURES / "x"
    assert joined.joined == (Features(root.kind("models") / FEATURES / "y"),)


def test_path(tmp_path: Path):
    assert Features(tmp_path).path(TRAIN) == tmp_path / "train.pt"
    with pytest.raises(ValueError, match="no features"):
        Features().path(TRAIN)


def test_save(tmp_path: Path):
    path = saved(tmp_path).path(TRAIN)
    assert torch.equal(torch.load(path, weights_only=True), FEATURE)


def test_read(tmp_path: Path):
    raw = windows()
    assert Features().read(TRAIN, raw) is raw
    read = saved(tmp_path).read(TRAIN, raw)
    assert torch.equal(read(torch.tensor([2, 0])), FEATURE[[2, 0], :, None])
    resolved = Features(tmp_path / "r")
    resolved.save(TRAIN, FEATURE.reshape(3, 1, 2))
    assert torch.equal(resolved.read(TRAIN, raw)(torch.tensor([1])), FEATURE[[1], None])
    with pytest.raises(ValueError, match="2 windows"):
        saved(tmp_path).read(TRAIN, windows(2))


def test_read_joined(tmp_path: Path):
    """Each word reads its first encoder's features, then the next's."""
    other = Features(tmp_path / "o")
    other.save(TRAIN, -FEATURE)
    joined = Features(tmp_path / "f", (other,))
    joined.save(TRAIN, FEATURE)
    read = joined.read(TRAIN, windows())(torch.tensor([1]))
    assert torch.equal(read, torch.cat([FEATURE, -FEATURE], dim=1)[[1], :, None])


def test_loaded(tmp_path: Path):
    assert saved(tmp_path).loaded(TRAIN, windows()).shape == (3, 2, 1)
    with pytest.raises(ValueError, match="3 train features for 2 windows"):
        saved(tmp_path).loaded(TRAIN, windows(2))


def test_split(tmp_path: Path):
    split = Split.of(windows(), (np.zeros(3, np.int64),) * 2, Spans.by_gap(np.array([0.0, 0.1, 9.0])), np.ones(3))
    read = saved(tmp_path).split(TRAIN, split)
    assert read.windows.length == 1
    assert read.y is split.y


def test_reads() -> None:
    """A span is read by its first word's run; unknown reads leave every span uncapped."""
    base = split()
    assert not base.reads().any()
    flagged = dataclasses.replace(base, read=torch.arange(12) < DENSE)
    assert flagged.reads().tolist() == [True] + [False] * (len(base.index) - 1)


def test_unmuted() -> None:
    held = split(4)
    assert held.unmuted(held.y) is held.y
    muted = dataclasses.replace(held, muted=torch.tensor([True, False, False, True]))
    assert muted.unmuted(muted.y).tolist() == [IGNORE, muted.y[1].item(), muted.y[2].item(), IGNORE]
    assert muted.y.tolist() == held.y.tolist()
