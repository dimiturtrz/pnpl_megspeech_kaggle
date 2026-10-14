from pathlib import Path

import pandas as pd
import torch

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE, PRIMARY
from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.training.timing.corpus import FILE, TimingCorpus

KNOWN = PRIMARY.words[0]


def frame() -> pd.DataFrame:
    """Span 5 of three words (one out of vocabulary), span 2 of eleven words."""
    rows = [(5, KNOWN, 0.0), (5, "zzzqx", 0.3), (5, KNOWN, 0.9)]
    rows += [(2, KNOWN, 0.2 * i) for i in range(MAX_WORDS + 2)]
    return pd.DataFrame(rows, columns=["span", "word", "onset"])


def test_from_frame() -> None:
    """Spans keep their words in order, onsets restart at each span's first word, long spans are cut to the width."""
    corpus = TimingCorpus.from_frame(frame())
    assert corpus.index.shape[1] == MAX_WORDS
    assert (corpus.index != PAD).sum().item() == len(corpus.y) == 14
    assert corpus.onset[:, 0].eq(0).all()
    last = corpus.index[-1][corpus.index[-1] != PAD]
    assert corpus.y[last].tolist() == [0, IGNORE, 0]
    assert torch.allclose(corpus.onset[-1][: len(last)], torch.tensor([0.0, 0.3, 0.9]))


def test_read(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    frame().to_parquet(root.processed("tiny") / FILE)
    assert len(TimingCorpus.read(root, "tiny").y) == 14


def test_to() -> None:
    moved = TimingCorpus.from_frame(frame()).to(torch.device("cpu"))
    assert moved.y.device.type == "cpu"


def test_describe() -> None:
    """Spans, words, and the labelled share (13 of 14 words are in the vocabulary)."""
    assert TimingCorpus.from_frame(frame()).describe() == "3 spans, 14 words, 0.9286 labelled"
