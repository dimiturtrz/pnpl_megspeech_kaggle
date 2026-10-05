from pathlib import Path

import numpy as np
import pandas as pd

from megspeech.data.cache.run_cache import WORD_TABLE, RunCache
from megspeech.evaluation.studies.oracle import Oracle

SPOKEN = ["the", "of", "the", "of", "zzzzqq", "the"]
DIM = 4


def oracle(tmp_path: Path) -> Oracle:
    cache = RunCache(tmp_path / "meg")
    cache.ready()
    np.save(cache.path("run"), np.zeros((2, 20_000), dtype=np.float16))
    count = len(SPOKEN)
    table = pd.DataFrame(
        {
            "run": "run",
            "onset": np.arange(count, dtype=float),
            "end": np.arange(count) + 0.5,
            "word": SPOKEN,
            "sentence": 0,
        }
    )
    table.to_csv(cache.cache / WORD_TABLE, sep="\t", index=False)
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "run.npy", np.random.default_rng(0).normal(size=(count, DIM)).astype(np.float16))
    return Oracle(source, cache)


def test_words(tmp_path: Path) -> None:
    """Each word keeps its place in its run and is labelled by the vocabulary, or ignored outside it."""
    words = oracle(tmp_path).words()
    assert words["position"].tolist() == list(range(len(SPOKEN)))
    assert words["target"].iloc[0] == words["target"].iloc[2] != words["target"].iloc[1]
    assert words["target"].iloc[4] < 0


def test_write(tmp_path: Path) -> None:
    """Words of one vocabulary entry share one target; a word outside the vocabulary has none."""
    destination = tmp_path / "oracle"
    oracle(tmp_path).write(destination)
    written = np.load(destination / "run.npy")
    assert written.shape == (len(SPOKEN), DIM)
    assert np.array_equal(written[0], written[2]) and np.array_equal(written[0], written[5])
    assert np.isnan(written[4]).all()
