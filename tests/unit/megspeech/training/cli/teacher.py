import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.objective.targets.speech import Speech
from megspeech.training.cli.teacher import BINS, HELD_EVERY, Probe

CPU = torch.device("cpu")
RUNS = 10
PER_RUN = 60
FEATURES = 6


def split(n: int = 8) -> Split:
    x = np.random.default_rng(0).standard_normal((n, 2, 4 * BINS)).astype(np.float32)
    return Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(np.arange(n) * 2.0), CPU)


def probe(noise: float = 0.0) -> Probe:
    """Targets linear in the features of ten runs of sixty words, the first word of each run without one."""
    generator = np.random.default_rng(1)
    features = torch.from_numpy(generator.standard_normal((RUNS * PER_RUN, FEATURES)).astype(np.float32))
    weights = torch.from_numpy(generator.standard_normal((FEATURES, 3)).astype(np.float32))
    vectors = features @ weights + noise * torch.from_numpy(
        generator.standard_normal((RUNS * PER_RUN, 3)).astype(np.float32)
    )
    positions = pd.DataFrame({"run": np.repeat([f"run{i:02d}" for i in range(RUNS)], PER_RUN)})
    rows = torch.arange(RUNS * PER_RUN)
    rows[::PER_RUN] = -1
    return Probe.of(features, positions, Speech(rows, vectors))


def test_binned() -> None:
    one = split()
    binned = Probe.binned(one)
    assert binned.shape == (8, 2 * BINS)
    expected = nnf.adaptive_avg_pool1d(one.windows(torch.arange(8)).float(), BINS).flatten(1)
    assert torch.allclose(binned.float(), expected, atol=1e-2)


def test_of() -> None:
    one = probe()
    assert one.x.shape == (RUNS * PER_RUN - RUNS, FEATURES)
    assert one.y.shape == (RUNS * PER_RUN - RUNS, 3)
    assert one.held.sum().item() == (RUNS // HELD_EVERY) * (PER_RUN - 1)
    assert one.x[~one.held].mean(0).abs().max().item() < 1e-5
    assert torch.allclose(one.y[~one.held].std(0), torch.ones(3), atol=1e-4)


def test_r2() -> None:
    """A noiseless linear target is read almost whole at a small penalty, and a huge one reads less of it."""
    scores = probe().r2((1e-3, 1e3))
    assert scores[1e-3] > 0.99
    assert scores[1e3] < scores[1e-3]
    assert probe(noise=5.0).r2((1.0,))[1.0] < 0.2
