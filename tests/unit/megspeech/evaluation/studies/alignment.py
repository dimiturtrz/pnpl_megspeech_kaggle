from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.evaluation.studies.alignment import Alignment, Anisotropy, Cca, Directory, Lexical, Reach, Study, Words
from megspeech.objective.targets.speech import Targets
from megspeech.objective.targets.t5 import Lexicon

ROWS = 400
SHUFFLES = 20


def pair(rows: int = ROWS, shared: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Two views whose first `shared` coordinates carry one signal and the rest are noise."""
    generator = torch.Generator().manual_seed(0)
    signal = torch.randn(rows, shared, generator=generator)
    x = torch.cat([signal, torch.randn(rows, 4, generator=generator)], dim=1)
    y = torch.cat(
        [signal @ torch.randn(shared, shared, generator=generator), torch.randn(rows, 6, generator=generator)], 1
    )
    return x, y


def test_anisotropy_of() -> None:
    generator = torch.Generator().manual_seed(0)
    spread = Anisotropy.of(torch.randn(ROWS, 8, generator=generator))
    lopsided = Anisotropy.of(torch.randn(ROWS, 8, generator=generator) + 50 * torch.ones(8))
    assert abs(spread.cosine) < 0.1
    assert lopsided.cosine > 0.9
    assert 0 < spread.top < 1


def test_fit() -> None:
    x, y = pair()
    cca = Cca.fit(x, y)
    assert cca.axes_x.shape == (6, 6)
    assert cca.axes_y.shape[0] == 8
    corr = cca.correlations(x, y)
    assert corr[:2].min() > 0.9
    assert corr[2:].max() < corr[:2].min()


def test_inverse_root() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    root = Cca.inverse_root(covariance, 0.0)
    assert torch.allclose(root @ covariance @ root, torch.eye(2, dtype=torch.float64))


def test_projected() -> None:
    x, y = pair()
    a, b = Cca.fit(x, y).projected(x, y)
    assert a.shape == (ROWS, 6)
    assert b.shape == (ROWS, 6)


def test_correlation() -> None:
    a = torch.randn(50, 3, dtype=torch.float64)
    assert torch.allclose(Cca.correlation(a, 2 * a + 1), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(Cca.correlation(a, -a), -torch.ones(3, dtype=torch.float64))


def test_correlations() -> None:
    x, y = pair()
    cca = Cca.fit(x[:200], y[:200])
    assert cca.correlations(x[200:], y[200:])[0] > 0.8


def test_shuffled() -> None:
    x, y = pair()
    null = Cca.fit(x, y).shuffled(x, y, SHUFFLES)
    assert null.shape == (SHUFFLES, 6)
    assert null.abs().max() < 0.4


def test_reach_of() -> None:
    x, y = pair()
    reach = Reach.of(Cca.fit(x[:200], y[:200]), x[200:], y[200:], SHUFFLES)
    assert reach.profile.shape == (6,)
    assert reach.null.shape == (SHUFFLES, 6)


def test_per_component() -> None:
    reach = Reach(torch.tensor([0.9, 0.5, 0.0]), torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
    assert reach.per_component == 2


def test_family() -> None:
    reach = Reach(torch.tensor([0.9, 0.15, 0.0]), torch.tensor([[0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]))
    assert reach.family == 1


def test_top() -> None:
    assert Reach(torch.tensor([0.9, 0.5]), torch.zeros(1, 2)).top(2) == pytest.approx(0.5)


def test_directory_gather(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    Targets(np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]], dtype=np.float16)).save(
        Targets.directory(root, "x") / "r.npy"
    )
    words = pd.DataFrame({"run": "r", "position": [0, 1, 2]})
    present, vectors = Directory("x", root).gather(words)
    assert present.tolist() == [True, False, True]
    assert vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_lexical_gather() -> None:
    lexicon = Lexicon(torch.tensor([3, 7]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    present, vectors = Lexical(lexicon).gather(pd.DataFrame({"token": [7, 5, 3]}))
    assert present.tolist() == [True, False, True]
    assert vectors.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_curve() -> None:
    reach = Reach(torch.tensor([0.9, 0.5]), torch.zeros(1, 2))
    assert Alignment("t", "v", 8, 10, 5, Anisotropy(0.5, 0.25), reach, 0.1).curve() == "0.90 0.50"


def test_row() -> None:
    reach = Reach(torch.tensor([0.9] * 20), torch.zeros(2, 20))
    line = Alignment("t", "v", 8, 10, 5, Anisotropy(0.5, 0.25), reach, 0.1).row()
    assert line.startswith("| t / v | 8 | 10/5 | 0.500 / 0.250 | 20 / 20 | 0.900 / 0.900 / 0.900 | 0.1000 |")


def words(rows: int = 6) -> Words:
    table = pd.DataFrame(
        {"run": "r", "position": np.arange(rows), "target": [0, 1] * (rows // 2), "token": np.arange(rows)}
    )
    return Words(table, torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2))


def test_sample() -> None:
    assert len(words().sample(4).table) == 4
    assert words().sample(4).vectors.shape == (4, 2)
    assert len(words().sample(10).table) == 6


def test_everything() -> None:
    vectors, labels = words().everything()
    assert vectors.shape == (6, 2)
    assert labels.tolist() == [0, 1, 0, 1, 0, 1]


def test_labels() -> None:
    assert words().labels(np.array([True, False, True, False, False, False])).tolist() == [0, 0]


def test_ceiling() -> None:
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(100) % 50
    train = torch.nn.functional.one_hot(labels, 50).float() + 0.01 * torch.randn(100, 50, generator=generator)
    assert Study.ceiling((train, labels), (train, labels)) > 0.9


def test_meg_ceiling() -> None:
    labels = torch.arange(100) % 50
    table = pd.DataFrame({"target": labels.numpy(), "run": "r", "position": 0, "token": 0})
    vectors = torch.nn.functional.one_hot(labels, 50).float()
    study = Study(Words(table, vectors), Words(table, vectors))
    assert study.meg_ceiling() > 0.9


def test_align() -> None:
    x, y = pair()
    table = pd.DataFrame({"target": np.arange(ROWS) % 50, "run": "r", "position": 0, "token": np.arange(ROWS)})
    lexicon = Lexicon(torch.arange(ROWS), y)
    study = Study(Words(table, x), Words(table, x))
    result = study.align("t", "v", Lexical(lexicon), SHUFFLES)
    assert result is not None
    assert (result.dims, result.train, result.val) == (8, ROWS, ROWS)
    assert result.reach.top(1) > 0.9
    assert Study(Words(table, x), Words(table.head(5), x[:5])).align("t", "v", Lexical(lexicon), SHUFFLES) is None


def test_by_runs() -> None:
    table = pd.DataFrame({"run": list("aabbccddeeff"), "position": 0, "target": 0, "token": 0})
    fit, held = Words(table, torch.arange(12.0).reshape(12, 1)).by_runs()
    assert sorted(held.table["run"].unique()) == ["a", "f"]
    assert len(fit.table) + len(held.table) == 12
    assert held.vectors.shape == (4, 1)


def test_held_out() -> None:
    table = pd.DataFrame({"run": list("aabbccddeeff"), "position": 0, "target": 0, "token": 0})
    words = Words(table, torch.zeros(12, 1))
    assert len(Study(words, words).held_out().val.table) == 4


def test_teacher(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert isinstance(Study.teacher(root, "gpt2-large"), Directory)
