from pathlib import Path

import torch

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.evaluation.studies.boosted import Rows
from megspeech.evaluation.studies.lmfusion import ARMS, CANDIDATES, GRID, Arm, Fusion, Priors

CLASSES = len(CANDIDATES)
KEYS = ((("a",), ()), ((), ("b", "c")))


def priors() -> Priors:
    generator = torch.Generator().manual_seed(0)
    return Priors(KEYS, torch.randn(2, CLASSES, 3, generator=generator), torch.randn(CLASSES, generator=generator))


def rows() -> Rows:
    count = 4
    return Rows(
        torch.randn(count, CLASSES).log_softmax(-1),
        torch.zeros(count, 1),
        torch.zeros(count, CLASSES),
        torch.zeros(count, 2),
        torch.tensor([0, IGNORE, 5, 7]),
        torch.full((count, 2), -1),
        torch.zeros(count, dtype=torch.long),
    )


def made() -> Fusion:
    return Fusion.of(rows(), [KEYS[0], KEYS[1]], torch.tensor([0, 2]), priors())


def test_path() -> None:
    assert Priors.path(DataRoot(Path("x")), "lm").name == "lmprior-lm.pt"


def test_empty() -> None:
    empty = Priors.empty(torch.zeros(CLASSES))
    assert empty.table.shape == (0, CLASSES, 3)
    assert empty.keys == ()


def test_evidence() -> None:
    made_priors = priors()
    evidence = made_priors.evidence([KEYS[1], KEYS[0], KEYS[1]])
    assert evidence.shape == (3, CLASSES, 4)
    assert torch.allclose(evidence.exp().sum(1), torch.ones(3, 4), atol=1e-4)
    assert torch.allclose(evidence[:, :, 0], evidence[0, :, 0].expand(3, -1))
    assert torch.equal(evidence[0], evidence[2])


def test_of() -> None:
    fusion = made()
    assert fusion.evidence.shape == (2, CLASSES, 4)
    assert fusion.scored.tolist() == [0, 2]


def test_fused() -> None:
    fusion = made()
    assert torch.allclose(fusion.fused(ARMS[1], 0.0), fusion.base, atol=1e-5)
    out = fusion.fused(ARMS[1], 1.0)
    assert torch.allclose(out[1], fusion.base[1])
    assert not torch.allclose(out[0], fusion.base[0])
    assert torch.allclose(out.exp().sum(-1), torch.ones(4), atol=1e-4)


def test_pmi_removes_unigram() -> None:
    fusion = made()
    plain, pmi = Arm("l", 1), Arm("p", 1, pmi=True)
    assert not torch.allclose(fusion.fused(plain, 1.0)[0], fusion.fused(pmi, 1.0)[0])


def test_accuracy() -> None:
    assert 0.0 <= made().accuracy(ARMS[1], GRID[1]) <= 1.0


def test_pick() -> None:
    y = torch.tensor([IGNORE, 1, 2, IGNORE, 3, 4])
    chosen = Fusion.pick(y, 3, 0)
    assert len(chosen) == 3
    assert (y[chosen] != IGNORE).all()
    assert chosen.tolist() == sorted(chosen.tolist())
    assert Fusion.pick(y, 3, 0).tolist() == chosen.tolist()
    assert len(Fusion.pick(y, 99, 0)) == 4
