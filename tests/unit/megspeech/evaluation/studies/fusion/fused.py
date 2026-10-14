import logging
from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.fusion.fused import Fusion
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded.model import loaded, runs

WEIGHTS = (0.0, 0.5, 1.0)


class Silent:
    """A source with no opinion: uniform zeros over the vocabulary."""

    def logp(self, split: Split, context: Float[Tensor, "n v"]) -> Float[Tensor, "n v"]:
        return torch.zeros(len(split.y), context.shape[1])


def fusion(tmp_path: Path) -> Fusion:
    return Fusion(loaded(tmp_path), Silent(), WEIGHTS, "silent")


def test_domain(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A silent source leaves every weight at the context head's own unseen-span score."""
    model = loaded(tmp_path)
    domains, _ = model.domains({RULER: runs(tmp_path)[:1]})
    split = domains[RULER]
    alone = model.evaluator.unseen(model.evaluator.logits(model.model, split)[Head.CONTEXT], split, model.log_prior)
    with caplog.at_level(logging.INFO):
        scores = fusion(tmp_path).domain(split)
    assert scores == {weight: pytest.approx(alone.bacc10) for weight in WEIGHTS}
    assert f"silent alone 0.0000 over {alone.n} words" in caplog.text


def test_table(tmp_path: Path) -> None:
    """Every eligible domain is read, each at every weight."""
    table = fusion(tmp_path).table({RULER: runs(tmp_path)[:1]})
    assert list(table) == [RULER]
    assert list(table[RULER]) == list(WEIGHTS)
