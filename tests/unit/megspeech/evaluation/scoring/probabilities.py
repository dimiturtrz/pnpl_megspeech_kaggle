from pathlib import Path

import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities, Words
from megspeech.evaluation.scoring.score import Head
from tests.unit.megspeech.evaluation.scoring.evaluator import CPU, Constant, split

PRIOR = torch.zeros(VOCAB_SIZE)


def evaluator() -> Evaluator:
    return Evaluator(CPU, min_novel=1)


def words() -> Words:
    domain = split()
    return Words.of(evaluator(), evaluator().logits(Constant(), domain), domain, PRIOR, 0.5)


def test_words_of() -> None:
    domain = split()
    read = words()
    assert torch.equal(read.ids, domain.labelled().cpu())
    assert torch.equal(read.labels, domain.y[domain.labelled()].cpu())
    assert read.probabilities.shape == (len(read.ids), VOCAB_SIZE)
    assert torch.allclose(read.probabilities.sum(-1), torch.ones(len(read.ids)), atol=1e-4)
    assert torch.equal(read.novel, torch.isin(read.ids, domain.novel(frozenset()).cpu()))


def test_record() -> None:
    record = words().record()
    assert set(record) == {"ids", "labels", "novel", "logits", "word_logits", "probabilities"}


def probabilities() -> Probabilities:
    return Probabilities.of(evaluator(), Constant(), {"a": split(), "b": split(n=8, seed=1)}, PRIOR, 0.5)


def test_probabilities_of() -> None:
    made = probabilities()
    assert list(made.domains) == ["a", "b"]
    assert (made.tau, made.min_novel) == (0.5, 1)
    assert torch.equal(made.domains["a"].logits, evaluator().logits(Constant(), split())[Head.CONTEXT])


def test_stored(tmp_path: Path) -> None:
    assert Probabilities.stored(tmp_path / PROBS) == {}
    probabilities().save(tmp_path / PROBS, "val")
    assert set(Probabilities.stored(tmp_path / PROBS)) == {"val#meta", "val/a", "val/b"}


def test_save(tmp_path: Path) -> None:
    """Saving a section leaves the others, and saving it again replaces only its own."""
    path = tmp_path / PROBS
    made = probabilities()
    assert made.save(path, "val") == path
    made.save(path, "local-test")
    Probabilities.of(evaluator(), Constant(), {"c": split()}, PRIOR, 1.0).save(path, "val")
    assert set(Probabilities.stored(path)) == {"val#meta", "val/c", "local-test#meta", "local-test/a", "local-test/b"}


def test_load(tmp_path: Path) -> None:
    """A section round-trips beside another and the first's replacement."""
    path = tmp_path / PROBS
    made = probabilities()
    made.save(path, "local-test")
    Probabilities.of(evaluator(), Constant(), {"c": split()}, PRIOR, 1.0).save(path, "val")
    val = Probabilities.load(path, "val")
    assert list(val.domains) == ["c"]
    assert val.tau == 1.0
    kept = Probabilities.load(path, "local-test")
    assert list(kept.domains) == ["a", "b"]
    assert kept.tau == 0.5
    assert torch.equal(kept.domains["a"].probabilities, made.domains["a"].probabilities)
    assert torch.equal(kept.log_prior, PRIOR)
    assert kept.min_novel == 1


def test_load_missing(tmp_path: Path) -> None:
    probabilities().save(tmp_path / PROBS, "val")
    with pytest.raises(KeyError):
        Probabilities.load(tmp_path / PROBS, "absent")


def test_probabilities_from_logits() -> None:
    domains = {"a": split(), "b": split(n=8, seed=1)}
    logits = {name: evaluator().logits(Constant(), s) for name, s in domains.items()}
    made = Probabilities.from_logits(evaluator(), logits, domains, PRIOR, 0.5)
    assert torch.equal(made.domains["b"].probabilities, probabilities().domains["b"].probabilities)
