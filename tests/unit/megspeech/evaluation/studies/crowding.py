from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.probabilities import Probabilities, Words
from megspeech.evaluation.scoring.score import ADJUSTMENTS
from megspeech.evaluation.studies.crowding import (
    BLOCK,
    GAMMAS,
    MIXES,
    TAUS,
    TOP,
    Arm,
    Delta,
    Finish,
    Finishes,
    Grading,
    Rows,
    Saved,
    Section,
    Study,
)

WORDS = 400
CROWDERS = 5
SEED = 0


def rows(n: int = WORDS, seed: int = SEED, crowd: float = 3.0) -> Rows:
    """Words whose head favours the true label a little and `CROWDERS` frequent classes a lot."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(VOCAB_SIZE, (n,), generator=generator)
    head = torch.randn(n, VOCAB_SIZE, generator=generator)
    head[torch.arange(n), labels] += 1.5
    head[:, :CROWDERS] += crowd
    other = torch.randn(n, VOCAB_SIZE, generator=generator)
    other[torch.arange(n), labels] += 1.0
    return Rows(head.log_softmax(-1), other.log_softmax(-1), labels)


def uniform() -> torch.Tensor:
    return torch.zeros(VOCAB_SIZE).log_softmax(-1)


def test_take() -> None:
    data = rows(10)
    keep = torch.arange(10) % 2 == 0
    assert len(data.take(keep)) == 5
    assert torch.equal(data.take(keep).labels, data.labels[::2])


def test_swapped() -> None:
    data = rows(10)
    assert torch.equal(data.swapped().head, data.other)
    assert torch.equal(data.swapped().other, data.head)


def test_halves() -> None:
    side = rows(5 * BLOCK + 3).halves(SEED)
    assert len(side) == 5 * BLOCK + 3
    assert set(side.tolist()) == {0, 1}
    blocks = side[: 5 * BLOCK].view(5, BLOCK)
    assert bool((blocks == blocks[:, :1]).all())


def test_mean() -> None:
    labels = torch.tensor([1, 2, 3])
    logits = torch.randn(3, VOCAB_SIZE)
    word = Words(torch.arange(3), labels, torch.tensor([True, False, True]), logits, logits, logits.softmax(-1))
    averaged = Rows.mean([word, word], word.novel)
    assert len(averaged) == 2
    assert torch.allclose(averaged.head, logits.log_softmax(-1)[word.novel])


def test_scores() -> None:
    data = rows(4)
    prior = uniform()
    assert torch.allclose(Finish(0.0).scores(data, prior), data.head)
    assert torch.allclose(Finish(1.0, mix=0.5).scores(data, prior), data.head + 0.5 * data.other - prior)
    penalty = torch.arange(VOCAB_SIZE).float()
    assert torch.allclose(Finish(0.0, 2.0, crowding=penalty).scores(data, prior), data.head - 2 * penalty)


def test_inclusion() -> None:
    log_rate = Finish(0.0).inclusion([rows()], uniform())
    assert bool((log_rate[:CROWDERS] > log_rate[CROWDERS:].max()).all())
    assert float(log_rate.exp().sum()) == pytest.approx(VOCAB_SIZE, rel=0.1)


def test_fitted() -> None:
    finish = Finish(0.0, 1.0).fitted([rows()], uniform())
    assert finish.crowding is not None
    assert torch.allclose(finish.crowding, Finish(0.0).inclusion([rows()], uniform()))


def test_label() -> None:
    assert Finish(0.5, 0.25, 0.1).label() == "tau 0.5 gamma 0.25 mix 0.10"


def test_all() -> None:
    arms = Arm.all()
    assert [arm.name for arm in arms] == ["tau3", "tau", "crowd", "headmix"]
    assert len(arms[2].grid) == len(TAUS) * len(GAMMAS)
    assert len(arms[3].grid) == len(TAUS) * len(MIXES)


def test_arm_fit() -> None:
    crowd = next(arm for arm in Arm.all() if arm.name == "crowd")
    finish = crowd.fit({"a": rows()}, uniform(), Ruler(k=TOP))
    assert finish.gamma > 0
    assert finish.crowding is not None


def test_point() -> None:
    data = {"a": rows()}
    ruler = Ruler(k=TOP)
    crowded = Grading.point(Finish(0.0), data, uniform(), ruler)
    spread = Grading.point(Finish(0.0, 1.0).fitted(list(data.values()), uniform()), data, uniform(), ruler)
    assert spread > crowded


def test_grading_crossfit() -> None:
    data = {"a": rows(), "b": rows(seed=1)}
    grading = Grading.crossfit(Arm.all()[2], data, uniform(), SEED)
    assert {name: len(hits) for name, hits in grading.hits.items()} == {"a": WORDS, "b": WORDS}


def test_grading_transfer() -> None:
    grading = Grading.transfer(Arm.all()[0], {"a": rows()}, {"b": rows(seed=2)}, uniform())
    assert len(grading.hits["b"]) == WORDS


def test_graded() -> None:
    data = {"a": rows(3)}
    graded = Grading({"a": torch.tensor([True, False, True])}).graded(data)
    assert torch.equal(graded.domains["a"][1], data["a"].labels)


def test_of() -> None:
    data = {"a": rows()}
    better = Grading({"a": torch.ones(WORDS, dtype=torch.bool)})
    worse = Grading({"a": torch.zeros(WORDS, dtype=torch.bool)})
    delta = Delta.of(better, worse, data)
    assert delta.point == pytest.approx(1.0)
    assert delta.low == pytest.approx(1.0)
    assert delta.domains["a"][0] == pytest.approx(1.0)


def test_record() -> None:
    record = Delta(0.1, 0.0, 0.2, {"a": (0.1, 0.0, 0.2)}).record()
    assert record["delta"] == pytest.approx(0.1)
    assert record["domains"] == {"a": (0.1, 0.0, 0.2)}


def probabilities(novel: torch.Tensor, seed: int = SEED) -> Probabilities:
    data = rows(len(novel), seed)
    words = Words(torch.arange(len(novel)), data.labels, novel, data.head, data.other, data.head.exp())
    return Probabilities({"a": words}, 0.5, uniform(), 100)


def saved(directory: Path, section: str, novel: torch.Tensor) -> None:
    probabilities(novel).save(directory / "probs.pt", section)


def members(seed: int = SEED) -> Saved:
    novel = torch.arange(WORDS) % 4 != 0
    lead, other = probabilities(novel), probabilities(novel, seed)
    tiny = replace(lead.domains["a"], novel=torch.arange(WORDS) < 3)
    return Saved(
        [
            replace(lead, domains={"a": lead.domains["a"], "tiny": tiny}),
            replace(other, domains={"a": other.domains["a"], "tiny": other.domains["a"]}),
        ]
    )


def written(root: Path) -> list[Path]:
    novel = torch.arange(WORDS) % 4 != 0
    for member in ("m1", "m2"):
        (root / member).mkdir()
        saved(root / member, "val", novel)
    return [root / "m1", root / "m2"]


def test_section_load(tmp_path: Path) -> None:
    section = Section.load(written(tmp_path), "val", {"a": torch.arange(WORDS) < 200})
    assert len(section.domains["a"]) == WORDS - WORDS // 4
    assert len(section.lone["a"]) == 200


def test_saved_load(tmp_path: Path) -> None:
    assert len(Saved.load(written(tmp_path), "val").members) == 2


def test_section_load_disagreeing(tmp_path: Path) -> None:
    for seed, member in enumerate(("m1", "m2")):
        (tmp_path / member).mkdir()
        probabilities(torch.ones(WORDS, dtype=torch.bool), seed).save(tmp_path / member / "probs.pt", "val")
    with pytest.raises(ValueError, match="members disagree on the words of a in val"):
        Section.load([tmp_path / "m1", tmp_path / "m2"], "val", {})


def test_lead() -> None:
    found = members()
    assert found.lead("tiny") is found.members[0].domains["tiny"]
    assert int(found.lead("tiny").novel.sum()) == 3


def test_counted() -> None:
    assert members().counted() == ["a"]


def test_words() -> None:
    assert len(members().words("a")) == 2


def test_agreed() -> None:
    assert members().agreed("a")
    assert not members(seed=1).agreed("a")


def test_domains() -> None:
    domains = members().domains()
    assert set(domains) == {"a"}
    assert len(domains["a"]) == WORDS - WORDS // 4


def test_ids() -> None:
    assert members().ids()["a"].tolist() == [i for i in range(WORDS) if i % 4]


def test_lone() -> None:
    lone = members().lone({"a": torch.arange(WORDS) < 200, "tiny": torch.arange(WORDS) < 3})
    assert set(lone) == {"a"}
    assert torch.equal(lone["a"].head, members().words("a")[0].word_logits.float().log_softmax(-1)[:200])


def test_log_prior() -> None:
    assert torch.equal(members().log_prior(), uniform())


def test_finishes_fit() -> None:
    finishes = Finishes.fit(members())
    assert finishes.context.tau in ADJUSTMENTS
    assert finishes.word.tau in ADJUSTMENTS
    assert torch.equal(finishes.log_prior, uniform())


def test_prior() -> None:
    assert Finishes(Finish(0.0), Finish(0.0), uniform()).prior() == pytest.approx(np.full(VOCAB_SIZE, 1 / VOCAB_SIZE))


def test_scored() -> None:
    test = {"a": rows(6), "b": rows(4, seed=1)}
    context, word = Finishes(Finish(0.0), Finish(1.0), uniform()).scored(test)
    assert context.domains.tolist() == ["a"] * 6 + ["b"] * 4
    assert np.allclose(context.probs[:6], test["a"].head.numpy())
    assert np.allclose(word.probs[6:], (test["b"].other - uniform()).numpy())
    assert np.array_equal(word.labels, context.labels)


def study() -> Study:
    val = Section({"a": rows()}, {"a": rows(seed=3)}, uniform())
    test = Section({"a": rows(seed=4)}, {"a": rows(seed=5)}, uniform())
    return Study(val, test)


def test_run() -> None:
    result = study().run(SEED)
    assert set(result) == {"val-by-tau", "val", "lone-val", "test", "lone-test"}


def test_study_crossfit() -> None:
    out = study().crossfit(Arm.all()[:2], {"a": rows()}, SEED)
    assert set(out) == {"base", "n", "tau"}


def test_study_transfer() -> None:
    out = study().transfer(Arm.all()[:3], {"a": rows()}, {"a": rows(seed=6)})
    assert set(out) == {"base", "n", "tau", "crowd"}


def test_compared() -> None:
    data = {"a": rows(3)}
    hits = Grading({"a": torch.tensor([True, True, False])})
    out = Study.compared(Arm.all()[:2], [hits, hits], data)
    record = out["tau"]
    assert isinstance(record, dict)
    assert record["delta"] == pytest.approx(0.0)
