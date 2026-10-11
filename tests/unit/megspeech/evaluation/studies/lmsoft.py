import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.crowding import Rows, Section
from megspeech.evaluation.studies.lmbound import CANDIDATES, RIGHT, Causal, Context, Table
from megspeech.evaluation.studies.lmrows import Posteriors
from megspeech.evaluation.studies.lmsoft import (
    CHANNELS,
    EMPTY,
    OOV,
    Batched,
    Belief,
    Dumped,
    Fusion,
    Kaggle,
    Neighbours,
    Study,
    Weighted,
)

TOKENS = 61
CAPITAL = CANDIDATES[3][:1].upper() + CANDIDATES[3][1:]


def score(key: Context) -> torch.Tensor:
    """A fake LM: candidate `len(left) + len(right)` is likely."""
    out = torch.zeros(VOCAB_SIZE, RIGHT + 1)
    out[(len(key[0]) + len(key[1])) % VOCAB_SIZE] = 5.0
    return out


def table(keys: list[Context]) -> Table:
    return Table.scored(score, keys)


def causal() -> Causal:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=TOKENS, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    return Causal(model, lambda texts: [[len(text) % TOKENS, ord(text[1]) % TOKENS] for text in texts], 0)


def test_soft() -> None:
    belief = Belief.soft(torch.zeros(2, VOCAB_SIZE), 0.25)
    assert belief.q.shape == (2, VOCAB_SIZE + 1)
    assert float(belief.q[0, OOV]) == pytest.approx(0.75)
    assert float(belief.q[0].sum()) == pytest.approx(1.0)


def test_oracle() -> None:
    belief = Belief.oracle(torch.tensor([3, IGNORE]))
    assert int(belief.q[0].argmax()) == 3
    assert int(belief.q[1].argmax()) == OOV


def test_outward() -> None:
    belief = Belief.oracle(torch.tensor([3, 4, IGNORE]))
    assert belief.outward([0, 1, 2], (4, 2)) == [((3, 4), 1.0)]
    assert belief.outward([2, 0], (4, 2)) == [((), 1.0)]
    assert belief.outward([], (4,)) == [((), 1.0)]
    soft = Belief.soft(torch.zeros(2, VOCAB_SIZE), 0.5).outward([0, 1], (2, 1))
    assert len(soft) == 1 + 2 * 2
    assert sum(weight for _, weight in soft) < 1


def test_weighted_of() -> None:
    weighted = Weighted.of([(EMPTY, 1.0), ((("a",), ()), 3.0)])
    assert torch.allclose(weighted.weights, torch.tensor([0.25, 0.75]))


def test_other() -> None:
    keys = [EMPTY, (("a",), ())]
    lone = Weighted.of([(EMPTY, 1.0)]).other(table(keys), 0)
    assert torch.allclose(lone, torch.zeros(VOCAB_SIZE))
    mixed = Weighted.of([((("a",), ()), 1.0)]).other(table(keys), 0)
    assert int(mixed.argmax()) == 1


def neighbours() -> Neighbours:
    return Neighbours.of(Belief.oracle(torch.tensor([3, IGNORE, 4, 5])), torch.tensor([[0, 1, 2, 3], [PAD] * 4]))


def test_neighbours_of() -> None:
    assert neighbours().place[2] == ([0, 1, 2, 3], 2)


def test_channels() -> None:
    out = neighbours().channels(2)
    assert out["pmi"].contexts == (EMPTY,)
    assert out["pmi-right"].contexts == (((), (CANDIDATES[5],)),)
    first = neighbours().channels(1)
    assert first["pmi"].contexts == (((CAPITAL,), ()),)
    assert Neighbours(Belief.oracle(torch.tensor([3])), {}).channels(0)["pmi-right"].contexts == (EMPTY,)


def test_spelled() -> None:
    assert Neighbours.spelled((3, 4), 0) == (CAPITAL, CANDIDATES[4])
    assert Neighbours.spelled((3,), 1) == (CANDIDATES[3],)
    assert Neighbours.spelled((), 0) == ()


def test_score() -> None:
    lm = causal()
    keys: list[Context] = [EMPTY, (("the", "dog"), ("was",)), (("a",), ("it", "was"))]
    batched = Batched(lm).score(keys)
    for key, out in zip(keys, batched, strict=True):
        assert torch.allclose(out, lm.score(key), atol=1e-4)


def test_steps() -> None:
    out = Batched(causal()).steps([[0, 1, 2], [0, 3]], torch.tensor([1, 1]))
    assert out.shape == (2, 2)
    assert bool((out[:, 0] < 0).all())
    assert float(out[0, 1]) < float(out[0, 0])


def test_table() -> None:
    lm = causal()
    known = {EMPTY: lm.score(EMPTY)}
    made = Batched(lm).table([(("a",), ()), EMPTY], known)
    assert set(made.scores) == {EMPTY, (("a",), ())}
    assert torch.equal(made.unigram, known[EMPTY][:, 0])


def section() -> Section:
    head = torch.zeros(4, VOCAB_SIZE).log_softmax(-1)
    return Section({"a": Rows(head, head, torch.tensor([3, 4, 5, 3]))}, {}, head[0])


def built() -> Fusion:
    index = torch.tensor([[0, 1, 2, 3]])
    beliefs = {
        "oracle": Neighbours.of(Belief.oracle(torch.tensor([3, 4, 5, 3])), index),
        "soft": Neighbours.of(Belief.soft(torch.zeros(4, VOCAB_SIZE), 0.3), index),
    }
    return Fusion.of(section(), {name: {"a": belief} for name, belief in beliefs.items()}, {"a": torch.arange(4)})


def test_fusion_of() -> None:
    assert set(built().weighted["soft"]) == set(CHANNELS)
    assert len(built().weighted["oracle"]["pmi"]["a"]) == 4


def test_wanted() -> None:
    assert EMPTY in built().wanted()


def test_arms() -> None:
    assert built().arms() == [(belief, channel) for belief in ("oracle", "soft") for channel in CHANNELS]


def test_fused() -> None:
    made = built()
    fused = made.fused(table(made.wanted()), "oracle", "pmi")
    assert fused["a"].other.shape == (4, VOCAB_SIZE)
    assert torch.allclose(fused["a"].other[0], torch.zeros(VOCAB_SIZE))


def test_run() -> None:
    labels = torch.arange(64) % VOCAB_SIZE
    head = torch.zeros(64, VOCAB_SIZE).log_softmax(-1)
    index = torch.arange(64).view(16, 4)
    graded = Section({"a": Rows(head, head, labels)}, {}, head[0])
    beliefs = {"oracle": {"a": Neighbours.of(Belief.oracle(labels), index)}}
    made = Fusion.of(graded, beliefs, {"a": torch.arange(64)})
    result = Study(made, made).run(table(made.wanted()), 0)
    assert set(result) == {f"oracle/{channel}" for channel in CHANNELS}


def test_fitted() -> None:
    made = built()
    finish = Study(made, made).fitted(table(made.wanted()), "pmi")
    assert finish.gamma == 0.0
    assert finish.mix >= 0.0


def test_fits() -> None:
    made = built()
    fits = Study(made, made).fits(table(made.wanted()))
    assert set(fits) == set(CHANNELS)
    assert all(set(fit) == {"tau", "lam"} for fit in fits.values())


def test_shared() -> None:
    rows = Posteriors(
        torch.zeros(4, VOCAB_SIZE),
        torch.zeros(4, VOCAB_SIZE),
        torch.tensor([1, IGNORE, 2, IGNORE]),
        torch.zeros(1, 4, dtype=torch.long),
    )
    assert Dumped.shared({"a": rows}) == pytest.approx(0.5)


def kaggle() -> Kaggle:
    per = [neighbours().channels(row) for row in range(4)]
    return Kaggle({channel: [row[channel] for row in per] for channel in CHANNELS})


def test_contexts() -> None:
    made = kaggle()
    assert made.contexts() == [key for rows in made.weighted.values() for row in rows for key in row.contexts]
    assert EMPTY in made.contexts()


def test_others() -> None:
    made = kaggle()
    others = made.others(table(made.contexts()))
    assert {channel: other.shape for channel, other in others.items()} == dict.fromkeys(CHANNELS, (4, VOCAB_SIZE))


def test_lone() -> None:
    made = kaggle()
    assert made.lone(made.others(table(made.contexts()))) == pytest.approx(0.0)


def test_aligned() -> None:
    head = torch.zeros(3, VOCAB_SIZE)
    head[:, 2] = 1.0
    rows = Posteriors(head, head, torch.tensor([IGNORE, 1, 2]), torch.zeros(1, 3, dtype=torch.long))
    graded = Rows(head[1:], head[1:], torch.tensor([1, 2]))
    assert Study.aligned(rows, graded, torch.tensor([1, 2])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="labels differ"):
        Study.aligned(rows, graded, torch.tensor([0, 1]))
