import torch
from transformers import GPT2Config, GPT2LMHeadModel

from megspeech.evaluation.studies.lmscore import LanguageModel

BOS = 0
SIZE = 40


def chars(texts: list[str]) -> list[list[int]]:
    """One token per character, so a spaced word is `1 + len(word)` tokens."""
    return [[1 + ord(c) % (SIZE - 1) for c in text] for text in texts]


def lm() -> LanguageModel:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=SIZE, n_positions=64, n_embd=16, n_layer=2, n_head=2)
    return LanguageModel(GPT2LMHeadModel(config).eval(), chars, BOS)


def brute(model: LanguageModel, context: list[str], candidate: str, suffix: list[str], upto: int) -> float:
    """The log-probability of the candidate and the first `upto` suffix words, one uncached pass."""
    words = [candidate, *suffix[:upto]]
    ids = model.opening(context) + [t for piece in model.pieces(words) for t in piece]
    with torch.no_grad():
        logp = model.model(torch.tensor([ids])).logits.float().log_softmax(-1)[0]
    start = len(model.opening(context))
    return float(sum(logp[i - 1, ids[i]] for i in range(start, len(ids))))


def test_device() -> None:
    assert lm().device.type == "cpu"


def test_pieces() -> None:
    assert [len(p) for p in lm().pieces(["ab", "c"])] == [3, 2]
    assert lm().pieces([]) == []


def test_opening() -> None:
    assert lm().opening([]) == [BOS]
    assert len(lm().opening(["ab"])) == 1 + 3


def test_score() -> None:
    model, context, suffix = lm(), ["ab", "c"], ["de", "f"]
    candidates = ["x", "yz"]
    scores = model.score(context, candidates, suffix)
    assert scores.shape == (2, 3)
    for row, candidate in enumerate(candidates):
        for upto in range(3):
            expected = brute(model, context, candidate, suffix, upto)
            assert abs(float(scores[row, upto]) - expected) < 1e-3


def test_score_without_context_or_suffix() -> None:
    model = lm()
    scores = model.score([], ["x", "yz"])
    assert scores.shape == (2, 1)
    assert abs(float(scores[1, 0]) - brute(model, [], "yz", [], 0)) < 1e-3


def test_cumulative() -> None:
    first = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), -1)
    after = torch.log_softmax(torch.randn(1, 2, 3), -1)
    ids = torch.tensor([[2, 1]])
    out = LanguageModel.cumulative(first, after, ids, torch.tensor([[1, 2]]))
    assert abs(float(out[0, 0]) - float(first[2])) < 1e-6
    assert abs(float(out[0, 1]) - float(first[2] + after[0, 0, 1])) < 1e-6
