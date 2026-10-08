import torch

from megspeech.model.context.masking import WordMask

GENERATOR = torch.Generator().manual_seed(0)
VALID = torch.tensor([[True, True, False], [True, False, False]])


def test_forward() -> None:
    """Rate 1 in training swaps every valid word's token for the mask embedding, padding stays; rate 0 and
    evaluation return the input itself."""
    content = torch.randn(2, 3, 8, generator=GENERATOR)
    full = WordMask(8, 1.0).train()
    out = full.forward(content, VALID)
    assert torch.equal(out[VALID], full.token.expand(int(VALID.sum()), 8))
    assert torch.equal(out[~VALID], content[~VALID])
    assert full.eval().forward(content, VALID) is content
    assert WordMask(8, 0.0).train().forward(content, VALID) is content


def test_rate() -> None:
    mask = WordMask(4, 0.3).train()
    valid = torch.ones(64, 16, dtype=torch.bool)
    hit = (mask.forward(torch.zeros(64, 16, 4), valid) != 0).any(dim=2).float().mean()
    assert abs(hit - 0.3) < 0.05
