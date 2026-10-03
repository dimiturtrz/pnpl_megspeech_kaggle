import torch

from megspeech.model.words.layers.attention_pool import AttentionPool

GENERATOR = torch.Generator().manual_seed(0)


def test_attention_pool_forward() -> None:
    """Zero weights leave a score constant over time, so each query reads the mean."""
    pool = AttentionPool(8, 2)
    torch.nn.init.zeros_(pool.score.weight)
    h = torch.randn(3, 8, 50, generator=GENERATOR)
    assert torch.allclose(pool.forward(h), h.mean(-1).repeat(1, 2), atol=1e-6)
