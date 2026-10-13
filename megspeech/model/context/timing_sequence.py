"""A timing expert that reads the whole span: every word's onset features attended across its sentence, no MEG."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.context.timing import FEATURES, TimingHead

WIDTH = 128
HEADS = 4


class TimingSequence(nn.Module):
    """`TimingHead`'s per-word features through a transformer over the span, so each word's logits read every gap in
    its sentence rather than two neighbours each side (megspeech-r9m: the local head saturates at a quarter of the
    data)."""

    def __init__(self, vocabulary: int, layers: int) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 2 * WIDTH, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocabulary))

    @override
    def forward(self, onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w v"]:
        """Each position's timing logits; padding neither attends nor is attended to."""
        hidden = self.encoder(self.embed(TimingHead.features(onset, mask)), src_key_padding_mask=~mask)
        return self.out(hidden)
