from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput
from megspeech.model.context.timing import TimingHead
from megspeech.model.context.timing_sequence import TimingSequence

FILE = "timing.pt"
"""The saved timing head's weights, under its run's directory."""


class TimingOnly(nn.Module):
    """A timing expert with no MEG: a `TimingHead` read as the context and the word head alike, so a trainer and an
    evaluator that drive a sentence model drive it too (megspeech-7gf.100). With `layers`, a `TimingSequence` reads the
    whole span instead."""

    def __init__(self, vocabulary: int, layers: int = 0) -> None:
        super().__init__()
        self.timing = TimingSequence(vocabulary, layers) if layers else TimingHead(vocabulary)

    @override
    def forward(self, spans: SpanInput) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        logits = self.timing(spans.onset, spans.mask)
        return logits, logits
