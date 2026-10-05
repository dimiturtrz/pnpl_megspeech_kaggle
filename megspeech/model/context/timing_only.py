from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.model.context.timing import TimingHead


class TimingOnly(nn.Module):
    """A timing expert with no MEG: a `TimingHead` read as the context and the word head alike, so a trainer and an
    evaluator that drive a sentence model drive it too (megspeech-7gf.100)."""

    def __init__(self, vocabulary: int) -> None:
        super().__init__()
        self.timing = TimingHead(vocabulary)

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        logits = self.timing(onset, slot < len(x))
        return logits, logits
