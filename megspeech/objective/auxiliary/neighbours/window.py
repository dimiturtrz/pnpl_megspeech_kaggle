"""The window auxiliary: each heard word's window predicts the embeddings of the words around it, each weighted by
where its response sits in the window. It generalises the neighbour auxiliaries (`NextWordTerm`, `PreviousWordTerm`):
the same in-span shift (`NeighbourTerm.beside`), over five offsets and the t5-large embedding rather than the 50-way
label."""

from dataclasses import dataclass, replace
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.objective.auxiliary.aligned.semantic import Semantic
from megspeech.objective.auxiliary.neighbours.neighbour import NeighbourTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import AuxiliaryTerm, Reading

WORDS_PER_SECOND = 4.0
"""Speech rate of the narratives: neighbour `k` begins `k / WORDS_PER_SECOND` seconds after the word."""
WINDOW_SECONDS = 1.0
"""The word window's length, read from the word's onset."""
RESPONSE_START = 0.2
"""Seconds after a word's onset its evoked response begins: recognition past the acoustic stage."""
RESPONSE_END = 0.5
"""Seconds after a word's onset its response ends (the N400 peaks ~0.4 s)."""
BEFORE = 1
"""Words before the word that the window predicts: its response still overlaps the window's start."""
AFTER = 3
"""Words after the word that the window predicts: at 4 words/s the window reaches ~3 onsets ahead."""
OFFSETS = BEFORE + AFTER + 1
"""Words the window predicts: those at offsets -`BEFORE` to `AFTER` around the word itself."""


class Window(Semantic):
    """A projection of word embeddings into one t5-large space per neighbour offset."""

    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.project = nn.Linear(width, OFFSETS * DIM)

    @override
    def forward(self, embedded: Float[Tensor, "*b w"]) -> Float[Tensor, "*b k d"]:
        return self.project(embedded).unflatten(-1, (OFFSETS, DIM))


@dataclass(frozen=True)
class WindowTerm(AuxiliaryTerm):
    """Every heard word's window trained toward the t5-large embeddings of the words at offsets -1 to +3 of its
    span, each offset's loss weighted by the share of its response the window holds (`lag_weights`)."""

    weight: float
    lexicon: Lexicon

    @staticmethod
    def lag_weights() -> Float[Tensor, " k"]:
        """Per offset, the share of the response band (`RESPONSE_START` to `RESPONSE_END` after that word's onset)
        inside the window: offset 0 and the two after it hold all of it, -1 and +3 a sliver, normalised to sum 1."""
        onsets = torch.arange(-BEFORE, AFTER + 1) / WORDS_PER_SECOND
        start = (onsets + RESPONSE_START).clamp(0, WINDOW_SECONDS)
        end = (onsets + RESPONSE_END).clamp(0, WINDOW_SECONDS)
        held = (end - start) / (RESPONSE_END - RESPONSE_START)
        return held / held.sum()

    @override
    def head(self, width: int) -> nn.Module:
        return Window(width).to(self.lexicon.vectors.device)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        window = cast("Window", reading.head)
        output, target = reading.output, reading.target
        weights = self.lag_weights().to(output.device)
        total = output.new_zeros(())
        for at, lag in enumerate(range(-BEFORE, AFTER + 1)):
            kinds = torch.where(target == IGNORE, -1, self.lexicon.rows(NeighbourTerm.beside(target, lag)))
            lagged = replace(reading, output=output[:, :, at])
            total = total + weights[at] * lagged.contrasted(window.loss, self.lexicon.vectors, kinds)
        return total
