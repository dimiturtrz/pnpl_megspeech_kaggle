"""The acoustic auxiliary: each training word's window, read through its encoder's per-timestep features, predicts the
50 Hz speech envelope heard over the window, and chunks of the prediction are contrasted against the batch's true
chunks (Défossez et al. 2023's segment-level CLIP loss over speech, not a regression)."""

from dataclasses import dataclass
from typing import ClassVar, cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from megspeech.objective.auxiliary.semantic import Semantic
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.term import Reading

CHUNK = 25
"""Frames of one contrasted chunk: 0.5 s, a few syllables, so a chunk's shape identifies it among the batch's."""
MIX = 16
"""Channels the learned readout mixes the encoder's features into before the lag."""
TAPS = 13
"""Frames the lag reads ahead of each envelope frame: the envelope at time `t` shows in the MEG up to 0.24 s later."""
PRIOR = 6
"""The tap the lag starts on: 0.12 s, the 100-150 ms cortical lag of the envelope response."""


class Acoustic(Semantic):
    """A learned channel mix of a window's `[width t]` features, resampled to the envelope's `frames` and read
    through a learned lag, scored in `CHUNK`-frame chunks."""

    def __init__(self, width: int, frames: int) -> None:
        if frames % CHUNK:
            raise ValueError(f"an envelope window of {frames} frames is no whole number of {CHUNK}-frame chunks")
        super().__init__(width, MIX)
        self.frames = frames
        self.lag = nn.Conv1d(MIX, 1, TAPS, bias=False)
        nn.init.zeros_(self.lag.weight)
        with torch.no_grad():
            self.lag.weight[:, :, PRIOR] = MIX**-0.5

    @override
    def forward(self, embedded: Float[Tensor, "*b w t"]) -> Float[Tensor, "*b f"]:
        """The predicted envelope of every window of `embedded`, `[*b width t]`."""
        times = embedded.flatten(0, -3)
        mixed = self.project(times.transpose(1, 2)).transpose(1, 2)
        fitted = functional.interpolate(mixed, size=self.frames, mode="linear")
        lagged = self.lag(functional.pad(fitted, (0, TAPS - 1)))
        return lagged.squeeze(1).unflatten(0, embedded.shape[:-2])


@dataclass(frozen=True)
class AcousticTerm(SpeechTerm):
    """Every training word whose window has an envelope (`Speech` over `Loudness`' targets) trained toward it: each
    window's chunks are each their own kind in the batch's SigLIP contrast; the head spans the targets' frames."""

    timed: ClassVar[bool] = True

    @override
    def head(self, width: int) -> nn.Module:
        return Acoustic(width, self.speech.vectors.shape[-1]).to(self.speech.vectors.device)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        kinds = reading.target.flatten()
        truth = self.speech.vectors[kinds.clamp(min=0)]
        chunks = truth.shape[-1] // CHUNK
        chunk = torch.arange(chunks, device=kinds.device)
        chunked = torch.where(kinds[:, None] >= 0, kinds[:, None] * chunks + chunk, -1)
        return cast("Acoustic", reading.head).loss(
            reading.output.flatten(0, 1).reshape(-1, CHUNK),
            truth.reshape(-1, CHUNK),
            chunked.flatten(),
            reading.rows.flatten().repeat_interleave(chunks),
        )
