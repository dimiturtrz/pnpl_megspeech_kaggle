"""Per-subject robust levels (megspeech-7gf.147.4): each training subject's sensors moved onto the pooled median and
interquartile range before the pooled scaler, so subjects recorded at different gains and noise floors reach the
shared encoder alike. Fit on the training split alone; a subject it never saw is read through the pooled scaler."""

from collections.abc import Callable
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.raw.smn4lang import SUBJECT_OFFSET
from megspeech.model.input.scaler import SCALER_SAMPLE, Scaler

BLOCK = 100
"""Subject ids each pooled corpus owns from its `SUBJECT_OFFSET` (MASC 100, Appleseed 200, Kymata 300, SCANS 400)."""
KNOWN = SUBJECT_OFFSET + BLOCK
"""Subjects with a row: past the last corpus's block (SMN4LANG at 500), so every id a training split can hold. A
Kaggle zero-shot subject (33-39) has a row it never fits: the identity."""


class Levels(nn.Module):
    """Each subject's sensors as `x * gain + offset`, the affine map taking its own training median and interquartile
    range to the pooled scaler's, so the scaler after it reads each subject in its own interquartile ranges. A
    subject with no training window, and every subject past `subjects` rows, keeps the identity row: the pooled
    scaler alone, as before."""

    gain: Tensor
    offset: Tensor

    def __init__(self, channels: int, *, active: bool = False, subjects: int = KNOWN) -> None:
        """An inactive stage passes the sensors through and saves nothing, so models trained without it load."""
        super().__init__()
        self.active = active
        self.register_buffer("gain", torch.ones(subjects + 1, channels, 1), persistent=active)
        self.register_buffer("offset", torch.zeros(subjects + 1, channels, 1), persistent=active)

    def fit(
        self,
        windows: Callable[[Int[Tensor, " k"]], Float[Tensor, "k c t"]],
        subject: Int[Tensor, " n"],
        pooled: Scaler,
    ) -> None:
        """Each subject's row from up to `SCALER_SAMPLE` of its training `windows` (the words heard by `subject`),
        mapped onto `pooled`; a channel flat in a subject's own windows carries nothing of it and reads the pooled
        median."""
        if not self.active:
            return
        self.to(pooled.median.device)
        present = subject.unique()
        for heard in present[present < len(self.gain) - 1].tolist():
            rows = torch.nonzero(subject == heard).flatten()
            median, spread = Scaler.quartiles(
                windows(rows[torch.randperm(len(rows), device=rows.device)[:SCALER_SAMPLE]])
            )
            gain = torch.where(spread > 0, pooled.spread / spread.clamp(min=torch.finfo(spread.dtype).tiny), 0.0)
            self.gain[heard] = gain
            self.offset[heard] = pooled.median - median * gain

    @override
    def forward(self, x: Float[Tensor, "b c t"], subject: Int[Tensor, " b"]) -> Float[Tensor, "b c t"]:
        return self.mapped(x, subject.clamp(max=len(self.gain) - 1)) if self.active else x

    def mapped(self, x: Float[Tensor, "b c t"], rows: Int[Tensor, " b"]) -> Float[Tensor, "b c t"]:
        """`x` through the rows' affine maps."""
        return x.float() * self.gain[rows] + self.offset[rows]
