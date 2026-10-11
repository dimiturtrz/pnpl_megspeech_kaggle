"""The audio contrastive term: each heard word's window, read through its encoder's per-timestep features, and the
50 Hz speech envelope heard over the same second, embedded by two small towers into one space and aligned by CLIP's
symmetric InfoNCE (Radford et al. 2021; Défossez et al. 2023 over MEG and speech). It reads no word label, only the
onset both windows are cut at, so every language whose audio is aligned trains it
(`megspeech.objective.targets.loudness`, `megspeech.objective.targets.scansloudness`). With `--optim.heads 0` it is a
label-free encoder pretrain whose run is an `--optim.stem` (megspeech-7gf.109.3).

A window's negatives are the batch's windows outside its own span: a span's words are ~0.25 s apart, so their 1 s
windows share most of their audio, and contrasting them would teach the encoder to tell one sound from itself. A
batch of `OptimConfig.batch` 64 sentence spans holds ~10^3 windows drawn across subjects and languages, so the n x n
logits take a few MB beside the encoder's activations, which the supervised runs already fit at that batch in ~14 GB;
InfoNCE bounds the mutual information it can show by log n, ~7 nats there against the 5.5 of Défossez et al.'s
256-segment batches."""

import math
from dataclasses import dataclass
from typing import ClassVar, cast, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.objective.auxiliary.acoustic import MIX, TAPS
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.targets.loudness import LENGTH
from megspeech.objective.term import AuxiliaryTerm, Reading

DIM = 256
"""Width of the shared space: ~5x the envelope window's `LENGTH` frames, so the audio tower can keep every frame."""
TEMPERATURE = 0.07
SCALE_CAP = 100.0
"""CLIP's initial temperature and the ceiling its learned logit scale is clipped to."""


class Listening(nn.Module):
    """The MEG tower (a learned channel mix of a window's `[width t]` features, resampled to the envelope's `LENGTH`
    frames, filtered over `TAPS` frames and projected), the audio tower (the envelope filtered and projected alike),
    and the learned logit scale of their cosine."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.mix = nn.Conv1d(width, MIX, 1)
        self.meg = self.tower(MIX)
        self.audio = self.tower(1)
        self.scale = nn.Parameter(torch.tensor(math.log(1 / TEMPERATURE)))

    @staticmethod
    def tower(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(channels, MIX, TAPS, padding=TAPS // 2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(MIX * LENGTH, DIM),
        )

    @override
    def forward(self, embedded: Float[Tensor, "*b w t"]) -> Float[Tensor, "*b d"]:
        """The MEG embedding of every window of `embedded`, `[*b width t]`."""
        times = embedded.flatten(0, -3)
        fitted = functional.interpolate(self.mix(times), size=LENGTH, mode="linear")
        return self.meg(fitted).unflatten(0, embedded.shape[:-2])

    def hear(self, envelope: Float[Tensor, "n f"]) -> Float[Tensor, "n d"]:
        """The audio embedding of each `LENGTH`-frame envelope window."""
        return self.audio(envelope.float()[:, None])

    def loss(
        self,
        meg: Float[Tensor, "n d"],
        audio: Float[Tensor, "n d"],
        kinds: Int[Tensor, " n"],
        spans: Int[Tensor, " n"],
        rows: Float[Tensor, " n"],
    ) -> Float[Tensor, ""]:
        """The mean of both directions' InfoNCE: each row's MEG against every candidate's audio, and each audio
        against every candidate's MEG, its own pair the label. A negative kind (padding, a word without audio) is
        neither a row nor a candidate; a window of the row's own span or of its own kind is no negative. Fixed
        shapes, no host sync; rows average at their weights in `rows`."""
        valid = kinds >= 0
        own = torch.eye(len(kinds), dtype=torch.bool, device=kinds.device)
        related = (spans[:, None] == spans[None, :]) | (kinds[:, None] == kinds[None, :])
        allowed = (own | ~related) & valid[:, None] & valid[None, :]
        cosine = functional.normalize(meg.float(), dim=-1) @ functional.normalize(audio.float(), dim=-1).T
        logits = (self.scale.exp().clamp(max=SCALE_CAP) * cosine).masked_fill(~allowed, torch.finfo(cosine.dtype).min)
        listened = -logits.log_softmax(-1).diagonal()
        heard = -logits.log_softmax(0).diagonal()
        return 0.5 * (AuxiliaryTerm.averaged(listened, valid, rows) + AuxiliaryTerm.averaged(heard, valid, rows))


@dataclass(frozen=True)
class AudioTerm(SpeechTerm):
    """Every training word whose window has an envelope (`Speech` over `Loudness`' targets) aligned to it by
    `Listening`'s contrast; the rest are left out."""

    timed: ClassVar[bool] = True

    @override
    def head(self, width: int) -> nn.Module:
        return Listening(width).to(self.speech.vectors.device)

    @override
    def unweighted(self, reading: Reading) -> Float[Tensor, ""]:
        listening = cast("Listening", reading.head)
        target = reading.target
        kinds = target.flatten()
        spans = torch.arange(target.shape[0], device=target.device).repeat_interleave(target.shape[1])
        audio = listening.hear(self.speech.vectors[kinds.clamp(min=0)])
        return listening.loss(reading.output.flatten(0, 1), audio, kinds, spans, reading.rows.flatten())
