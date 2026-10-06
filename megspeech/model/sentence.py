"""The sentence-context model: per-word embeddings, onset timing, and a transformer across the span."""

from pathlib import Path
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import CONTEXT, ContextConfig, OnsetEncoding
from megspeech.model.context.masking import WordMask
from megspeech.model.context.timing import Jitter, TimingHead
from megspeech.model.input.stem import Stem
from megspeech.model.input.subjects import Unseen
from megspeech.model.mixer import Mixer
from megspeech.model.words.layers.mean_max import MeanMax
from megspeech.objective.anchored import AnchoredHead
from megspeech.objective.scorer import DIM

LATENCY_INIT = 0.02
"""Spread of each latency tag at initialisation, a transformer's usual position-embedding scale (BERT, GPT-2)."""

SCALE_MEDIAN = "scale.median"
"""A SentenceNet state's per-channel scaler median: one entry per channel the model reads."""

ENCODER = ("scale", "whiten", "subjects", "words")
"""The stage-1 modules: everything between a window and its word embedding, and the word head the CNN owns."""


class SentenceNet(nn.Module):
    """Per-word embeddings refined by a transformer over the span of words heard together.

    Returns the context head's logits and the word head's, both per word slot, so one model is both the
    isolated-word classifier and the sentence-context one.
    """

    latency: Tensor

    def __init__(
        self,
        scale: nn.Module,
        context: ContextConfig = CONTEXT,
        prototypes: Float[Tensor, "v d"] | None = None,
        unseen: Unseen = Unseen.IDENTITY,
        *,
        channels: int = CHANNELS,
    ) -> None:
        """`unseen` reads a subject the subject layer has no matrix for (`SubjectLayer`); `channels` are the rows a
        recording reaches the model in: the sensors, or a `Projection`'s components."""
        super().__init__()
        self.scale, self.whiten, self.subjects = Stem.of(
            scale, channels, whiten=context.whiten, subjects=context.subjects, unseen=unseen
        )
        self.words = context.encoder.build(context, channels)
        self.reading = context.conv.head
        self.reading.attach(self.words.head, context.width)
        self.tokens = context.tokens
        self.project = nn.Linear(self.words.width // self.tokens, context.width)
        self.clock = context.clock
        self.onsets = OnsetEncoding(context)
        self.intervals = context.intervals
        self.jitter = Jitter(context.onset_jitter)
        self.masking = WordMask(context.width, context.word_mask)
        self.context = Mixer.of(context)
        self.head = self.readout(context, prototypes)
        self.timing = TimingHead.over_vocabulary(active=context.rhythm)
        self._tag(context)

    @classmethod
    def load(cls, path: Path, unseen: Unseen = Unseen.IDENTITY) -> "SentenceNet":
        """The checkpoint at `path` in the shape its directory's config records; its state overwrites a placeholder
        scaler shaped as the state's own: MEG channels or subspace components, or a cached encoder's feature
        channels."""
        state = torch.load(path, map_location="cpu")
        channels = state[SCALE_MEDIAN].shape[0]
        model = cls(
            Stem.placeholder(channels), context=ContextConfig.load(path.parent), unseen=unseen, channels=channels
        )
        model.load_state_dict(state)
        return model

    @staticmethod
    def readout(context: ContextConfig, prototypes: Float[Tensor, "v d"] | None) -> nn.Module:
        """The context head over the fixed per-word `prototypes` (None stands in zeros for a checkpoint that will
        overwrite them): cosine to them as anchors when the config names a source, else the objective's own."""
        if not context.anchor.active:
            return context.objective.head(context.width, prototypes)
        placeholder = torch.zeros(VOCAB_SIZE, context.anchor_dim or DIM) if prototypes is None else prototypes
        return AnchoredHead(context.width, placeholder, trainable=context.anchor_train)

    def _tag(self, context: ContextConfig) -> None:
        """Each latency token's learned tag as `latency`; with one token per word a zero outside the state, so the
        checkpoints trained before it load, and their initialisation draws no more."""
        if context.latency:
            self.latency = nn.Parameter(torch.randn(self.tokens, context.width) * LATENCY_INIT)
        else:
            self.register_buffer("latency", torch.zeros(1, context.width), persistent=False)

    @override
    def forward(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """`slot` is each span position's row of `x`; `len(x)` marks an empty position, and rows no slot names are
        padding that neither normalises nor reaches the transformer."""
        _, _, context, word = self.heard(x, onset, slot, subject)
        return context, word

    def heard(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"], Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The refined positions and the word embeddings, then the context head's logits with the timing head's
        added, and the word head's."""
        mask = slot < len(x)
        h, slots = self.features(x, onset, slot, subject)
        refined = self.refine(h, mask)
        context, word = self.read(refined, slots)
        return refined, slots, context + self.timing(onset, mask), word

    def features(
        self,
        x: Float[Tensor, "r c t"],
        onset: Float[Tensor, "g w"],
        slot: Int[Tensor, "g w"],
        subject: Int[Tensor, " r"],
    ) -> tuple[Float[Tensor, "g w h"], Float[Tensor, "g w e"]]:
        """The transformer's input at each span position, and the position's word embedding (zeros when empty)."""
        rows = len(x)
        valid = torch.zeros(rows + 1, dtype=torch.bool, device=x.device).index_fill_(0, slot.flatten(), value=True)
        embedded = self.embed(x, subject, valid[:rows])
        slots = nn.functional.pad(embedded, (0, 0, 0, 1))[slot]
        mask = slot < rows
        onset = self.jitter.apply(onset, mask) if self.training else onset
        timing = self.onsets(self.clock.times(onset, mask), self.intervals.features(onset, mask))
        content = self.masking(self.project(MeanMax.segmented(slots, self.tokens)), mask)
        chunks = content + timing[..., None, :] + self.latency
        return chunks.flatten(1, 2), slots

    def embed(
        self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"], valid: Bool[Tensor, " r"] | None = None
    ) -> Float[Tensor, "r e"]:
        """Each window's word embedding through the scaler, the subject layer and the encoder; `valid` rows alone
        set any batch statistics."""
        return self.words.embed(self.prepare(x, subject), valid)

    def word_logits(self, embeddings: Float[Tensor, "r e"]) -> Float[Tensor, "r v"]:
        """The stage-1 word head on `embeddings`."""
        return self.words.head(embeddings)

    def prepare(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        """Each window as the word encoder reads it: scaled, whitened, in its subject's space, normalised."""
        sensors = self.subjects(self.whiten(self.scale(x)), subject)
        return self.words.normalise(sensors, x.ne(0).any(dim=1, keepdim=True))

    def classify(
        self, h: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over `h` refined across its span's `mask`ed positions, and the word head's."""
        return self.read(self.refine(h, mask), slots)

    def refine(self, h: Float[Tensor, "g w h"], mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w h"]:
        """`h`, each word's `tokens` in a row, refined across its span's `mask`ed words and averaged back to one per
        word: what the context head reads."""
        tokens = mask.repeat_interleave(self.tokens, dim=1)
        return self.context(h, src_key_padding_mask=~tokens).unflatten(1, (-1, self.tokens)).mean(2)

    def read(
        self, refined: Float[Tensor, "g w h"], slots: Float[Tensor, "g w e"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        """The context head's logits over the `refined` positions, and the word head's over their `slots`."""
        return self.head(refined), self.reading.read(self.words.head, slots, refined)

    @property
    def width(self) -> int:
        """Width of the refined positions the context head reads."""
        return self.project.out_features

    def encoder(self) -> list[nn.Parameter]:
        """The stage-1 parameters, every one a word's embedding and word-head logits depend on: the subject layer, the
        word CNN and its head; the span transformer and context head are the rest."""
        return [p for name in ENCODER for p in getattr(self, name).parameters()]

    def adopt(self, state: dict[str, Tensor]) -> None:
        """The stage-1 entries of another run's `state` loaded in, every other weight left as it is."""
        own = {key: value for key, value in state.items() if key.split(".")[0] in ENCODER}
        missing, _ = self.load_state_dict(own, strict=False)
        stale = [key for key in missing if key.split(".")[0] in ENCODER]
        if stale:
            raise KeyError(f"stem state lacks encoder entries {stale}")

    def readouts(self) -> tuple[nn.Module, ...]:
        """The modules reading each of `forward`'s outputs, in order: the context head, then the word head."""
        return self.head, self.words.head
