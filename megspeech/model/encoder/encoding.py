"""The stage-1 modules: everything between a window and its word embedding."""

from typing import Self, override

from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.model.config import ContextConfig
from megspeech.model.encoder.builder import WordBuilder
from megspeech.model.encoder.motion import Moving
from megspeech.model.input.stem import Stages, Stem
from megspeech.model.input.subjects import Unseen
from megspeech.model.words.base import WordEncoder

ENCODER = ("scale", "whiten", "subjects", "words")
"""The stage-1 modules: everything between a window and its word embedding, and the word head the CNN owns."""


class Encoding(nn.Module):
    """A window's path to its word embedding: subject levels, scaler, whitener, subject layer and word encoder."""

    words: WordEncoder

    def __init__(self, scale: nn.Module, context: ContextConfig, channels: int, unseen: Unseen) -> None:
        super().__init__()
        self.levels, self.scale, self.whiten, self.subjects = Stem.of(
            scale, channels, Stages(context.levels, context.whiten, context.subjects, context.subject_drop), unseen
        )
        self.words = WordBuilder.build(context.encoder, context, channels)
        self.motion = Moving()

    @override
    def train(self, mode: bool = True) -> Self:
        """`mode` set throughout, but for whatever of the stage `motion` holds."""
        super().train(mode)
        self.motion.hold(self.stage())
        return self

    def move(self, motion: Moving) -> None:
        """The stage held by `motion` from now on, through every later mode change."""
        self.motion = motion
        self.train(self.training)

    def embed(
        self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"], valid: Bool[Tensor, " r"] | None = None
    ) -> Float[Tensor, "r e"]:
        """Each window's word embedding through the scaler, the subject layer and the encoder; `valid` rows alone
        set any batch statistics."""
        return self.words.embed(self.prepare(x, subject), valid)

    def prepare(self, x: Float[Tensor, "r c t"], subject: Int[Tensor, " r"]) -> Float[Tensor, "r c t"]:
        """Each window as the word encoder reads it: levelled, scaled, whitened, in its subject's space, normalised."""
        sensors = self.subjects(self.whiten(self.scale(self.levels(x, subject))), subject)
        return self.words.normalise(sensors, x.ne(0).any(dim=1, keepdim=True))

    def encoder(self) -> list[nn.Parameter]:
        """The stage-1 parameters, every one a word's embedding and word-head logits depend on: the subject layer, the
        word CNN and its head, or for an encoder over a pretrained trunk (`WordEncoder.backbone`) the trunk alone, its
        read layers training with the span transformer and context head, the rest."""
        return [p for module in self.stage() for p in module.parameters()]

    def stage(self) -> tuple[nn.Module, ...]:
        """The modules `encoder`'s parameters are of."""
        stem = tuple(getattr(self, name) for name in ENCODER if name != "words")
        return (*stem, *self.words.core())

    def adopt(self, state: dict[str, Tensor]) -> None:
        """The stage-1 entries of another run's `state` loaded in, every other weight left as it is; a subject layer
        the state has none of (a single-subject stem) stays at its identity initialisation."""
        own = {key: value for key, value in state.items() if key.split(".")[0] in ENCODER}
        missing, _ = self.load_state_dict(own, strict=False)
        exempt = () if any(key.startswith("subjects.") for key in own) else ("subjects",)
        stale = [key for key in missing if key.split(".")[0] in ENCODER and key.split(".")[0] not in exempt]
        if stale:
            raise KeyError(f"stem state lacks encoder entries {stale}")
