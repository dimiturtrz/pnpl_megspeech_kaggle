"""The context model's shape, saved beside every trained model; the per-word encoder it reads and the onset
encoding it builds."""

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.tree import Retired, Tree
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.context.timing import Clock, Intervals
from megspeech.model.words.conv import CONV, ConvConfig
from megspeech.model.words.expert import Crop
from megspeech.model.words.spanned import SPANNED, SpannedConfig
from megspeech.objective.anchors import Source
from megspeech.objective.context import Objective


class Covariance(StrEnum):
    """The covariance `whiten` inverts. `TOTAL`: that of the training windows, evoked signal and noise together.
    `NOISE`: that of their residuals about each word's mean window, the noise alone (Engemann & Gramfort 2015)."""

    TOTAL = "total"
    NOISE = "noise"


class Encoder(StrEnum):
    """The CNN trained from scratch, or a head over the features `DataConfig.features` names: MEG-XL's pooled over
    the sensors or each channel's own, or a trained run's CNN embeddings."""

    CONV = "conv"
    CACHED = "cached"
    RESOLVED = "resolved"
    TRAINED = "trained"
    """A head over a trained run's own CNN embeddings (`megspeech.features.trained`), `conv` giving their width."""
    EXPERTS = "experts"
    """Trained runs' CNNs, each `conv`-shaped, reading their own crops of the window side by side and fine-tuning
    together (`words.experts`)."""

    SPANNED = "spanned"
    """MEG-XL fine-tuned over each span's whole continuous segment (`DataConfig.segment`), words read at their onsets
    (`words.spanned`)."""

    @property
    def cached(self) -> bool:
        """Whether the encoder reads features saved ahead of training rather than the MEG's sensors."""
        return self not in {Encoder.CONV, Encoder.EXPERTS, Encoder.SPANNED}


@dataclass(frozen=True)
class ContextConfig:
    RETIRED: ClassVar[dict[str, Retired]] = {
        "latency": Retired(
            default=False, reason="per-chunk latency tokens (latency) are retired, killed by megspeech-7gf.67"
        ),
    }

    width: int = 256
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1
    word_mask: float = 0.0
    """Probability, in training only, that a word's tokens enter the transformer as one learned mask embedding (its
    timing kept) and are predicted from its neighbours; 0 hides none (`context.masking.WordMask`)."""
    drop_path: float = 0.0
    """Stochastic depth (Huang et al. 2016): each sample's attention, feed-forward and mixing branches dropped in
    training at a rate rising linearly to this at the last layer; 0 drops none (`context.droppath.DropPath`)."""
    shortest: float = 0.04
    longest: float = 8.0
    """Onset periods span two 50 Hz samples up to a nine-word span at the Kaggle test's widest gap."""
    timescales: int = 16
    clock: Clock = Clock.ONSET
    intervals: Intervals = Intervals.NONE
    onset_jitter: float = 0.0
    """Seconds of Gaussian error added to every word onset the timing input reads, in training only (`Jitter`)."""
    """Added beside the clock: `log` tells each word its gaps to its neighbours' onsets."""
    objective: Objective = Objective.SOFTMAX
    anchor: Source = Source.NONE
    """Where the context head's fixed per-word anchors come from (`objective.anchors`): its logits are then a scaled
    cosine to them; none keeps the objective's own head."""
    anchor_train: bool = False
    """The anchors train from their initial values rather than staying frozen."""
    anchor_dim: int = 0
    """Principal components of the centred anchors kept (at most one fewer than the words they span); 0 keeps the
    anchors whole."""
    conv: ConvConfig = CONV
    """The per-word encoder the transformer reads."""
    encoder: Encoder = Encoder.CONV
    """Whether that encoder is the CNN `conv` shapes or a head over saved MEG-XL features, which reads only its
    `dropout`."""
    mixing: int = 0
    """Depthwise kernel, in words, of a convolution module after each transformer layer (`span.SpanConformer`); 0
    keeps the plain transformer. Kaggle test sentences run 1-9 words, so 3 mixes each word with its neighbours."""
    subjects: int = 0
    final_norm: bool = False
    """A LayerNorm after the pre-norm transformer's last layer, before the context head (plain transformer only)."""
    whiten: bool = False
    """Decorrelate the scaled sensors by their training covariance before the subject layer (`input.whitener`)."""
    covariance: Covariance = Covariance.TOTAL
    """Which covariance `whiten` inverts; the noise's is fit on training residuals (`training.regularise.noise`)."""
    rhythm: bool = False
    """A timing head's logits added to the context head's (`context.timing.TimingHead`); with `intervals` none, the
    transformer is told no gaps and timing reaches the words only through that head."""
    starts: tuple[int, ...] = ()
    """Samples into the window each `experts` encoder's crop starts at."""
    lengths: tuple[int, ...] = ()
    """Samples each `experts` encoder's crop runs, one per `starts`."""
    spanned: SpannedConfig = SPANNED
    """The `spanned` encoder's read of the trunk's tokens."""
    fused: tuple[str, ...] = ()
    """Saved runs under models/ whose word encoders read every window beside the model's own, their features
    projected and summed into the transformer's input (`model.fusion`, megspeech-7gf.131); each keeps the weights it
    was saved with, at the rate `OptimConfig.fused_lr` gives. Empty fuses none."""
    branch_drop: float = 0.3
    """Chance, in training, that a source of features (the model's own encoder or a fused one) is dropped from a span,
    the survivors rescaled so each source's expected weight is 1 and a span never losing every source; read only with
    `fused`. A frozen encoder's features fit the training words better than the validation's, and a head that
    sees them always learns to trust them."""

    def __post_init__(self) -> None:
        if self.anchor.active and self.objective is Objective.SIGLIP:
            raise ValueError("an anchored head and the siglip head are both the context head's scorer")
        if not 0 <= self.anchor_dim < VOCAB_SIZE or (self.anchor_dim and not self.anchor.active):
            raise ValueError(f"anchor_dim {self.anchor_dim} needs an anchor source and at most {VOCAB_SIZE - 1}")
        if len(self.starts) != len(self.lengths) or (self.encoder is Encoder.EXPERTS) != bool(self.starts):
            raise ValueError(
                f"the experts encoder, and it alone, reads one crop per expert: starts={self.starts}, "
                f"lengths={self.lengths} for encoder {self.encoder}"
            )
        if self.encoder.cached and self.subjects:
            raise ValueError(
                f"encoder {self.encoder} reads saved features, not sensors, so a per-subject sensor matrix "
                f"(subjects={self.subjects}) has nothing to mix"
            )
        if self.whiten and self.encoder.cached:
            raise ValueError(f"encoder {self.encoder} reads saved features, not sensors, so there is nothing to whiten")
        if not 0 <= self.branch_drop < 1:
            raise ValueError(f"branch_drop {self.branch_drop} is a probability below 1")
        if self.covariance is Covariance.NOISE and not self.whiten:
            raise ValueError("a noise covariance is only inverted by whiten")

    @property
    def crops(self) -> tuple[Crop, ...]:
        """The `experts` encoder's crops of the window, one per expert."""
        return tuple(Crop(start, length) for start, length in zip(self.starts, self.lengths, strict=True))

    def save(self, directory: Path) -> Path:
        path = directory / FILE
        path.write_text(json.dumps(asdict(self)))
        return path

    @classmethod
    def load(cls, directory: Path) -> "ContextConfig":
        """The shape a model in `directory` was trained with; fields recorded before they existed keep the default."""
        path = directory / FILE
        return Tree(cls).load(json.loads(path.read_text())) if path.exists() else CONTEXT


FILE = "context.json"
CONTEXT = ContextConfig()


class OnsetEncoding(nn.Module):
    """Each word's onset (seconds from the span's first word) as sinusoids over geometric periods."""

    frequency: Tensor

    def __init__(self, config: ContextConfig) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(config.shortest), math.log10(config.longest), config.timescales)
        self.register_buffer("frequency", 2 * math.pi / periods)
        self.project = nn.Linear(2 * config.timescales + config.intervals.width, config.width)

    @override
    def forward(self, onset: Float[Tensor, "g w"], intervals: Float[Tensor, "g w k"]) -> Float[Tensor, "g w d"]:
        """`intervals` are read beside the onset's sinusoids (`Intervals.features`)."""
        phase = onset[..., None] * self.frequency
        return self.project(torch.cat([phase.sin(), phase.cos(), intervals], dim=-1))
