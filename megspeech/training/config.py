"""Everything a training run is made of, and the record a finished run leaves of it."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from core.tree import Tree
from megspeech.data.config import METRICS, DataConfig
from megspeech.evaluation.scoring.score import Epoch, SelectionConfig
from megspeech.model.auxiliary import Tap
from megspeech.model.config import CONTEXT, ContextConfig
from megspeech.training.regularise.augment import Augment


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    clip: float = math.inf
    seed: int = 0
    ema: float = 0.0
    """Decay of the weight average scored and kept; 0 keeps the raw weights. At 0.999 the average spans ~1000
    steps, about half an epoch of the ~1750 a sentence-path epoch takes."""
    semantic: float = 0.0
    """Weight of the semantic auxiliary (`megspeech.objective.auxiliary.semantic`): every heard word's window, the
    ~60% outside the 50 included, SigLIP-aligned to the word's t5-large layer-12 embedding, d'Ascoli et al.'s
    objective. It supervises ~2.5x the windows the heads see; 0 trains the heads alone and never reads the lexicon."""
    lexical: float = 0.0
    """Weight of the lexical auxiliary: every heard word's window, in or out of the 50, classified into its word among
    the ~25k distinct words of the lexicon (`megspeech.objective.auxiliary.lexical`). Word identity, where the
    semantic auxiliary aligns to a frozen embedding; 0 never reads the lexicon."""
    phonetic: float = 0.0
    """Weight of the phonetic auxiliary: every heard word's window, in or out of the 50, classified into the first of
    its 39 ARPAbet phonemes (`megspeech.data.labels.phonemes`). Speech MEG decodes phonetic features (Gwilliams et al.
    2022), and the onset phoneme is what the window's first ~100 ms hear; 0 never reads the table."""
    speech: float = 0.0
    """Weight of the speech auxiliary (`megspeech.objective.auxiliary.speech`): every training word whose chapter audio
    is aligned, SigLIP-aligned to the wav2vec2 latent of its 0.5 s of heard speech (Défossez et al. 2023), one target
    per word token, not per word type as the semantic auxiliary's; 0 never reads the targets."""
    speech_centred: bool = False
    """Centre each run's speech targets on its own mean (`Targets.centred`), removing the direction every target
    shares: raw, two random targets' cosine is ~0.6, so the L2-normalised SigLIP target is mostly that one direction."""
    tap: Tap = Tap.WORD
    """What every auxiliary head reads: the encoder's word embedding, or the span transformer's output at the word
    (d'Ascoli et al. train that output of every heard word toward t5; the context head alone sees only the 50's)."""
    positions: int = 1
    """Leading phonemes the phonetic auxiliary classifies, up to `megspeech.data.labels.phonemes.POSITIONS`: a 0.5 s
    window hears ~4 of ~0.1 s, and 1 is the onset phoneme alone."""
    balance: float = 0.0
    """Class-weight exponent of the cross-entropies: each class weighs `count ** -balance` over the training labels,
    mean 1, so 0.5 is 1/sqrt(count) (the PNPL 2025 phoneme winner's). The metric is balanced accuracy, and the
    ruler's `tau` already corrects the class prior after training (Menon et al. 2021 logit adjustment); balancing
    during training overlaps it, so an arm asks whether it adds beyond `tau`. SigLIP's context loss is unweighted."""
    curriculum: float = 0.0
    """The Deep subject's share of the spans drawn at the last epoch, annealed linearly from an equal share per
    subject at the first (`megspeech.training.sampling.curriculum`); 0 draws every subject's spans at their own
    weight. Only meaningful when `DataConfig.subjects` trains on more than the Deep subject."""
    anchor: float = 1.0
    """Share of each subject matrix's deviation from the Deep subject's that the run's decay leaves by its end, were
    its gradient never to restore it (`megspeech.training.regularise.anchor`); 1 decays the matrices toward 0 like
    every other weight. The Deep matrix is what a zero-shot subject is read through."""
    warm: str = ""
    """A trained run under models/ whose weights the model starts from instead of its initialisation, its context
    config the same as this run's: a run trained on every subject fine-tuned on the Deep subject alone. Empty starts
    from scratch."""
    teacher: str = ""
    """A trained run or ensemble under models/ whose context-head distribution on every labelled training word the
    context head is also trained toward (`megspeech.training.teacher`); empty trains on the labels alone."""
    distil: float = 0.5
    """Share of the context loss that is the KL to the `teacher`'s distribution, the rest the label's; read only
    with a teacher."""
    temperature: float = 1.0
    """Temperature both the teacher's and the student's distributions are softened by in the KL; read only with a
    teacher."""
    unlabelled: bool = False
    """Distil on every training word in a span, the ~60% outside the 50 too, not only the labelled; read only with a
    teacher."""
    compiled: bool = True
    """Compile the model's forward and backward into CUDA graphs; batches have fixed shapes, so they replay."""
    pretrained: float = 0.1
    """Retired with the end-to-end MEG-XL encoder it scaled; kept so the configs saved with it still load."""


@dataclass(frozen=True)
class RunConfig:
    """Everything a training run is made of; `core.tree.Tree` turns it into the command line."""

    data: DataConfig = DataConfig()
    context: ContextConfig = CONTEXT
    augment: Augment = Augment()
    optim: OptimConfig = OptimConfig()
    selection: SelectionConfig = SelectionConfig()

    def selected(self, history: list[Epoch]) -> Epoch:
        """The epoch whose weights the run keeps: the best on validation, or the last when validation was trained on."""
        return history[-1] if self.data.refit else max(history, key=lambda row: row.macro.bacc10)

    @classmethod
    def saved(cls, directory: Path) -> "RunConfig":
        """The config a run in `directory` recorded, flat records from before the tree included."""
        return Tree(cls).load(json.loads((directory / METRICS).read_text())["config"])
