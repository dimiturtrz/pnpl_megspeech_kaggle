"""Everything a training run is made of, and the record a finished run leaves of it."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from core.tree import Tree
from megspeech.data.corpus import METRICS, DataConfig
from megspeech.evaluation.report import MIN_NOVEL, Epoch
from megspeech.model.onsets import CONTEXT, ContextConfig
from megspeech.training.augment import Augment


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
    """Weight of the semantic auxiliary (`megspeech.training.auxiliary`): every heard word's window, the ~60% outside
    the 50 included, SigLIP-aligned to the word's t5-large layer-12 embedding, d'Ascoli et al.'s objective. It
    supervises ~2.5x the windows the heads see; 0 trains the heads alone and never reads the lexicon."""
    phonetic: float = 0.0
    """Weight of the phonetic auxiliary: every heard word's window, in or out of the 50, classified into the first of
    its 39 ARPAbet phonemes (`megspeech.data.phonemes`). Speech MEG decodes phonetic features (Gwilliams et al.
    2022), and the onset phoneme is what the window's first ~100 ms hear; 0 never reads the table."""
    positions: int = 1
    """Leading phonemes the phonetic auxiliary classifies, up to `megspeech.data.phonemes.POSITIONS`: a 0.5 s window
    hears ~4 of ~0.1 s, and 1 is the onset phoneme alone."""
    balance: float = 0.0
    """Class-weight exponent of the cross-entropies: each class weighs `count ** -balance` over the training labels,
    mean 1, so 0.5 is 1/sqrt(count) (the PNPL 2025 phoneme winner's). The metric is balanced accuracy, and the
    ruler's `tau` already corrects the class prior after training (Menon et al. 2021 logit adjustment); balancing
    during training overlaps it, so an arm asks whether it adds beyond `tau`. SigLIP's context loss is unweighted."""
    curriculum: float = 0.0
    """The Deep subject's share of the spans drawn at the last epoch, annealed linearly from an equal share per
    subject at the first (`megspeech.training.curriculum.Annealed`); 0 draws every subject's spans at their own
    weight. Only meaningful when `DataConfig.subjects` trains on more than the Deep subject."""
    compiled: bool = True
    """Compile the model's forward and backward into CUDA graphs; batches have fixed shapes, so they replay."""


@dataclass(frozen=True)
class SelectionConfig:
    min_novel: int = MIN_NOVEL
    """Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""


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
