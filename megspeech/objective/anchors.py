"""Fixed per-word anchors for the context head, built from training-partition stimuli only: the mean wav2vec2 latent
of each word's training occurrences (`audio`), or its t5 embedding (`text`)."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.objective.targets.t5 import Prototypes

KEY_LENGTH = 12

REPORTED_DIMS = (4, 8, 16, 32, 49)
"""The component counts whose cumulative explained variance is logged."""

LOG = logging.getLogger(__name__)


class Source(StrEnum):
    """Where the context head's anchors come from."""

    NONE = "none"
    AUDIO = "audio"
    TEXT = "text"

    @property
    def active(self) -> bool:
        return self is not Source.NONE


@dataclass(frozen=True)
class Anchors:
    """One vector per vocabulary word and how many training occurrences it averaged (0 where it is not an average)."""

    vectors: Float[Tensor, "v d"]
    counts: Int[Tensor, " v"]

    @classmethod
    def audio(cls, root: DataRoot, words: pd.DataFrame, labels: Int[Tensor, " n"]) -> "Anchors":
        """Each word's mean latent over its training occurrences that have a speech target, in the rows `words`
        (`run`, `position`) names, labelled `labels`; a word with none has no anchor."""
        if len(words) != len(labels):
            raise ValueError(f"{len(words)} words against {len(labels)} labels")
        speech = Speech.gather(Targets.directory(root), words)
        labels = labels.cpu()
        keep = (speech.rows >= 0) & (labels >= 0) & (labels < VOCAB_SIZE)
        latents = speech.vectors.float()[speech.rows[keep]]
        counts = torch.bincount(labels[keep], minlength=VOCAB_SIZE)
        if (counts == 0).any():
            raise ValueError(f"vocabulary words {torch.nonzero(counts == 0).flatten().tolist()} have no speech target")
        sums = torch.zeros(VOCAB_SIZE, latents.shape[1]).index_add_(0, labels[keep], latents)
        return cls(sums / counts[:, None], counts)

    @classmethod
    def text(cls, root: DataRoot, words: pd.DataFrame, labels: Int[Tensor, " n"]) -> "Anchors":
        """The t5 embedding of each vocabulary word, whatever the rows."""
        del words, labels
        vectors = Prototypes.load(Prototypes.path(root)).vectors.float()
        return cls(vectors, torch.zeros(len(vectors), dtype=torch.long))

    @staticmethod
    def key(words: pd.DataFrame, labels: Int[Tensor, " n"]) -> str:
        """A digest of the training rows, so a cache built on other rows is never read."""
        rows = pd.util.hash_pandas_object(words[["run", "position"]], index=False).to_numpy()
        digest = hashlib.sha1(rows.tobytes() + labels.cpu().numpy().tobytes(), usedforsecurity=False)
        return digest.hexdigest()[:KEY_LENGTH]

    @staticmethod
    def path(root: DataRoot, source: Source, key: str) -> Path:
        return root.processed() / "anchors" / f"{source}-{key}.npz"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self.vectors.numpy(), counts=self.counts.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Anchors":
        with np.load(path) as saved:
            return cls(torch.from_numpy(saved["vectors"]), torch.from_numpy(saved["counts"]))

    def spectrum(self) -> Float[Tensor, " r"]:
        """Cumulative share of the variance about the mean the leading principal components of the anchors explain."""
        energy = torch.linalg.svdvals(self.vectors - self.vectors.mean(dim=0)) ** 2
        return energy.cumsum(0) / energy.sum()

    def reduce(self, dim: int) -> "Anchors":
        """The anchors centred and projected onto their top `dim` principal components, an orthonormal basis; 0 keeps
        them whole."""
        if dim == 0:
            return self
        if not 0 < dim < len(self.vectors):
            raise ValueError(f"anchor_dim {dim} outside 1..{len(self.vectors) - 1}")
        centred = self.vectors - self.vectors.mean(dim=0)
        _, _, basis = torch.linalg.svd(centred, full_matrices=False)
        return Anchors(centred @ basis[:dim].T, self.counts)

    @classmethod
    def of(cls, source: Source, root: DataRoot, words: pd.DataFrame, labels: Int[Tensor, " n"]) -> "Anchors | None":
        """`source`'s anchors, read from the cache or computed once and written; None for no source."""
        if not source.active:
            return None
        path = cls.path(root, source, cls.key(words, labels))
        if path.exists():
            return cls.load(path)
        anchors = BUILDERS[source](root, words, labels)
        anchors.save(path)
        return anchors

    def report(self, source: Source) -> None:
        """Log the cumulative explained variance at each of `REPORTED_DIMS`."""
        spectrum = self.spectrum()
        shares = {k: round(spectrum[k - 1].item(), 4) for k in REPORTED_DIMS if k <= len(spectrum)}
        LOG.info("%s anchors cumulative explained variance %s", source, shares)


BUILDERS: dict[Source, Callable[[DataRoot, pd.DataFrame, Int[Tensor, " n"]], Anchors]] = {
    Source.AUDIO: Anchors.audio,
    Source.TEXT: Anchors.text,
}
