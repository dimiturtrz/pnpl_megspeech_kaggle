"""Which teacher's 50-word geometry matches what the MEG model confuses (megspeech-7gf.141.1).

The MEG matrix is the trained model's mean softmax score of each word over the 50 candidates, read on the scored
(validation) unseen-span words and symmetrised. A teacher's matrix is the cosine of its type-level word vectors. The
representational similarity of the two is the Spearman correlation of their off-diagonal upper triangles, with a
bootstrap over words for its interval and a Mantel permutation of the teacher's word labels for its one-sided p. CPU
only.

    uv run --no-sync python -m megspeech.evaluation.studies.rsa --run ema-cd25-masc27w-s1
    PYTHONPATH=<transformers>=5.19 uv run --no-sync python -m megspeech.evaluation.studies.rsa --write-gemma
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from scipy.stats import spearmanr
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.studies.expertcheck import Centroids
from megspeech.objective.targets.speech import Targets
from megspeech.objective.targets.t5 import Prototypes

LOG = logging.getLogger(__name__)

SEED = 0
DRAWS = 1000
FLAT = 1e-9
"""Spread under which a side is constant: a centred one-hot differs only by float rounding."""
INTERVAL = (2.5, 97.5)
GEMMA = "google/embeddinggemma-2"
GEMMA_PREFIX = "task: sentence similarity | query: "
"""The model's own prompt for symmetric word-to-word similarity (config_sentence_transformers.json)."""
GEMMA_DIMS = (768, 256, 128)
CACHED = (
    "wav2vec2-large-xlsr-53",
    "hubert-large-ll60k",
    "data2vec-audio-large",
    "gpt2-large",
    "whisper-medium-L8",
    "whisper-medium-L12",
    "whisper-medium-L16",
    "whisper-medium-L20",
    "whisper-medium-L24",
)
HEADER = (
    "| teacher | dims | words | RSA raw | 95% CI raw | perm p raw | min cos dist raw "
    "| RSA centred | 95% CI centred | perm p centred | min cos dist centred |"
)
DEFAULT_RUNS = (
    "ema-cd25-xsp10sem05-hub-s1-ft-s1",
    "ema-cd25-xsp10sem05-d2v-s1-ft-s1",
    "ema-cd25-masc27w-s1",
)
"""One member of each Deep ensemble kind: H (HuBERT speech aux), D (data2vec speech aux), M (MASC-27 pretrained)."""


@dataclass(frozen=True)
class Geometry:
    """A teacher's type-level word vectors, one row per vocabulary word; an all-zero row is a word it lacks."""

    vectors: Float[np.ndarray, "v d"]

    @property
    def present(self) -> Bool[np.ndarray, " v"]:
        return np.linalg.norm(self.vectors, axis=1) > 0

    def cosine(self) -> Float[np.ndarray, "v v"]:
        """Pairwise cosine; NaN on the rows and columns of absent words."""
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        unit = np.divide(self.vectors, norm, out=np.full(self.vectors.shape, np.nan), where=norm > 0)
        return unit @ unit.T

    def min_distance(self) -> float:
        """The smallest cosine distance between two distinct present words."""
        similarity = self.cosine()[np.ix_(self.present, self.present)]
        return float(1.0 - np.max(similarity[np.triu_indices(len(similarity), k=1)]))

    def centred(self) -> "Geometry":
        """Less the mean of the present words' vectors: the direction every word shares removed before the cosine."""
        mean = self.vectors[self.present].mean(axis=0)
        return Geometry(np.where(self.present[:, None], self.vectors - mean, 0.0))

    def truncated(self, dims: int) -> "Geometry":
        """The first `dims` coordinates of a Matryoshka embedding, re-normalised."""
        head = self.vectors[:, :dims]
        return Geometry(head / np.linalg.norm(head, axis=1, keepdims=True))

    @classmethod
    def one_hot(cls) -> "Geometry":
        return cls(np.eye(VOCAB_SIZE))


@dataclass(frozen=True)
class Meg:
    """The model's mean softmax over the 50 candidates for each true word (row), and how many words made each row."""

    scores: Float[np.ndarray, "v v"]
    counts: Int[np.ndarray, " v"]

    @classmethod
    def of(cls, probabilities: Float[np.ndarray, "n v"], labels: Int[np.ndarray, " n"]) -> "Meg":
        counts = np.bincount(labels, minlength=VOCAB_SIZE)
        sums = np.zeros((VOCAB_SIZE, probabilities.shape[1]))
        np.add.at(sums, labels, probabilities)
        return cls(sums / np.maximum(counts, 1)[:, None], counts)

    @property
    def present(self) -> Bool[np.ndarray, " v"]:
        return self.counts > 0

    def similarity(self) -> Float[np.ndarray, "v v"]:
        """The scores averaged with their transpose: how alike two words are, whichever is the truth."""
        return (self.scores + self.scores.T) / 2


@dataclass(frozen=True)
class Rsa:
    """One teacher's agreement with the MEG matrix over the words both have."""

    rho: float
    low: float
    high: float
    words: int
    p: float

    @staticmethod
    def spearman(meg: Float[np.ndarray, " p"], teacher: Float[np.ndarray, " p"]) -> float:
        """Spearman correlation; 0 where either side is constant (a one-hot teacher has no geometry to correlate)."""
        if np.ptp(meg) < FLAT or np.ptp(teacher) < FLAT:
            return 0.0
        return float(spearmanr(meg, teacher).statistic)

    @classmethod
    def pairs(
        cls, meg: Float[np.ndarray, "v v"], teacher: Float[np.ndarray, "v v"], rows: Int[np.ndarray, " k"]
    ) -> float:
        """The Spearman correlation over the distinct-word pairs among `rows` (a word drawn twice pairs with itself
        nowhere)."""
        first, second = np.triu_indices(len(rows), k=1)
        apart = rows[first] != rows[second]
        a, b = rows[first][apart], rows[second][apart]
        return cls.spearman(meg[a, b], teacher[a, b])

    @classmethod
    def mantel(
        cls,
        meg: Float[np.ndarray, "v v"],
        teacher: Float[np.ndarray, "v v"],
        words: Int[np.ndarray, " k"],
        draws: int = DRAWS,
    ) -> float:
        """One-sided permutation p of the correlation over `words`: the teacher's word labels shuffled, the MEG's
        kept, so the null keeps each matrix's own structure and breaks only their correspondence."""
        first, second = np.triu_indices(len(words), k=1)
        observed = meg[words[first], words[second]]
        rho = cls.spearman(observed, teacher[words[first], words[second]])
        rng = np.random.default_rng(SEED)
        null = np.array(
            [
                cls.spearman(observed, teacher[shuffled[first], shuffled[second]])
                for shuffled in (rng.permutation(words) for _ in range(draws))
            ]
        )
        return float((1 + np.sum(null >= rho)) / (1 + draws))

    @classmethod
    def of(
        cls,
        meg: Float[np.ndarray, "v v"],
        teacher: Float[np.ndarray, "v v"],
        shared: Bool[np.ndarray, " v"],
        draws: int = DRAWS,
    ) -> "Rsa":
        """The correlation over the `shared` words, the percentile interval of `draws` word resamples, and the
        permutation p of `draws` label shuffles."""
        words = np.flatnonzero(shared)
        rng = np.random.default_rng(SEED)
        resampled = [cls.pairs(meg, teacher, rng.choice(words, size=len(words))) for _ in range(draws)]
        low, high = np.percentile(resampled, INTERVAL)
        return cls(
            cls.pairs(meg, teacher, words), float(low), float(high), len(words), cls.mantel(meg, teacher, words, draws)
        )


@dataclass(frozen=True)
class Row:
    """One teacher's line of the report: raw and centred."""

    name: str
    raw: Geometry
    centred: Geometry
    rsa_raw: Rsa
    rsa_centred: Rsa

    def markdown(self) -> str:
        return (
            f"| {self.name} | {self.raw.vectors.shape[1]} | {self.rsa_raw.words} "
            f"| {self.rsa_raw.rho:+.3f} | [{self.rsa_raw.low:+.3f}, {self.rsa_raw.high:+.3f}] | {self.rsa_raw.p:.3f} "
            f"| {self.raw.min_distance():.3f} "
            f"| {self.rsa_centred.rho:+.3f} | [{self.rsa_centred.low:+.3f}, {self.rsa_centred.high:+.3f}] "
            f"| {self.rsa_centred.p:.3f} | {self.centred.min_distance():.3f} |"
        )

    @classmethod
    def of(cls, name: str, geometry: Geometry, meg: Meg, draws: int = DRAWS) -> "Row":
        centred = geometry.centred()
        shared = meg.present & geometry.present
        similarity = meg.similarity()
        return cls(
            name,
            geometry,
            centred,
            Rsa.of(similarity, geometry.cosine(), shared, draws),
            Rsa.of(similarity, centred.cosine(), shared, draws),
        )


@dataclass(frozen=True)
class GemmaWords:
    """EmbeddingGemma 2's text embedding of each of the 50 words, written once by a transformers that knows the
    architecture (>= 5.19) and read back by the study."""

    vectors: Float[np.ndarray, "v d"]

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "prototypes" / "embeddinggemma-2-words.npy"

    @staticmethod
    def pooled(hidden: Float[Tensor, "b t d"], mask: Int[Tensor, "b t"]) -> Float[Tensor, "b d"]:
        """Mean over the unmasked tokens, unit length: the model's own pooling and normalisation."""
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * weight).sum(1) / weight.sum(1)
        return mean / mean.norm(dim=-1, keepdim=True)

    @classmethod
    def encode(cls, words: tuple[str, ...]) -> "GemmaWords":  # devtools-ignore: test-mirror
        tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(GEMMA))
        model = AutoModel.from_pretrained(GEMMA).eval()
        batch = tokenizer([GEMMA_PREFIX + word for word in words], return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state.float()
        return cls(cls.pooled(hidden, batch["attention_mask"]).numpy())

    @classmethod
    def load(cls, root: DataRoot) -> "GemmaWords":
        return cls(np.load(cls.path(root)))

    def save(self, root: DataRoot) -> Path:
        path = self.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)
        return path


@dataclass(frozen=True)
class Study:
    """The trained model, on CPU, and the data root its teachers come from."""

    root: DataRoot
    loaded: ExportedModel

    def meg(self) -> Meg:  # devtools-ignore: test-mirror
        """The model's softmax scores on the unseen-span words of every counted validation domain."""
        runs = Run.discover(self.root.raw())
        tau = self.loaded.report(runs).tau
        domains, _ = self.loaded.domains(self.loaded.ruled(runs))
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for split in domains.values():
            novel = split.novel(self.loaded.evaluator.seen)
            probabilities.append(self.loaded.probabilities(split, tau)[novel.cpu().numpy()])
            labels.append(split.y[novel].cpu().numpy())
        return Meg.of(np.concatenate(probabilities), np.concatenate(labels))

    def cached(self, name: str, words: pd.DataFrame) -> Geometry | None:  # devtools-ignore: test-mirror
        """The mean unit per-token target of each word over the training words, if `name` is cached."""
        directory = Targets.directory(self.root, name)
        if not directory.is_dir():
            return None
        return Geometry(Centroids.fit(words, directory).means.numpy())

    def training(self, subject: int | None = None) -> pd.DataFrame:  # devtools-ignore: test-mirror
        """The training partition's words, of `subject` alone when given."""
        runs = [run for run in Run.discover(self.root.raw()) if run.partition is Partition.TRAIN]
        return self.loaded.spoken([run for run in runs if subject in (None, run.subject)])[1]

    def teachers(self, subject: int | None = None) -> dict[str, Geometry]:  # devtools-ignore: test-mirror
        """Every teacher whose vectors exist, fitted on `subject`'s training words; a missing cache is logged and
        skipped."""
        teachers = {"one-hot": Geometry.one_hot()}
        t5 = Prototypes.path(self.root)
        if t5.exists():
            teachers["t5-large h12"] = Geometry(Prototypes.load(t5).vectors.numpy())
        words = self.training(subject)
        for name in CACHED:
            cached = self.cached(name, words)
            if cached is None:
                LOG.info("skipped %s: no cache under processed/speech", name)
            else:
                teachers[name] = cached
        if GemmaWords.path(self.root).exists():
            gemma = Geometry(GemmaWords.load(self.root).vectors)
            teachers.update({f"embeddinggemma-2 {dims}": gemma.truncated(dims) for dims in GEMMA_DIMS})
        else:
            LOG.info("skipped embeddinggemma-2: run --write-gemma with transformers >= 5.19 first")
        return teachers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--write-gemma", action="store_true", help="embed the 50 words with EmbeddingGemma 2 and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    if args.write_gemma:
        LOG.info("wrote %s", GemmaWords.encode(PRIMARY.words).save(root))
        return
    studies = {name: Study(root, Loader.of(Group(root, Scoring(), Device.CPU.as_torch()), [name])) for name in args.run}
    teachers = next(iter(studies.values())).teachers()
    for name, study in studies.items():
        meg = study.meg()
        LOG.info(
            "%s: MEG matrix %d of %d words, %d scored words",
            name,
            int(meg.present.sum()),
            VOCAB_SIZE,
            int(meg.counts.sum()),
        )
        LOG.info(HEADER)
        LOG.info("|---|---|---|---|---|---|---|---|---|---|---|")
        for teacher, geometry in teachers.items():
            LOG.info(Row.of(teacher, geometry, meg).markdown())


if __name__ == "__main__":
    main()
