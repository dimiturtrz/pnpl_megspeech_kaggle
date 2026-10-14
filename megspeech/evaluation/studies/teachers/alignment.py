"""How much of a teacher's target space a trained MEG word encoder can reach (megspeech-99y): the regularised canonical
correlations between the encoder's word embeddings and each teacher's per-word targets, fit on official training words
and read on official validation words, against the same correlations with the validation pairing shuffled.

A teacher whose canonical correlations stay inside the shuffle null past its first few components has a few directions
MEG predicts and a thousand it cannot; the probe's balanced accuracy at 10 candidates is the same teacher's own
50-word ceiling and the MEG embedding's the reference it is read against.

    uv run python -m megspeech.evaluation.studies teachers.alignment --run sent40-ioi-ema9998-s0
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import DEEP, Column, Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.reader import Reader
from megspeech.evaluation.studies.experts.owntask import Probe
from megspeech.evaluation.studies.pos.syntax import Embedded
from megspeech.model.auxiliary import Tap
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.objective.targets.t5 import Lexicon

LOG = logging.getLogger(__name__)

RIDGE = 0.1
"""Ridge of each side's covariance as a share of its mean eigenvalue: 512 embedding and 1024 target dimensions over
tens of thousands of words, enough to keep the whitening off the near-null directions."""
PROFILE = 64
TOP = 5
PAIRS = 20000
SHUFFLES = 200
QUANTILE = 0.95
SEED = 0
CHUNK = 8
"""Runs loaded to the device at once; the embeddings leave it before the next chunk."""
T5 = "t5-large"
DEFAULT_TEACHERS = (
    T5,
    "gpt2-large",
    "gpt2-large-white",
    "gpt2-large-sent",
    "wav2vec2-large-xlsr-53",
    "whisper-medium-L24",
    "whisper-medium-L24-white",
    "whisper-medium-L24-sent",
)
MIN_VAL = 200
"""Fewest validation words a teacher needs to be read at all."""
HELD_EVERY = 5
"""Every this-many-th training run (in name order) is held out when a teacher has no official validation targets."""
HEADER = (
    "| teacher / val set | dims | n train/val | anisotropy cos / top-5 | #sig comps (per-comp / family) "
    "| val corr top-1/5/20 | teacher-probe BAcc@10 |"
)


@dataclass(frozen=True)
class Anisotropy:
    """How lopsided vectors are: the mean cosine of random pairs and the variance share of the top `TOP` directions."""

    cosine: float
    top: float

    @classmethod
    def of(cls, vectors: Float[Tensor, "n d"]) -> "Anisotropy":
        unit = functional.normalize(vectors.float(), dim=-1)
        generator = torch.Generator().manual_seed(SEED)
        first, second = torch.randint(len(unit), (2, PAIRS), generator=generator)
        apart = first != second
        cosine = (unit[first[apart]] * unit[second[apart]]).sum(-1).mean()
        variance = torch.linalg.svdvals(vectors.float() - vectors.float().mean(0)).square()
        return cls(float(cosine), float(variance[:TOP].sum() / variance.sum()))


@dataclass(frozen=True)
class Cca:
    """Canonical axes of two sets of vectors, fit on one set of pairs and read on another."""

    mean_x: Float[Tensor, " p"]
    mean_y: Float[Tensor, " q"]
    axes_x: Float[Tensor, "p k"]
    axes_y: Float[Tensor, "q k"]

    @classmethod
    def fit(cls, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"], ridge: float = RIDGE) -> "Cca":
        """The ridge-regularised canonical axes of `x` and `y`, strongest correlation first."""
        x, y = x.double(), y.double()
        mean_x, mean_y = x.mean(0), y.mean(0)
        x, y = x - mean_x, y - mean_y
        n = len(x)
        root_x, root_y = cls.inverse_root(x.T @ x / n, ridge), cls.inverse_root(y.T @ y / n, ridge)
        left, _, right = torch.linalg.svd(root_x @ (x.T @ y / n) @ root_y, full_matrices=False)
        return cls(mean_x, mean_y, root_x @ left, root_y @ right.T)

    @staticmethod
    def inverse_root(covariance: Float[Tensor, "d d"], ridge: float) -> Float[Tensor, "d d"]:
        """`covariance` plus `ridge` of its mean eigenvalue, to the power -1/2."""
        values, vectors = torch.linalg.eigh(covariance)
        values = values.clamp(min=0) + ridge * values.mean()
        return (vectors / values.sqrt()) @ vectors.T

    def projected(
        self, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"]
    ) -> tuple[Float[Tensor, "n k"], Float[Tensor, "n k"]]:
        return (x.double() - self.mean_x) @ self.axes_x, (y.double() - self.mean_y) @ self.axes_y

    @staticmethod
    def correlation(a: Float[Tensor, "n k"], b: Float[Tensor, "n k"]) -> Float[Tensor, " k"]:
        """Column-wise Pearson correlation."""
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=torch.finfo(a.dtype).tiny)

    def correlations(self, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"]) -> Float[Tensor, " k"]:
        """Each canonical component's correlation over the pairs `x`, `y`."""
        return self.correlation(*self.projected(x, y))

    def shuffled(
        self, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"], shuffles: int = SHUFFLES
    ) -> Float[Tensor, "s k"]:
        """The correlations of `shuffles` random re-pairings of `x`'s rows with `y`'s."""
        a, b = self.projected(x, y)
        generator = torch.Generator().manual_seed(SEED)
        return torch.stack(
            [self.correlation(a, b[torch.randperm(len(b), generator=generator)]) for _ in range(shuffles)]
        )


@dataclass(frozen=True)
class Reach:
    """Validation canonical correlations against the shuffle null."""

    profile: Float[Tensor, " k"]
    null: Float[Tensor, "s k"]

    @classmethod
    def of(cls, cca: Cca, x: Float[Tensor, "n p"], y: Float[Tensor, "n q"], shuffles: int = SHUFFLES) -> "Reach":
        return cls(cca.correlations(x, y), cca.shuffled(x, y, shuffles))

    @property
    def per_component(self) -> int:
        """Components whose correlation tops their own shuffle null's `QUANTILE`: with `k` components, about
        `(1 - QUANTILE) * k` pass by chance."""
        return int((self.profile > torch.quantile(self.null, QUANTILE, dim=0)).sum())

    @property
    def family(self) -> int:
        """Components whose correlation tops the `QUANTILE` of the shuffles' strongest component: no chance passes."""
        return int((self.profile > torch.quantile(self.null.max(dim=1).values, QUANTILE)).sum())

    def top(self, k: int) -> float:
        """The `k`-th strongest validation correlation (1-based)."""
        return float(self.profile[k - 1])


class Teacher(Protocol):
    def gather(self, words: pd.DataFrame) -> tuple[Bool[np.ndarray, " n"], Float[Tensor, "m d"]]: ...


@dataclass(frozen=True)
class Directory:
    """Per-word targets saved per run under `processed/speech/<name>`."""

    name: str
    root: DataRoot

    def gather(self, words: pd.DataFrame) -> tuple[Bool[np.ndarray, " n"], Float[Tensor, "m d"]]:
        """The words (`run`, `position`) that have a target, and their targets."""
        speech = Speech.gather(Targets.directory(self.root, self.name), words)
        present = (speech.rows >= 0).numpy()
        return present, speech.vectors.float()[speech.rows[torch.from_numpy(present)]]


@dataclass(frozen=True)
class Lexical:
    """A teacher that embeds a word's text alone: one vector per distinct `token`."""

    lexicon: Lexicon

    def gather(self, words: pd.DataFrame) -> tuple[Bool[np.ndarray, " n"], Float[Tensor, "m d"]]:
        """The words (`token`) the lexicon holds, and their vectors."""
        rows = self.lexicon.rows(torch.from_numpy(words[Column.TOKEN].to_numpy(np.int64, copy=True)))
        present = (rows >= 0).numpy()
        vectors = self.lexicon.vectors
        return present, vectors.float()[rows[torch.from_numpy(present)]]


@dataclass(frozen=True)
class Alignment:
    """One teacher's reach from MEG, its anisotropy and its probe's 50-word score, on the validation set `split`."""

    teacher: str
    split: str
    dims: int
    train: int
    val: int
    anisotropy: Anisotropy
    reach: Reach
    ceiling: float

    def curve(self) -> str:
        """The validation correlations of the first `PROFILE` components."""
        return " ".join(f"{value:.2f}" for value in self.reach.profile[:PROFILE].tolist())

    def row(self) -> str:
        """A markdown table row under `HEADER`."""
        return (
            f"| {self.teacher} / {self.split} | {self.dims} | {self.train}/{self.val} "
            f"| {self.anisotropy.cosine:.3f} / {self.anisotropy.top:.3f} "
            f"| {self.reach.per_component} / {self.reach.family} "
            f"| {self.reach.top(1):.3f} / {self.reach.top(5):.3f} / {self.reach.top(20):.3f} "
            f"| {self.ceiling:.4f} |"
        )


@dataclass(frozen=True)
class Words:
    """Words (`run`, `position`, `target`, `token`) and the trained encoder's embedding of each."""

    table: pd.DataFrame
    vectors: Float[Tensor, "n e"]

    @classmethod
    def embed(  # devtools-ignore: test-mirror
        cls, embedded: Embedded, reader: Reader, runs: list[Run], limit: int
    ) -> "Words":
        """`runs`' words and their word embeddings, `CHUNK` runs on the device at a time, at most `limit` words."""
        tables: list[pd.DataFrame] = []
        vectors: list[Tensor] = []
        for start in range(0, len(runs), CHUNK):
            for part, split in reader.splits(runs[start : start + CHUNK]).values():
                tables.append(part[[Column.RUN, Column.POSITION, Column.TARGET, Column.TOKEN]])
                vectors.append(torch.from_numpy(embedded(split)[Tap.WORD]))
        return cls(pd.concat(tables, ignore_index=True), torch.cat(vectors)).sample(limit)

    def sample(self, limit: int) -> "Words":
        """At most `limit` words, drawn at random."""
        if len(self.table) <= limit:
            return self
        chosen = torch.randperm(len(self.table), generator=torch.Generator().manual_seed(SEED))[:limit].sort().values
        return Words(self.table.iloc[chosen.numpy()].reset_index(drop=True), self.vectors[chosen])

    def everything(self) -> tuple[Float[Tensor, "n e"], Int[Tensor, " n"]]:
        """Every word's embedding and 50-way target."""
        return self.vectors.float(), self.labels(np.ones(len(self.table), dtype=bool))

    def by_runs(self) -> tuple["Words", "Words"]:
        """The words of all but every `HELD_EVERY`-th run, and those of the rest (runs in name order)."""
        names = sorted(self.table[Column.RUN].unique())
        held = self.table[Column.RUN].isin(names[::HELD_EVERY]).to_numpy()
        keep, drop = torch.from_numpy(~held), torch.from_numpy(held)
        return (
            Words(self.table[~held].reset_index(drop=True), self.vectors[keep]),
            Words(self.table[held].reset_index(drop=True), self.vectors[drop]),
        )

    def labels(self, present: Bool[np.ndarray, " n"]) -> Int[Tensor, " m"]:
        """The 50-way target of each `present` word, IGNORE where it is outside the 50."""
        return torch.from_numpy(self.table[Column.TARGET].to_numpy(np.int64)[present])


@dataclass(frozen=True)
class Study:
    """The train and validation words, each with its MEG embedding."""

    train: Words
    val: Words

    @staticmethod
    def ceiling(
        train: tuple[Float[Tensor, "n d"], Int[Tensor, " n"]], val: tuple[Float[Tensor, "m d"], Int[Tensor, " m"]]
    ) -> float:
        """Balanced accuracy at 10 candidates of a softmax probe fit on `train` and read on `val`."""
        return Probe.learn(*train).accuracy(*val)

    def meg_ceiling(self) -> float:
        """The probe's score on the MEG embeddings themselves, over every word."""
        return self.ceiling(self.train.everything(), self.val.everything())

    def held_out(self) -> "Study":
        """The training words alone, some runs held out as the validation words: for a teacher whose targets are
        saved for training runs only. The encoder trained on those runs, so its embeddings of them are in-sample."""
        return Study(*self.train.by_runs())

    def align(self, name: str, split: str, teacher: Teacher, shuffles: int = SHUFFLES) -> Alignment | None:
        """`name`'s teacher against the MEG embeddings of the words it has targets for; None if too few validation
        words have one."""
        sides = []
        for words in (self.train, self.val):
            present, targets = teacher.gather(words.table)
            sides.append((words.vectors[torch.from_numpy(present)].float(), targets, words.labels(present)))
        (x_train, y_train, labels_train), (x_val, y_val, labels_val) = sides
        if len(y_val) < MIN_VAL:
            return None
        cca = Cca.fit(x_train, y_train)
        return Alignment(
            name,
            split,
            y_train.shape[1],
            len(y_train),
            len(y_val),
            Anisotropy.of(y_train),
            Reach.of(cca, x_val, y_val, shuffles),
            self.ceiling((y_train, labels_train), (y_val, labels_val)),
        )

    @staticmethod
    def teacher(root: DataRoot, name: str) -> Teacher:
        """The T5 lexicon for `T5`, else the saved per-word targets under `name`."""
        return Lexical(Lexicon.load(Lexicon.path(root))) if name == T5 else Directory(name, root)

    def report(self, root: DataRoot, teachers: list[str], shuffles: int = SHUFFLES) -> None:
        """Log each teacher's row and component curve on the validation words, then on held-out training runs."""
        for split, scored in (("official-val", self), ("held-out train runs", self.held_out())):
            for name in teachers:
                result = scored.align(name, split, self.teacher(root, name), shuffles)
                LOG.info(result.row() if result else f"| {name} / {split} | no targets |")
                if result:
                    LOG.info("  %s val corr by component: %s", name, result.curve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="sent40-ioi-ema9998-s0")
    parser.add_argument("--teachers", nargs="+", default=list(DEFAULT_TEACHERS))
    parser.add_argument("--max-train", type=int, default=40000)
    parser.add_argument("--shuffles", type=int, default=SHUFFLES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    directory = root.models() / args.run
    data = DataConfig.saved(directory)
    reader = Reader(root, data, device, geometry=data.geometry(root, device))
    embedded = Embedded.load(directory / "model.pt", device)
    runs = [run for run in Run.everywhere(root) if run.subject == DEEP]
    parts = {part: [run for run in runs if run.partition is part] for part in (Partition.TRAIN, Partition.VALIDATION)}
    study = Study(
        Words.embed(embedded, reader, parts[Partition.TRAIN], args.max_train),
        Words.embed(embedded, reader, parts[Partition.VALIDATION], args.max_train),
    )
    LOG.info("words: train %d val %d", len(study.train.table), len(study.val.table))
    LOG.info("meg-word probe BAcc@10 (all words): %.4f", study.meg_ceiling())
    LOG.info(HEADER)
    study.report(root, args.teachers, args.shuffles)


if __name__ == "__main__":
    main()
