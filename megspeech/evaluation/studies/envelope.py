"""Band-envelope instrument: does a band's onset-locked power carry word evidence a linear reader can use.

Subject 0's 250 Hz recordings, per band: a train-fit covariance of the band-passed signal (`COVARIANCE_RUNS` strided
training runs) whitened to its top `COMPONENTS` components, Butterworth band-pass (order `ORDER`, zero phase),
|Hilbert|, log, a per-run z-score per component, then `BINS` bins of 0.1 s over the 0.8 s after each word's onset. A
Ledoit-Wolf-shrunk LDA under a uniform prior reads the PCA of those features, the PCA width picked on an inner fold
(every `INNER_STRIDE`th training run); it is scored by top-10 balanced accuracy on the official validation domains'
novel-span words (text training never heard), macro over the domains that count. The mechanism check reads the
features themselves on the training words: the onset-locked grand mean of every component per bin against its
standard error, and the between over within class variance of each feature against its chance value
`(V - 1) / (N - V)`. A band whose grand mean is flat and whose variance ratio sits at chance holds no word-locked
power response at this resolution, whatever the classifier does.

    uv run python -m megspeech.evaluation.studies.envelope --bands 30-45 55-95 105-120 --joint
    uv run python -m megspeech.evaluation.studies.envelope --bands 4-8 8-13 13-25
"""

import argparse
import logging
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import reduce
from itertools import repeat
from typing import Self

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import SpanText
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import PICO, SFREQ, WORKERS
from megspeech.data.heard import Heard
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.score import MIN_NOVEL
from megspeech.evaluation.studies.templates import ABSENT, Noise

LOG = logging.getLogger(__name__)

ORDER = 4
COMPONENTS = 80
BINS = 8
BIN_SAMPLES = SFREQ // 10
"""0.1 s at the recordings' rate."""
WINDOW = BINS * BIN_SAMPLES
COVARIANCE_RUNS = 16
INNER_STRIDE = 10
DIM_FRACTIONS = (0.05, 0.1, 0.2, 0.34, 0.52, 0.63, 0.79, 0.89, 0.98)
"""PCA widths tried on the inner fold, as shares of the feature width."""
HILBERT_CHUNK = 16
EPS = 1e-12
MS = 1000.0
HIGH = ("30-45", "55-95", "105-120")
"""The 7gf.80 bands, the 50 and 100 Hz line harmonics avoided."""


@dataclass(frozen=True)
class Band:
    """A pass band in Hz."""

    low: float
    high: float

    @classmethod
    def parse(cls, spec: str) -> Self:
        low, high = spec.split("-")
        return cls(float(low), float(high))

    @property
    def label(self) -> str:
        return f"{self.low:g}-{self.high:g}Hz"

    def filtered(self, x: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "c t"]:
        """`x` through the zero-phase Butterworth band-pass."""
        sos = butter(ORDER, [self.low, self.high], btype="band", fs=SFREQ, output="sos")
        return sosfiltfilt(sos, x, axis=-1)


@dataclass(frozen=True)
class Recording:
    """One run's 250 Hz MEG in picotesla."""

    meg: Float[np.ndarray, "c t"]

    @classmethod
    def load(cls, run: Run) -> Self:
        with h5py.File(run.meg) as stored:
            rate = float(stored.attrs["sample_frequency"])
            if rate != SFREQ:
                raise ValueError(f"{run.meg.name} is {rate} Hz, not {SFREQ}")
            return cls(stored["data"][:].astype(np.float32) * np.float32(PICO))


@dataclass(frozen=True)
class Covariance:
    """The summed outer products of a band-passed signal's channels, centred per run, and the samples behind them."""

    total: Float[np.ndarray, "c c"]
    seen: int

    @classmethod
    def of(cls, filtered: Float[np.ndarray, "c t"]) -> Self:
        centred = filtered - filtered.mean(axis=1, keepdims=True)
        return cls(centred @ centred.T, filtered.shape[1])

    def merge(self, other: Self) -> Self:
        return type(self)(self.total + other.total, self.seen + other.seen)

    def whitener(self, components: int) -> Float[np.ndarray, "c k"]:
        """The top `components` eigenvectors scaled to unit variance: `x.T @ whitener` are whitened components."""
        values, vectors = np.linalg.eigh(self.total / self.seen)
        top = slice(None, -components - 1, -1)
        return (vectors[:, top] / np.sqrt(values[top])).astype(np.float32)


@dataclass(frozen=True)
class Envelope:
    """A band's whitened components' z-scored log envelope."""

    band: Band
    matrix: Float[np.ndarray, "c k"]

    def log(self, meg: Float[np.ndarray, "c t"]) -> Float[np.ndarray, "k t"]:
        """Per component, the log |Hilbert| of the band-passed signal, z-scored over the run."""
        comps = self.band.filtered(self.matrix.T @ meg)
        length = comps.shape[1]
        out = np.empty(comps.shape, dtype=np.float32)
        for start in range(0, len(comps), HILBERT_CHUNK):
            part = comps[start : start + HILBERT_CHUNK]
            analytic = hilbert(part, N=next_fast_len(length), axis=-1)[:, :length]
            out[start : start + HILBERT_CHUNK] = np.log(np.abs(analytic) + EPS)
        return (out - out.mean(axis=1, keepdims=True)) / out.std(axis=1, keepdims=True)

    def binned(self, meg: Float[np.ndarray, "c t"], onsets: Int[np.ndarray, " n"]) -> Float[np.ndarray, "n k bins"]:
        """The log envelope averaged over `BINS` bins of 0.1 s from each onset sample."""
        env = self.log(meg)
        at = onsets[:, None] + np.arange(WINDOW)
        window = env[:, at].reshape(len(env), len(onsets), BINS, BIN_SAMPLES)
        return window.mean(axis=-1).transpose(1, 0, 2)


@dataclass(frozen=True)
class Extractor:
    """Features of word rows from the runs' h5 files, one process per run."""

    bands: tuple[Band, ...]
    runs: Mapping[str, Run]
    workers: int = WORKERS

    @staticmethod
    def covariances(run: Run, bands: tuple[Band, ...]) -> list[Covariance]:
        meg = Recording.load(run).meg
        return [Covariance.of(band.filtered(meg)) for band in bands]

    @staticmethod
    def features(
        run: Run, onsets: Int[np.ndarray, " n"], envelopes: tuple[Envelope, ...]
    ) -> Float[np.ndarray, "n b k bins"]:
        meg = Recording.load(run).meg
        return np.stack([envelope.binned(meg, onsets) for envelope in envelopes], axis=1)

    def projections(self, names: list[str], components: int = COMPONENTS) -> tuple[Envelope, ...]:
        """Each band's envelope pipeline, its whitener fit on the runs `names`."""
        with ProcessPoolExecutor(self.workers) as pool:
            parts = list(pool.map(self.covariances, [self.runs[name] for name in names], repeat(self.bands)))
        merged = [reduce(Covariance.merge, per_band) for per_band in zip(*parts, strict=True)]
        return tuple(Envelope(band, cov.whitener(components)) for band, cov in zip(self.bands, merged, strict=True))

    def extract(self, table: pd.DataFrame, envelopes: tuple[Envelope, ...]) -> Float[np.ndarray, "n b k bins"]:
        """Every row of `table` (`run`, `onset` seconds) as binned envelopes of each band, in row order."""
        rows = {str(name): np.asarray(at) for name, at in table.groupby("run", sort=False).indices.items()}
        onsets = np.round(table["onset"].to_numpy() * SFREQ).astype(np.int64)
        with ProcessPoolExecutor(self.workers) as pool:
            parts = pool.map(
                self.features,
                [self.runs[name] for name in rows],
                [onsets[at] for at in rows.values()],
                repeat(envelopes),
            )
            out = np.empty((len(table), len(envelopes), envelopes[0].matrix.shape[1], BINS), dtype=np.float32)
            for at, part in zip(rows.values(), parts, strict=True):
                out[at] = part
        return out


@dataclass(frozen=True)
class Cohort:
    """The labelled training words and each counted validation domain's labelled novel-span words."""

    train: pd.DataFrame
    domains: dict[str, pd.DataFrame]

    @staticmethod
    def novel(words: pd.DataFrame, grouping: Grouping, seen: frozenset[tuple[int, ...]]) -> pd.DataFrame:
        """The labelled words of multi-word spans of `words` whose text training never heard."""
        spans = grouping.spans(words)
        picked = words.iloc[SpanText(spans.index, words["token"].to_numpy()).novel(seen)]
        return picked[picked["target"] != IGNORE]

    @classmethod
    def of(
        cls,
        trained: pd.DataFrame,
        held: Mapping[str, pd.DataFrame],
        grouping: Grouping,
        seen: frozenset[tuple[int, ...]],
    ) -> Self:
        """`trained` and `held` (a domain's words) cut to the words scored; a domain under `MIN_NOVEL` is dropped."""
        novel = {name: cls.novel(words, grouping, seen) for name, words in held.items()}
        return cls(
            trained[trained["target"] != IGNORE].reset_index(drop=True),
            {name: rows.reset_index(drop=True) for name, rows in novel.items() if len(rows) >= MIN_NOVEL},
        )

    def table(self) -> pd.DataFrame:
        """Every row to extract: the training words, then each domain's, tagged by `part`."""
        parts = {"train": self.train, **self.domains}
        return pd.concat([rows.assign(part=name) for name, rows in parts.items()], ignore_index=True)


@dataclass(frozen=True)
class Dataset:
    """Feature rows, their word labels and the index of the run each came from."""

    x: Float[Tensor, "n d"]
    y: Int[Tensor, " n"]
    group: Int[Tensor, " n"]

    def take(self, mask: Bool[Tensor, " n"]) -> Self:
        return type(self)(self.x[mask], self.y[mask], self.group[mask])


@dataclass(frozen=True)
class Pca:
    mean: Float[Tensor, " d"]
    axes: Float[Tensor, "d d"]
    """Principal axes as columns, by decreasing variance."""

    @classmethod
    def fit(cls, x: Float[Tensor, "n d"]) -> Self:
        mean = x.mean(dim=0)
        centred = x - mean
        _, vectors = torch.linalg.eigh((centred.T @ centred).double())
        return cls(mean, vectors.flip(1).float())

    def project(self, x: Float[Tensor, "n d"], dims: int) -> Float[Tensor, "n k"]:
        return (x - self.mean) @ self.axes[:, :dims]


@dataclass(frozen=True)
class Lda:
    """Class means under one Ledoit-Wolf-shrunk pooled covariance, a uniform class prior."""

    means: Float[Tensor, "v k"]
    precision: Float[Tensor, "k k"]
    present: Tensor

    @classmethod
    def fit(cls, z: Float[Tensor, "n k"], y: Int[Tensor, " n"]) -> Self:
        counts = torch.bincount(y, minlength=VOCAB_SIZE)
        sums = torch.zeros(VOCAB_SIZE, z.shape[1]).index_add_(0, y, z)
        means = sums / counts.clamp(min=1)[:, None]
        residual = z - means[y]
        total = (residual.T @ residual).double()
        quartic = residual.pow(2).sum(dim=1).double().pow(2).sum()
        cov = Noise.shrunk(total, quartic, torch.tensor(float(len(z)), dtype=torch.float64))
        return cls(means, torch.linalg.inv(cov).float(), counts > 0)

    def scores(self, z: Float[Tensor, "n k"]) -> Float[Tensor, "n v"]:
        """Every word's discriminant per row; `ABSENT` for a word with no training row."""
        weights = self.means @ self.precision
        bias = 0.5 * (weights * self.means).sum(dim=1)
        return (z @ weights.T - bias).masked_fill(~self.present, ABSENT)


@dataclass(frozen=True)
class Outcome:
    """The PCA width the inner fold chose and each domain's novel-span top-10 balanced accuracy."""

    dims: int
    domains: dict[str, float]

    @property
    def macro(self) -> float:
        return float(np.mean(list(self.domains.values())))


@dataclass(frozen=True)
class Harness:
    fractions: tuple[float, ...] = DIM_FRACTIONS

    def dims(self, width: int) -> list[int]:
        return sorted({max(1, round(share * width)) for share in self.fractions})

    @staticmethod
    def balanced(scores: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> float:
        ruler = Ruler()
        return ruler.balanced(ruler.hits(scores, y), y)

    def decoded(self, fit: Dataset, test: Dataset, dims: int, pca: Pca) -> float:
        lda = Lda.fit(pca.project(fit.x, dims), fit.y)
        return self.balanced(lda.scores(pca.project(test.x, dims)), test.y)

    def inner(self, train: Dataset) -> int:
        """The PCA width of best balanced accuracy on every `INNER_STRIDE`th training run, fit on the rest."""
        held = train.group % INNER_STRIDE == 0
        fit, test = train.take(~held), train.take(held)
        pca = Pca.fit(fit.x)
        found = {dims: self.decoded(fit, test, dims, pca) for dims in self.dims(train.x.shape[1])}
        LOG.info("inner fold %s", " ".join(f"{dims}:{score:.4f}" for dims, score in found.items()))
        return max(found, key=lambda dims: found[dims])

    def outcome(self, train: Dataset, domains: Mapping[str, Dataset]) -> Outcome:
        dims = self.inner(train)
        pca = Pca.fit(train.x)
        lda = Lda.fit(pca.project(train.x, dims), train.y)
        return Outcome(
            dims, {name: self.balanced(lda.scores(pca.project(d.x, dims)), d.y) for name, d in domains.items()}
        )


@dataclass(frozen=True)
class Mechanism:
    """Per bin over components: the grand-mean RMS, the RMS of its standard error, and the mean between over within
    class variance ratio against its chance value."""

    mean_rms: Float[np.ndarray, " b"]
    se_rms: Float[np.ndarray, " b"]
    ratio: Float[np.ndarray, " b"]
    chance: float

    @classmethod
    def of(cls, x: Float[Tensor, "n k b"], y: Int[Tensor, " n"]) -> Self:
        n, comps, bins = x.shape
        flat = x.flatten(1).double()
        grand = flat.mean(dim=0)
        total = (flat - grand).pow(2).sum(dim=0)
        counts = torch.bincount(y, minlength=VOCAB_SIZE).double()
        means = torch.zeros(VOCAB_SIZE, flat.shape[1], dtype=torch.float64).index_add_(0, y, flat)
        means = means / counts.clamp(min=1)[:, None]
        between = (counts[:, None] * (means - grand).pow(2)).sum(dim=0)
        classes = int((counts > 0).sum())

        def per_bin(values: Tensor) -> np.ndarray:
            return values.view(comps, bins).mean(dim=0).numpy()

        return cls(
            np.sqrt(per_bin(grand.pow(2))),
            np.sqrt(per_bin(total / (n - 1) / n)),
            per_bin(between / (total - between)),
            (classes - 1) / (n - classes),
        )

    def summary(self) -> str:
        """The peak grand-mean RMS over bins and the standard error there, and the variance ratio against chance."""
        peak = int(self.mean_rms.argmax())
        rms = " ".join(f"{v:.4f}" for v in self.mean_rms)
        return (
            f"grand-mean RMS per bin [{rms}] peak {self.mean_rms[peak]:.4f} (bin {peak}) vs SE {self.se_rms[peak]:.4f} "
            f"= {self.mean_rms[peak] / self.se_rms[peak]:.1f} SE; between/within ratio mean {self.ratio.mean():.6f} "
            f"max {self.ratio.max():.6f} vs chance {self.chance:.6f}"
        )


class Study:
    @classmethod
    def datasets(
        cls, cohort: Cohort, features: Float[np.ndarray, "n b k bins"], bands: list[int]
    ) -> tuple[Dataset, dict[str, Dataset]]:
        """The training rows, then each domain's, reading the bands at indices `bands` flattened side by side."""
        table = cohort.table()
        flat = torch.from_numpy(features[:, bands].reshape(len(table), -1))
        labels = torch.from_numpy(table["target"].to_numpy().astype(np.int64))
        runs = torch.from_numpy(pd.factorize(table["run"])[0].astype(np.int64))
        every = Dataset(flat, labels, runs)
        train = every.take(torch.from_numpy((table["part"] == "train").to_numpy()))
        domains = {name: every.take(torch.from_numpy((table["part"] == name).to_numpy())) for name in cohort.domains}
        return train, domains

    @classmethod
    def report(cls, label: str, harness: Harness, train: Dataset, domains: Mapping[str, Dataset]) -> str:
        outcome = harness.outcome(train, domains)
        parts = " ".join(f"{name} {score:.4f}(n={len(domains[name].y)})" for name, score in outcome.domains.items())
        return f"{label}: macro {outcome.macro:.4f} {parts} dims {outcome.dims} of {train.x.shape[1]}"

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bands", nargs="+", default=list(HIGH), help="pass bands in Hz, e.g. 4-8 8-13")
        parser.add_argument("--joint", action="store_true", help="also one LDA over every band's features together")
        parser.add_argument("--components", type=int, default=COMPONENTS)
        parser.add_argument("--workers", type=int, default=WORKERS)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        started = time.monotonic()
        root = DataRoot.from_config()
        reader = Reader(root, DataConfig(val=Val.OFFICIAL), torch.device("cpu"))
        runs = Run.everywhere(root)
        trained = reader.data.trained(runs)
        held = reader.ruled(runs)
        seen, _ = Heard.under(root, reader.data, reader.device)
        cohort = Cohort.of(
            reader.spoken(trained)[1],
            {name: reader.spoken(group)[1] for name, group in held.items()},
            reader.data.grouping,
            seen,
        )
        LOG.info(
            "train runs=%d labelled words=%d; novel %s",
            len(trained),
            len(cohort.train),
            {name: len(rows) for name, rows in cohort.domains.items()},
        )
        bands = tuple(Band.parse(spec) for spec in args.bands)
        table = cohort.table()
        extractor = Extractor(bands, {run.name: run for run in runs}, args.workers)
        names = list(dict.fromkeys(cohort.train["run"]))
        picked = [names[at] for at in np.linspace(0, len(names) - 1, COVARIANCE_RUNS).astype(int)]
        envelopes = extractor.projections(picked, args.components)
        LOG.info("whiteners fit on %d runs, %.0f s", len(picked), time.monotonic() - started)
        features = extractor.extract(table, envelopes)
        LOG.info("features %s, %.0f s", features.shape, time.monotonic() - started)
        harness = Harness()
        for at, band in enumerate(bands):
            train, domains = cls.datasets(cohort, features, [at])
            train_rows = (table["part"] == "train").to_numpy()
            mechanism = Mechanism.of(torch.from_numpy(features[train_rows, at]), train.y)
            LOG.info("%s mechanism %s", band.label, mechanism.summary())
            LOG.info("%s", cls.report(band.label, harness, train, domains))
        if args.joint:
            train, domains = cls.datasets(cohort, features, list(range(len(bands))))
            LOG.info("%s", cls.report("joint " + "+".join(b.label for b in bands), harness, train, domains))
        LOG.info("done in %.0f s", time.monotonic() - started)


if __name__ == "__main__":
    Study.main()
