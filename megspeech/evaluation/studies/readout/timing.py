"""How much of a trained model's MEG representation is timing, and how much of its score the timing alone earns.

Each word's duration (its gap to the next onset) and place in its sentence are what a test row's onsets give besides
MEG, and also what evoked responses time-lock to. Probe (a): linear R^2 of both, read off the frozen word and context
representations, out of run-grouped folds on subject 0's training runs and train-fit on the official validation
words. Probe (b): the model's top-10 hits against its own hits with every word's MEG window swapped for another
word's of the same domain, onsets untouched (megspeech-7gf.71): the hits only timing can earn, how many of them the
real model shares, and the overlap two independent hitters would show.

    uv run python -m megspeech.evaluation.studies readout.timing sent40-ioi-ema9998-s0
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import DEEP, Column, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.loaded.reader import Reader
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.studies.pos.syntax import Embedded
from megspeech.model.auxiliary import Tap

LOG = logging.getLogger(__name__)

FOLDS = 3
"""Run-grouped folds the out-of-fold R^2 is read from."""
ALPHA = 100.0
"""Ridge penalty on standardised features: ~1e5 words against 256 dimensions, so it only steadies the fit."""
PERMUTATIONS = 3
"""Shuffles of the validation windows the timing-only hits are averaged over."""
SENTENCE = [Column.RUN, Column.SENTENCE]


@dataclass(frozen=True)
class Readings:
    """Each probed word's frozen representation per `Tap`, its run, log duration (NaN for a sentence's last word) and
    place in its sentence."""

    embeddings: dict[Tap, Float[np.ndarray, "n _"]]
    runs: np.ndarray
    duration: Float[np.ndarray, " n"]
    position: Float[np.ndarray, " n"]

    @staticmethod
    def targets(words: pd.DataFrame) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
        """`words`' log gap to the next onset of the sentence, NaN for its last word, and each word's index in it."""
        onset = words[Column.ONSET].astype(np.float64)
        grouped = words.groupby(SENTENCE)[Column.ONSET]
        gap = grouped.shift(-1).astype(np.float64) - onset
        position = grouped.rank(method="first").to_numpy(np.float64) - 1
        return np.log(gap.clip(lower=np.finfo(np.float64).eps).to_numpy()), position

    @classmethod
    def of(cls, reader: Reader, embedded: Embedded, runs: list[Run]) -> "Readings":  # devtools-ignore: test-mirror
        """The words of `runs`, each run read as the model reads it."""
        pieces = []
        for words, split in reader.splits(runs).values():
            duration, position = cls.targets(words)
            pieces.append((embedded(split), words[Column.RUN].to_numpy(), duration, position))
        read, names, duration, position = zip(*pieces, strict=True)
        embeddings = {tap: np.concatenate([r[tap] for r in read]) for tap in Tap.pooled()}
        return cls(embeddings, np.concatenate(names), np.concatenate(duration), np.concatenate(position))

    def target(self, name: str) -> Float[np.ndarray, " n"]:
        return {"duration": self.duration, "position": self.position}[name]


@dataclass(frozen=True)
class Probe:
    """Ridge readouts of duration and position from each tap, fit on `train`'s words."""

    train: Readings
    valid: Readings

    @staticmethod
    def readout() -> Pipeline:
        return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))

    def report(self) -> dict[str, tuple[float, int]]:
        """R^2 and word count per (`tap`, target, `folds` or `valid`) reading."""
        out = {}
        for tap in Tap.pooled():
            for name in ("duration", "position"):
                x, y = self.train.embeddings[tap], self.train.target(name)
                known = ~np.isnan(y)
                folded = cross_val_predict(
                    self.readout(), x[known], y[known], groups=self.train.runs[known], cv=GroupKFold(FOLDS)
                )
                out[f"{tap.value} {name} out-of-fold"] = (float(r2_score(y[known], folded)), int(known.sum()))
                held = self.valid.target(name)
                seen = ~np.isnan(held)
                fitted = self.readout().fit(x[known], y[known])
                predicted = fitted.predict(self.valid.embeddings[tap][seen])
                out[f"{tap.value} {name} official-val"] = (float(r2_score(held[seen], predicted)), int(seen.sum()))
        return out


@dataclass(frozen=True)
class Agreement:
    """Two models' top-10 hits on the same words: counts of each outcome and the both-hit count two independent
    models with the same hit rates would show."""

    n: int
    both: int
    first_only: int
    second_only: int
    expected: float

    @classmethod
    def of(cls, first: Bool[np.ndarray, " n"], second: Bool[np.ndarray, " n"]) -> "Agreement":
        n = len(first)
        return cls(
            n,
            int((first & second).sum()),
            int((first & ~second).sum()),
            int((~first & second).sum()),
            float(first.mean() * second.mean() * n),
        )

    @staticmethod
    def pooled(graded: Graded) -> tuple[Bool[np.ndarray, " n"], np.ndarray]:
        """Every domain's hits end to end, and the domain of each."""
        hits = np.concatenate([h.cpu().numpy() for h, _ in graded.domains.values()])
        names = np.concatenate([[name] * len(h) for name, (h, _) in graded.domains.items()])
        return hits, names

    def neither(self) -> int:
        return self.n - self.both - self.first_only - self.second_only

    def summary(self) -> dict[str, float]:
        return {
            "n": self.n,
            "both": self.both,
            "first-only": self.first_only,
            "second-only": self.second_only,
            "neither": self.neither(),
            "both-expected-if-independent": round(self.expected, 1),
            "agreement": round((self.both + self.neither()) / self.n, 4),
            "agreement-if-independent": round(
                (2 * self.expected + self.n - 2 * self.both - self.first_only - self.second_only) / self.n,
                4,
            ),
        }


@dataclass(frozen=True)
class Shuffled:
    """A model scored on validation domains whose words each read another word's MEG window of the same domain, with
    the word's own onsets: what its timing alone earns."""

    loaded: LoadedModel
    tau: float

    def graded(self, domains: dict[str, Split], seed: int) -> Graded:
        shuffled = {name: split.permuted(seed) for name, split in domains.items()}
        loaded = self.loaded
        return loaded.evaluator.graded(loaded.model, shuffled, loaded.calibration(self.tau))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--stride", type=int, default=4, help="every stride-th subject-0 training run is probed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    loaded = Loader.of(Group(root, Scoring(), device), [args.name])
    runs = Run.discover(root.raw())
    held = [run for group in loaded.ruled(runs).values() for run in group]
    trained = [run for run in loaded.data.trained(runs) if run.subject == DEEP][:: args.stride]
    embedded = Embedded.load(root.models() / args.name / "model.pt", device)
    probe = Probe(Readings.of(loaded, embedded, trained), Readings.of(loaded, embedded, held))
    for reading, (r2, n) in probe.report().items():
        LOG.info("R2 %s %.4f (n=%d)", reading, r2, n)
    tau = loaded.report(runs).tau
    domains, _ = loaded.domains(loaded.ruled(runs))
    base, names = Agreement.pooled(loaded.graded(runs, tau))
    shuffled = Shuffled(loaded, tau)
    for seed in range(PERMUTATIONS):
        timing, _ = Agreement.pooled(shuffled.graded(domains, seed))
        overall = Agreement.of(base, timing).summary()
        LOG.info("seed %d base %.4f timing-only %.4f %s", seed, base.mean(), timing.mean(), overall)
        for domain in dict.fromkeys(names):
            here = names == domain
            LOG.info("  %s %s", domain, Agreement.of(base[here], timing[here]).summary())


if __name__ == "__main__":
    main()
