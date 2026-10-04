"""How well a word's part of speech decodes at all: classical probes on timing alone, on a frozen model's MEG
embeddings, and on both stacked, fit on subject 0's training runs and scored on the official validation words.

Timing is what a test row carries besides MEG: each word's onset beside its neighbours' and its sentence's bounds.
Function words are short, so timing alone predicts part of speech; MEG counts only for what it adds beyond it. The
tags are `megspeech.data.labels.syntax`'s, a training target: no holdout text is read.

    uv run python -m megspeech.evaluation.studies.syntax sent40-ioi-ema9998-s0
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from core.paths import DataRoot
from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.data.raw.runs import DEEP, Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.reader import Reader
from megspeech.model.auxiliary import Tap
from megspeech.model.sentence import SentenceNet

LOG = logging.getLogger(__name__)

FOLDS = 3
"""Run-grouped folds the stacker's training inputs are predicted out of, so it never reads a probe's fit."""
FLOOR = 1e-6
"""Probability floor under each logged probability: a fold missing a tag predicts it at 0."""

Piece = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, str]
"""One run's timing features, word and context embeddings, tags, name, membership in the 50 and domain."""


@dataclass(frozen=True)
class Timing:
    """Each word's timing within its sentence, the features a test row's onsets give."""

    @staticmethod
    def features(words: pd.DataFrame) -> Float[np.ndarray, "n f"]:
        """Seconds to the next onset (the sentence's end for its last word), from the previous onset (0 for its
        first), the word's position, its sentence's length and the position as a share of it, in `words`' order."""
        ordered = words.reset_index(drop=True).sort_values(["run", "sentence", "onset"])
        grouped = ordered.groupby(["run", "sentence"], sort=False)
        following = grouped["onset"].shift(-1).fillna(ordered["end"]) - ordered["onset"]
        preceding = (ordered["onset"] - grouped["onset"].shift(1)).fillna(0.0)
        position = grouped.cumcount()
        length = grouped["onset"].transform("size")
        table = pd.DataFrame(
            {
                "following": following,
                "preceding": preceding,
                "position": position,
                "length": length,
                "share": position / length,
            }
        )
        return table.sort_index().to_numpy(np.float64)


@dataclass(frozen=True)
class Embedded:
    """A frozen sentence model's per-word representations, one per `Tap`."""

    net: SentenceNet

    def __call__(self, split: Split) -> dict[Tap, Float[np.ndarray, "n _"]]:  # devtools-ignore: test-mirror
        """Every word of `split`'s encoder embedding (its own 0.5 s window) and span transformer output (its
        neighbours' MEG and timing too), in word order."""
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        device = split.index.device
        out = {tap: torch.zeros(words + 1, tap.width(self.net), device=device) for tap in Tap}
        self.net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                refined, slots, _, _ = self.net.heard(batch.x, batch.onset, batch.slot, batch.subject)
                at = torch.where(batch.index != PAD, batch.index, words)
                for tap, buffer in out.items():
                    buffer[at] = tap.pick(slots, refined).float()
        return {tap: buffer[:words].cpu().numpy() for tap, buffer in out.items()}


@dataclass(frozen=True)
class Probed:
    """One feature set's part-of-speech predictions on the validation words."""

    tags: Int[np.ndarray, " n"]
    vocabulary: np.ndarray
    domain: np.ndarray

    def scores(self, predicted: Int[np.ndarray, " n"]) -> dict[str, tuple[float, int]]:
        """Balanced accuracy over the tags present and its denominator: every word, the 50's, the rest, and per
        domain."""
        subsets = {"all": np.ones(len(self.tags), bool), "in-50": self.vocabulary, "outside": ~self.vocabulary}
        subsets |= {str(name): self.domain == name for name in np.unique(self.domain)}
        return {
            name: (float(balanced_accuracy_score(self.tags[chosen], predicted[chosen])), int(chosen.sum()))
            for name, chosen in subsets.items()
            if chosen.any()
        }


@dataclass(frozen=True)
class Table:
    """Every probed word's features, tag, run, membership in the 50 and domain."""

    features: dict[str, Float[np.ndarray, "n _"]]
    tags: Int[np.ndarray, " n"]
    runs: np.ndarray
    vocabulary: np.ndarray
    domain: np.ndarray

    @classmethod
    def of(  # devtools-ignore: test-mirror
        cls, reader: Reader, embedded: Embedded, held: dict[str, list[Run]], parts: PartsOfSpeech
    ) -> "Table":
        pieces = []
        for name, runs in held.items():
            for run, (words, split) in reader.splits(runs).items():
                read = embedded(split)
                tags = parts.classes(torch.from_numpy(words["token"].to_numpy(np.int64))).numpy()
                in_50 = (words["target"] >= 0).to_numpy()
                pieces.append((Timing.features(words), read[Tap.WORD], read[Tap.CONTEXT], tags, run.name, in_50, name))
        return cls.joined(pieces)

    @classmethod
    def joined(cls, pieces: list[Piece]) -> "Table":
        """The pieces' rows concatenated, the untagged dropped."""
        timing, word, context, tags, run, vocabulary, domain = zip(*pieces, strict=True)
        tag = np.concatenate(tags)
        known = tag != UNKNOWN
        repeat = [len(t) for t in tags]
        return cls(
            {
                "timing": np.concatenate(timing)[known],
                "meg-word": np.concatenate(word)[known],
                "meg-context": np.concatenate(context)[known],
            },
            tag[known],
            np.repeat(run, repeat)[known],
            np.concatenate(vocabulary)[known],
            np.repeat(domain, repeat)[known],
        )


@dataclass(frozen=True)
class Probe:
    """Classical probes fit on the training table and scored on the validation table, alone and stacked."""

    train: Table
    valid: Table

    @staticmethod
    def model(name: str) -> HistGradientBoostingClassifier | Pipeline:
        """Gradient boosting for the few tabular timing features; a linear readout for an embedding."""
        if name == "timing":
            return HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=0)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))

    def probabilities(self, name: str) -> tuple[Float[np.ndarray, "m k"], Float[np.ndarray, "n k"]]:
        """`name`'s probe's log-probabilities: out-of-fold on the training runs, then fit on them all, on validation."""
        fit = self.model(name)
        x, y = self.train.features[name], self.train.tags
        folded = cross_val_predict(fit, x, y, cv=GroupKFold(FOLDS), groups=self.train.runs, method="predict_proba")
        return np.log(folded + FLOOR), np.log(fit.fit(x, y).predict_proba(self.valid.features[name]) + FLOOR)

    def predicted(
        self, names: tuple[str, ...], probed: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> Int[np.ndarray, " n"]:
        """The validation tags predicted from `names`' `probed` log-probabilities, read by a logistic stacker fit on
        their out-of-fold training ones."""
        inputs, outputs = zip(*(probed[name] for name in names), strict=True)
        stacker = LogisticRegression(max_iter=500).fit(np.hstack(inputs), self.train.tags)
        return stacker.predict(np.hstack(outputs))

    def report(self) -> dict[str, dict[str, tuple[float, int]]]:
        scored = Probed(self.valid.tags, self.valid.vocabulary, self.valid.domain)
        probed = {name: self.probabilities(name) for name in self.train.features}
        arms = (("timing",), ("meg-word",), ("meg-context",), ("timing", "meg-word"), ("timing", "meg-context"))
        return {"+".join(arm): scored.scores(self.predicted(arm, probed)) for arm in arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--stride", type=int, default=4, help="every stride-th subject-0 training run is probed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    trained_in = root.models() / args.name
    data = DataConfig.shared([trained_in], DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False), official=True)
    runs = Run.discover(root.raw())
    trained = [run for run in data.trained(runs) if run.subject == DEEP][:: args.stride]
    parts = PartsOfSpeech.load(PartsOfSpeech.path(root))
    embedded = Embedded(SentenceNet.load(trained_in / "model.pt").to(device))
    train = Table.of(reader, embedded, {"train": trained}, parts)
    valid = Table.of(reader, embedded, reader.ruled(runs), parts)
    LOG.info("probing on %d words of %d runs, scoring %d words", len(train.tags), len(trained), len(valid.tags))
    for arm, scores in Probe(train, valid).report().items():
        LOG.info("%s %s", arm, {name: (round(score, 4), n) for name, (score, n) in scores.items()})


if __name__ == "__main__":
    main()
