"""How well a word's part of speech decodes at all: classical probes on timing alone, on a frozen model's MEG
embeddings, and on both stacked, fit on subject 0's training runs and scored on the official validation words.

Timing is what a test row carries besides MEG: each word's onset beside its neighbours'. Function words are short, so
timing alone predicts part of speech; MEG counts only for what it adds beyond it. The classes are
`megspeech.data.labels.context_tags`', a training target: no holdout text is read.

    uv run python -m megspeech.evaluation.studies.pos.syntax sent40-ioi-ema9998-s0
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int, Shaped
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.context_tags import CLASSES, ContextTags
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Column, Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.pos.grammar import Grammar
from megspeech.evaluation.studies.pos.logistic import TorchLogistic
from megspeech.model.auxiliary import Tap
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.tagging import Tagging

LOG = logging.getLogger(__name__)

FOLDS = 3
"""Run-grouped folds the stacker's training inputs are predicted out of, so it never reads a probe's fit."""
FLOOR = 1e-6
"""Probability floor under each logged probability: a fold missing a tag predicts it at 0."""

Piece = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]
"""One run's timing features, word and context embeddings, tags, name, membership in the 50, domain, sentence keys,
onsets and 50-way targets."""


@dataclass(frozen=True)
class Timing:
    """Each word's timing within its sentence, the features a test row's onsets give: a test row carries no word end,
    so a word's duration is its gap to the next onset. A sentence's last word has no next onset and takes `final`,
    the median gap of the other words (the least assuming stand-in: the speaking rhythm), the same in every split."""

    final: float

    @staticmethod
    def gaps(words: pd.DataFrame) -> pd.Series:
        """Seconds from each word's onset to the next in its sentence, NaN for the last, in `words`' order."""
        ordered = words.reset_index(drop=True).sort_values([Column.RUN, Column.SENTENCE, Column.ONSET])
        following = (
            ordered.groupby([Column.RUN, Column.SENTENCE], sort=False)[Column.ONSET].shift(-1) - ordered[Column.ONSET]
        )
        return following.sort_index()

    @classmethod
    def fitted(cls, words: pd.DataFrame) -> "Timing":
        """The timing whose `final` is the median gap over `words` (the training words)."""
        return cls(float(np.nanmedian(cls.gaps(words))))

    def features(self, words: pd.DataFrame) -> Float[np.ndarray, "n f"]:
        """Seconds to the next onset (`final` for the last word), from the previous onset (0 for its first), the
        word's position, its sentence's length and the position as a share of it, in `words`' order."""
        ordered = words.reset_index(drop=True).sort_values([Column.RUN, Column.SENTENCE, Column.ONSET])
        grouped = ordered.groupby([Column.RUN, Column.SENTENCE], sort=False)
        preceding = (ordered[Column.ONSET] - grouped[Column.ONSET].shift(1)).fillna(0.0)
        position = grouped.cumcount()
        length = grouped[Column.ONSET].transform("size")
        table = pd.DataFrame(
            {
                "following": self.gaps(words).fillna(self.final).reindex(ordered.index),
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

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Embedded":  # devtools-ignore: test-mirror
        """The model saved at `path`, on `device`."""
        return cls(Checkpoint.load(path).to(device))

    def tagged(self, split: Split) -> Int[np.ndarray, " n"]:  # devtools-ignore: test-mirror
        """Every word of `split`'s most probable tag by the word head's own tag head (a `TaggedHead`)."""
        head = self.net.readouts()[1]
        if not isinstance(head, Tagging):
            raise ValueError("the word head has no tag head: train with context.conv.head tagged")
        tap = {False: Tap.WORD, True: Tap.CONTEXT}[head.reads_context]
        slots = torch.as_tensor(self(split)[tap], device=next(self.net.parameters()).device)
        with torch.no_grad():
            return head.readout()(slots).argmax(-1).cpu().numpy()

    def read(self, split: Split) -> tuple[Float[Tensor, "n v"], Float[Tensor, "n e"]]:  # devtools-ignore: test-mirror
        """Every word of `split`'s context-head log-probabilities and encoder embedding, in word order."""
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        device = split.index.device
        logp = torch.zeros(words + 1, VOCAB_SIZE, device=device)
        slots = torch.zeros(words + 1, Tap.WORD.width(self.net), device=device)
        self.net.eval()
        with torch.no_grad(), torch.autocast(Device.CUDA, dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                _, embedded, context, _ = self.net.heard(batch.inputs())
                at = torch.where(batch.index != PAD, batch.index, words)
                logp[at] = context.float().log_softmax(-1)
                slots[at] = embedded.float()
        return logp[:words], slots[:words]

    def tag_probabilities(self, slots: Float[Tensor, "n e"]) -> Float[Tensor, "n t"]:  # devtools-ignore: test-mirror
        """The word head's own tag head's probabilities over `CLASSES` for each embedding in `slots`."""
        head = self.net.readouts()[1]
        if not isinstance(head, Tagging):
            raise ValueError("the word head has no tag head: train with context.conv.head tagged")
        with torch.no_grad():
            return head.readout()(slots).softmax(-1)

    def __call__(self, split: Split) -> dict[Tap, Float[np.ndarray, "n _"]]:  # devtools-ignore: test-mirror
        """Every word of `split`'s encoder embedding (its own 0.5 s window) and span transformer output (its
        neighbours' MEG and timing too), in word order."""
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        device = split.index.device
        out = {tap: torch.zeros(words + 1, tap.width(self.net), device=device) for tap in Tap.pooled()}
        self.net.eval()
        with torch.no_grad(), torch.autocast(Device.CUDA, dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                refined, slots, _, _ = self.net.heard(batch.inputs())
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

    def scores(self, predicted: Int[np.ndarray, " n"]) -> dict[str, tuple[float, float, int]]:
        """Balanced accuracy over the tags present, accuracy and their denominator: every word, the 50's, the rest,
        and per domain."""
        subsets = {"all": np.ones(len(self.tags), bool), "in-50": self.vocabulary, "outside": ~self.vocabulary}
        subsets |= {str(name): self.domain == name for name in np.unique(self.domain)}
        return {
            name: (
                float(balanced_accuracy_score(self.tags[chosen], predicted[chosen])),
                float((self.tags[chosen] == predicted[chosen]).mean()),
                int(chosen.sum()),
            )
            for name, chosen in subsets.items()
            if chosen.any()
        }

    def recalls(self, predicted: Int[np.ndarray, " n"]) -> dict[str, tuple[float, int]]:
        """Each present tag's recall and its count of words."""
        return {
            CLASSES[tag]: (float((predicted[self.tags == tag] == tag).mean()), int((self.tags == tag).sum()))
            for tag in np.unique(self.tags)
        }

    def confusion(self, predicted: Int[np.ndarray, " n"], *, in_50: bool) -> pd.DataFrame:
        """Word counts of each true tag (row) predicted as each tag (column), over the 50's words or the rest."""
        chosen = self.vocabulary if in_50 else ~self.vocabulary
        present = np.union1d(self.tags[chosen], predicted[chosen])
        names = [CLASSES[tag] for tag in present]
        counts = confusion_matrix(self.tags[chosen], predicted[chosen], labels=present)
        return pd.DataFrame(counts, index=names, columns=names)

    def majority(self) -> float:
        """The accuracy of always predicting the most frequent tag."""
        return float(np.bincount(self.tags).max() / len(self.tags))

    def logged(self, name: str, predicted: Int[np.ndarray, " n"], *, confusions: bool) -> None:
        """Log the `scores` and `recalls` of `predicted` as `name`, then with `confusions` its confusion over the 50's
        words and over the rest."""
        scores = {k: (round(b, 4), round(a, 4), n) for k, (b, a, n) in self.scores(predicted).items()}
        LOG.info("%s (bacc, acc, n) %s", name, scores)
        LOG.info("%s recall %s", name, {k: (round(r, 3), n) for k, (r, n) in self.recalls(predicted).items()})
        if confusions:
            for in_50 in (True, False):
                LOG.info("%s confusion, in-50 %s:\n%s", name, in_50, self.confusion(predicted, in_50=in_50).to_string())


@dataclass(frozen=True)
class Table:
    """Every probed word's features, tag, run, membership in the 50, domain, sentence, onset and 50-way target."""

    features: dict[str, Float[np.ndarray, "n _"]]
    tags: Int[np.ndarray, " n"]
    runs: np.ndarray
    vocabulary: np.ndarray
    domain: np.ndarray
    sentences: Int[np.ndarray, " n"]
    onsets: Float[np.ndarray, " n"]
    targets: Int[np.ndarray, " n"]

    @classmethod
    def of(  # devtools-ignore: test-mirror
        cls, reader: Reader, embedded: Embedded, held: dict[str, list[Run]], contexts: ContextTags, timing: Timing
    ) -> "Table":
        pieces = []
        for name, runs in held.items():
            for run, (words, split) in reader.splits(runs).items():
                read = embedded(split)
                onsets = words[Column.ONSET].to_numpy(np.float64)
                tags = contexts.classes(words[Column.RUN].to_numpy(), onsets)
                in_50 = (words[Column.TARGET] >= 0).to_numpy()
                sentences = (run.name + "/" + words[Column.SENTENCE].astype(str)).to_numpy()
                targets = words[Column.TARGET].to_numpy(np.int64)
                features = timing.features(words)
                pieces.append(
                    (
                        features,
                        read[Tap.WORD],
                        read[Tap.CONTEXT],
                        tags,
                        run.name,
                        in_50,
                        name,
                        sentences,
                        onsets,
                        targets,
                    )
                )
        return cls.joined(pieces)

    @classmethod
    def joined(cls, pieces: list[Piece]) -> "Table":
        """The pieces' rows concatenated, each sentence numbered."""
        timing, word, context, tags, run, vocabulary, domain, sentences, onsets, targets = zip(*pieces, strict=True)
        repeat = [len(t) for t in tags]
        return cls(
            {
                "timing": np.concatenate(timing),
                "meg-word": np.concatenate(word),
                "meg-context": np.concatenate(context),
            },
            np.concatenate(tags),
            np.repeat(run, repeat),
            np.concatenate(vocabulary),
            np.repeat(domain, repeat),
            np.unique(np.concatenate(sentences), return_inverse=True)[1],
            np.concatenate(onsets),
            np.concatenate(targets),
        )

    def rows(self, runs: Shaped[np.ndarray, " m"], onsets: Float[np.ndarray, " m"]) -> Int[np.ndarray, " m"]:
        """Each (run, onset) word's row in this table, -1 where it has none: a word is named by its run and onset."""
        keys = pd.MultiIndex.from_arrays([self.runs, self.onsets])
        if not keys.is_unique:
            raise ValueError("a run holds two words at one onset")
        return keys.get_indexer(pd.MultiIndex.from_arrays([runs, onsets]))


@dataclass(frozen=True)
class Probe:
    """Classical probes fit on the training table and scored on the validation table, alone and stacked."""

    train: Table
    valid: Table
    balanced: bool = False

    @property
    def known(self) -> Bool[np.ndarray, " n"]:
        """The training words whose tag is known: an `UNKNOWN` word is neither fit nor stacked on."""
        return self.train.tags != UNKNOWN

    def model(self, name: str) -> HistGradientBoostingClassifier | TorchLogistic:
        """Gradient boosting for the few tabular timing features; a linear readout for an embedding. Each weighs the
        classes equally when `balanced`, by their frequency otherwise."""
        if name == "timing":
            return HistGradientBoostingClassifier(
                max_iter=200, early_stopping=False, random_state=0, class_weight="balanced" if self.balanced else None
            )
        return TorchLogistic(c=0.1, balanced=self.balanced)

    def probabilities(self, name: str) -> tuple[Float[np.ndarray, "m k"], Float[np.ndarray, "n k"]]:
        """`name`'s probe's log-probabilities: out-of-fold on the known training words (`m` of them), then fit on
        them all, on every validation word."""
        fit = self.model(name)
        x, y, runs = self.train.features[name][self.known], self.train.tags[self.known], self.train.runs[self.known]
        folded = cross_val_predict(fit, x, y, cv=GroupKFold(FOLDS), groups=runs, method="predict_proba")
        return np.log(folded + FLOOR), np.log(fit.fit(x, y).predict_proba(self.valid.features[name]) + FLOOR)

    def stacked(
        self, names: tuple[str, ...], probed: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[Int[np.ndarray, " k"], Float[np.ndarray, "n k"]]:
        """The tags a logistic stacker fit on `names`' out-of-fold `probed` log-probabilities knows, and its
        validation log-posterior over them."""
        inputs, outputs = zip(*(probed[name] for name in names), strict=True)
        stacker = TorchLogistic(standardize=False, balanced=self.balanced).fit(
            np.hstack(inputs), self.train.tags[self.known]
        )
        return stacker.classes_, stacker.predict_log_proba(np.hstack(outputs))

    def predicted(
        self, names: tuple[str, ...], probed: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> Int[np.ndarray, " n"]:
        """Each validation word's most probable tag on its own."""
        classes, posterior = self.stacked(names, probed)
        return classes[posterior.argmax(1)]

    def parsed(self, names: tuple[str, ...], probed: dict[str, tuple[np.ndarray, np.ndarray]]) -> Int[np.ndarray, " n"]:
        """Each validation sentence's most probable tag path under the training sentences' grammar, each word's
        emission its stacked posterior over the training tag prior."""
        classes, posterior = self.stacked(names, probed)
        train = np.searchsorted(classes, self.train.tags[self.known])
        grammar = Grammar.of(train, self.train.sentences[self.known], self.train.onsets[self.known], len(classes))
        prior = np.log(np.bincount(train, minlength=len(classes)) / len(train))
        return classes[grammar.decode(posterior - prior, self.valid.sentences, self.valid.onsets)]

    def report(self) -> dict[str, Int[np.ndarray, " n"]]:
        """Each arm's predicted validation tags, the best stack also parsed per sentence."""
        probed = {name: self.probabilities(name) for name in self.train.features}
        arms = (("timing",), ("meg-word",), ("meg-context",), ("timing", "meg-word"), ("timing", "meg-context"))
        predicted = {"+".join(arm): self.predicted(arm, probed) for arm in arms}
        return predicted | {"timing+meg-context+grammar": self.parsed(("timing", "meg-context"), probed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--stride", type=int, default=4, help="every stride-th subject-0 training run is probed on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    trained_in = root.models() / args.name
    data = DataConfig.shared([trained_in], DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False))
    runs = Run.discover(root.raw())
    trained = [run for run in data.trained(runs) if run.subject == DEEP][:: args.stride]
    contexts = ContextTags.load(ContextTags.path(root))
    timing = Timing.fitted(data.run_cache(root).table([run.name for run in trained]))
    embedded = Embedded.load(trained_in / "model.pt", device)
    train = Table.of(reader, embedded, {"train": trained}, contexts, timing)
    valid = Table.of(reader, embedded, reader.ruled(runs), contexts, timing)
    LOG.info("probing on %d words of %d runs, scoring %d words", len(train.tags), len(trained), len(valid.tags))
    known = valid.tags != UNKNOWN
    scored = Probed(valid.tags[known], valid.vocabulary[known], valid.domain[known])
    LOG.info("scoring %d known-tag words of %d, majority-tag accuracy %.4f", known.sum(), len(known), scored.majority())
    for balanced in (False, True):
        weighting = "balanced" if balanced else "natural"
        for arm, predicted in Probe(train, valid, balanced).report().items():
            scored.logged(f"{weighting} {arm}", predicted[known], confusions=arm.startswith("timing+meg-context"))


if __name__ == "__main__":
    main()
