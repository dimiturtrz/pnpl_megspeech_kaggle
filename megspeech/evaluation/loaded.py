"""A trained model, or several averaged as one, loaded to score as their training scored them."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Features, Split
from megspeech.data.cache.alignment import Alignment
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import Grouping
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Head, Score, SelectionConfig
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import AS_TRAINED, Fallback, Reading, SubjectLayer

RULER = "Sherlock1"
"""The validation task whose text the pooled runs repeat."""


@dataclass(frozen=True)
class Text:
    """One run's heard words (a word table with `sentence`, `position`, `word`, `onset`), keyed word by word: by its
    sentence, its place in that sentence and its lower-cased spelling, so two runs of one text match word for word."""

    words: pd.DataFrame

    @classmethod
    def of(cls, table: pd.DataFrame) -> "Text":
        """`table`'s words with their `position` in their sentence of their run."""
        return cls(table.assign(position=table.groupby(["run", "sentence"]).cumcount()))

    @property
    def keys(self) -> pd.MultiIndex:
        spelled = self.words["word"].astype(str).str.lower()
        return pd.MultiIndex.from_arrays([self.words["sentence"], self.words["position"], spelled])

    def matches(self, other: "Text") -> Int[np.ndarray, " n"]:
        """Each word's row in `other` with the same key, -1 where `other` has none."""
        return other.keys.get_indexer(self.keys)


@dataclass(frozen=True)
class Scoring:
    """How a loaded model is scored: spans cut by `grouping`, subjects read as `fallback` and presented as `reading`
    says, each scored set reshaped by `balance` and every window read moved by each of `shifts`."""

    grouping: Grouping = DataConfig.grouping
    fallback: Fallback = AS_TRAINED
    reading: Reading = Reading.OWN
    balance: Sinkhorn = Sinkhorn()
    shifts: tuple[int, ...] = (0,)
    aligned: bool = False
    """Re-colour every other subject's recording to subject 0's sensor covariance (`Alignment`) before the model
    reads it: +0.040 on the Broad ruler for a subject-0 model (0.499 -> 0.539, n=25322, megspeech-7gf.56)."""
    adapted: bool = False
    """Re-estimate batch-norm statistics on each scored split (`Evaluator.adapt`); the holdout's would read its own
    MEG, so a submission written this way needs a rules decision (megspeech-7gf.61)."""


@dataclass(frozen=True)
class LoadedModel:
    """Trained runs averaged as one model, with the prior and evaluator their training scored them by."""

    root: DataRoot
    model: nn.Module
    data: DataConfig
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]
    reading: Reading = Reading.OWN
    names: tuple[str, ...] = ()
    alignment: Alignment | None = None

    @classmethod
    def of(cls, root: DataRoot, names: list[str], scoring: Scoring, device: torch.device) -> "LoadedModel":
        """The trained runs behind `names` averaged as one model, scored as their training scored them: the prior of
        the draws they trained on, and the selection that counted their domains."""
        directories = [member for name in names for member in cls.members(root.models(), name)]
        model = Ensemble.load([directory / "model.pt" for directory in directories], scoring.fallback.unseen)
        model = model.to(device)
        for layer in (module for module in model.modules() if isinstance(module, SubjectLayer)):
            layer.shrink(scoring.fallback.keep, scoring.fallback.unseen)
        data = DataConfig.shared(directories, scoring.grouping)
        heard = Corpus.heard(data.run_cache(root), Run.everywhere(root), data)
        evaluator = Evaluator(
            device,
            Corpus.sequences(heard, scoring.grouping),
            SelectionConfig.shared(directories).min_novel,
            balance=scoring.balance,
            shifts=scoring.shifts,
            adapted=scoring.adapted,
        )
        prior = Corpus.prior(heard, data).to(device)
        alignment = data.alignment(root, device) if scoring.aligned or data.recoloured else None
        named = tuple(d.name for d in directories)
        return cls(root, model, data, evaluator, prior, scoring.reading, named, alignment)

    @staticmethod
    def members(models: Path, name: str) -> list[Path]:
        """The trained runs behind `name`: those an ensemble's record lists, else the run itself."""
        record = models / name / METRICS
        saved: dict[str, list[str]] = json.loads(record.read_text()) if record.exists() else {}
        return [models / member for member in saved.get("members", [name])]

    def load(self, runs: list[Run]) -> tuple[DeviceRecording, pd.DataFrame]:
        """`runs` on the device, and their words keyed by `Text` and cut as the model's grouping cuts them."""
        cache = self.data.run_cache(self.root)
        names = [run.name for run in runs]
        source = DeviceRecording.load(
            cache, self.evaluator.device, names, standardised=self.data.standardised, alignment=self.alignment
        )
        return source, Corpus.words(Text.of(source.words).words, self.data.grouping, self.data.rate)

    @property
    def features(self) -> Features:
        """What the model reads in place of MEG windows, if it trained on saved features."""
        return Features.under(self.root, self.data.features)

    def read(self, name: str, split: Split) -> Split:
        """`split` as the model reads it: `name`'s saved features if it trained on them, each word presented as
        `reading`'s subject."""
        split = self.features.split(name, split)
        return replace(split, subject=self.reading.subjects(split.subject))

    def splits(self, runs: list[Run]) -> dict[Run, tuple[pd.DataFrame, Split]]:
        """Each of `runs`, loaded together: its words and its split as the model reads it."""
        source, words = self.load(runs)
        rows = {run: words[words["run"] == run.name] for run in runs}
        return {
            run: (part, self.read(run.name, Corpus.split(source, part, self.data, np.ones(len(part)))))
            for run, part in rows.items()
        }

    def domains(self, held: dict[str, list[Run]]) -> tuple[dict[str, Split], pd.DataFrame]:
        """Subject 0's validation domains as the model reads them, and the ruler domain's words in its split's order."""
        source, words = self.load([run for group in held.values() for run in group])
        ruler = words[words["run"].isin([run.name for run in held.get(RULER, [])])]
        domains = Corpus.domains(source, words, held, self.data)
        return {name: self.read(name, split) for name, split in domains.items()}, ruler

    def context(self, split: Split) -> Float[Tensor, "n v"]:
        return self.evaluator.logits(self.model, split)[Head.CONTEXT]

    def labelled(self, split: Split) -> Score:
        """The context head on every labelled word of `split`."""
        labelled = split.labelled()
        return self.evaluator.evaluate(self.context(split)[labelled], split.y[labelled], self.log_prior)

    def unseen(self, run: Run) -> Score:
        """The context head on `run`'s unseen-span words, read from its own MEG."""
        _, split = self.splits([run])[run]
        return self.evaluator.unseen(self.context(split), split, self.log_prior)

    def report(self, runs: list[Run]) -> Epoch:
        """Every validation domain of `runs` scored alone, and the macro that selects."""
        domains, _ = self.domains(self.data.held(runs))
        return self.evaluator.report(self.model, domains, self.log_prior, 0, 0.0)

    def graded(self, runs: list[Run], tau: float) -> Graded:
        """Every counted validation domain of `runs`: its unseen-span words' hits at `tau`."""
        domains, _ = self.domains(self.data.held(runs))
        return self.evaluator.graded(self.model, domains, self.log_prior, tau)

    def probabilities(self, split: Split, tau: float) -> Float[np.ndarray, "n v"]:
        """Every word of `split`'s probabilities, logit-adjusted by `tau`: the context head's, but the word head's for
        a word alone in its span. The context head on a lone word scores near chance (bAcc@10 0.24 vs the word head's
        0.39 on Sherlock1, 0.23 vs 0.29 on TheMoth, n=1296/1216), and 92 of the Deep holdout's 960 rows are lone."""
        logits = self.evaluator.logits(self.model, split)
        chosen = torch.where(split.lone()[:, None], logits[Head.WORD], logits[Head.CONTEXT])
        adjusted = self.evaluator.adjusted(chosen, self.log_prior, tau)
        return torch.softmax(adjusted, dim=-1).cpu().numpy()
