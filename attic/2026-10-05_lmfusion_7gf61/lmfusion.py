"""Oracle-context LM fusion (megspeech-7gf.61, stage 1): the peer protocol's frozen base times an LM prior over the 50
words given the TRUE words around each word, an upper bound on what a decoder that must guess its neighbours can add.

`lam` of each arm is tuned on the out-of-fold rows, then every arm is scored once on the official validation split.
Arms: `unigram` (the LM's sentence-start frequencies: the prior with no context), `left` (p(w | left words)), `pmi`
(left minus unigram: context evidence without the frequency prior the base already has), `right1` and `right2`
(p(w | left) times the likelihood of the next 1 or 2 true words given w).

    uv run python -m megspeech.evaluation.studies.lmfusion --folds kf5t-f0 kf5t-f1 kf5t-f2 kf5t-f3 kf5t-f4 \\
        --base peerbase-s0 --lm gpt2-large
"""

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.labels.vocabulary import IGNORE, PRIMARY
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.boosted import Rows
from megspeech.evaluation.studies.lmscore import LanguageModel
from megspeech.evaluation.studies.lmtext import RIGHT, Context, Sentences
from megspeech.evaluation.studies.peer import OUT, Dump, Study
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

CANDIDATES = tuple(word.replace("’", "'") for word in PRIMARY.words)
GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
"""Weights of the LM prior (`lam`) tried: its log-probabilities are added to the base's, times `lam`."""
TOP = 10
LOG_EVERY = 2000


@dataclass(frozen=True)
class Arm:
    """One fusion: the evidence channel of `Priors.evidence` it adds, with the unigram taken out when `pmi`."""

    name: str
    channel: int
    pmi: bool = False


ARMS = (Arm("unigram", 0), Arm("left", 1), Arm("pmi", 1, pmi=True), Arm("right1", 2), Arm("right2", 3))


@dataclass(frozen=True)
class Priors:
    """The LM scores of the 50 per distinct context: `table[u, c, j]` is word `c`'s log-probability with the first
    `j` following words, and `unigram` the same with no context at all."""

    keys: tuple[Context, ...]
    table: Float[Tensor, "u c j"]
    unigram: Float[Tensor, " c"]

    @staticmethod
    def path(root: DataRoot, lm: str) -> Path:
        return root.models() / OUT / f"lmprior-{lm}.pt"

    def save(self, path: Path) -> None:  # devtools-ignore: test-mirror
        torch.save({"keys": self.keys, "table": self.table, "unigram": self.unigram}, path)

    @classmethod
    def load(cls, path: Path) -> "Priors":  # devtools-ignore: test-mirror
        saved = torch.load(path)
        return cls(tuple(saved["keys"]), saved["table"], saved["unigram"])

    @classmethod
    def empty(cls, unigram: Float[Tensor, " c"]) -> "Priors":
        return cls((), unigram.new_zeros(0, len(unigram), RIGHT + 1), unigram)

    @staticmethod
    def scores(lm: LanguageModel, key: Context) -> Float[Tensor, "c j"]:  # devtools-ignore: test-mirror
        """`lm`'s scores of the candidates in `key`, the columns past its last following word repeating the last."""
        left, right = key
        raw = lm.score(left, CANDIDATES, right).cpu()
        return torch.cat([raw, raw[:, -1:].expand(-1, RIGHT + 1 - raw.shape[1])], dim=1)

    def extended(self, lm: LanguageModel, wanted: Iterable[Context]) -> "Priors":  # devtools-ignore: test-mirror
        """These priors and the `wanted` contexts they lack, scored by `lm`."""
        missing = [key for key in dict.fromkeys(wanted) if key not in set(self.keys)]
        fresh: list[Tensor] = []
        for count, key in enumerate(missing):
            fresh.append(self.scores(lm, key))
            if count % LOG_EVERY == 0:
                LOG.info("scored %d of %d new contexts", count, len(missing))
        table = torch.cat([self.table, torch.stack(fresh)]) if fresh else self.table
        return Priors((*self.keys, *missing), table, self.unigram)

    def evidence(self, contexts: Sequence[Context]) -> Float[Tensor, "n c k"]:
        """Each row's four channels, each normalised over the 50: the unigram, then left, +1 right word, +2."""
        where = {key: i for i, key in enumerate(self.keys)}
        rows = self.table[torch.tensor([where[key] for key in contexts])].permute(0, 2, 1)
        unigram = self.unigram.expand(len(contexts), 1, -1)
        return torch.cat([unigram, rows], dim=1).log_softmax(-1).permute(0, 2, 1)


@dataclass(frozen=True)
class Fusion:
    """Base log-probabilities of some rows, those with a label also carrying LM `evidence` (the rest stay the base)."""

    base: Float[Tensor, "n c"]
    y: Int[Tensor, " n"]
    evidence: Float[Tensor, "n c k"]
    unigram: Float[Tensor, " c"]
    scored: Int[Tensor, " n"]
    """The rows `evidence` is for, as row indices of `base`."""

    def fused(self, arm: Arm, lam: float) -> Float[Tensor, "n c"]:
        """`base` plus `lam` times `arm`'s evidence on the scored rows."""
        add = self.evidence[:, :, arm.channel] - (self.unigram if arm.pmi else 0.0)
        out = self.base.clone()
        out[self.scored] = out[self.scored] + lam * add
        return out.log_softmax(-1)

    @classmethod
    def of(cls, rows: Rows, contexts: Sequence[Context], take: Int[Tensor, " k"], priors: Priors) -> "Fusion":
        """`rows`, the `take` rows of them scored with `priors` at their `contexts`."""
        return cls(rows.base, rows.y, priors.evidence(contexts), priors.unigram, take)

    @staticmethod
    def pick(y: Int[Tensor, " n"], most: int, seed: int) -> Int[Tensor, " k"]:
        """At most `most` of the labelled rows of `y`, drawn once from `seed`."""
        labelled = torch.nonzero(y != IGNORE).flatten()
        order = torch.randperm(len(labelled), generator=torch.Generator().manual_seed(seed))
        return labelled[order[:most]].sort().values

    def accuracy(self, arm: Arm, lam: float) -> float:
        """The BAcc@10 of the fused scored rows."""
        ruler = Ruler(k=TOP)
        labels = self.y[self.scored]
        return ruler.balanced(ruler.hits(self.fused(arm, lam)[self.scored], labels), labels)


@dataclass(frozen=True)
class Experiment:
    """The dumps, the true text of their rows and the official ruler."""

    dump: Dump
    study: Study
    oof_text: Sentences
    val_text: Sentences

    @classmethod
    def build(  # devtools-ignore: test-mirror
        cls, root: DataRoot, folds: Sequence[str], base: str, device: torch.device
    ) -> "Experiment":
        loaded = Loader.of(root, [base], Scoring(official=True), device)
        domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
        dump = Dump.load(Dump.path(root, base))
        study = Study(dump.oof, dump.val, dump.domains, loaded.evaluator, loaded.log_prior, domains, device)
        oof = [cls.fold_text(root, name, device) for name in folds]
        directories = Ensemble.trained(root.models(), base)
        data = DataConfig.shared(directories, Scoring(official=True).grouping)
        held = [run for runs in loaded.ruled(Run.discover(root.raw())).values() for run in runs]
        val_text = Sentences.of(Reader(root, data, device, official=True), held)
        oof_text = Sentences.joined(oof)
        experiment = cls(dump, study, oof_text, val_text)
        experiment.check()
        return experiment

    @staticmethod
    def fold_text(root: DataRoot, name: str, device: torch.device) -> Sentences:  # devtools-ignore: test-mirror
        data = DataConfig.shared([root.models() / name], DataConfig.grouping)
        return Sentences.of(Reader(root, data, device), data.split(Run.everywhere(root))[Partition.VALIDATION])

    def check(self) -> None:  # devtools-ignore: test-mirror
        """The text rows are the dump's rows: same count and the same vocabulary label at every row."""
        for text, rows, kind in ((self.oof_text, self.dump.oof, "oof"), (self.val_text, self.dump.val, "val")):
            share = text.agrees(rows.y.numpy()) if len(text) == len(rows.y) else 0.0
            LOG.info("%s: %d text rows, %d dump rows, label agreement %.5f", kind, len(text), len(rows.y), share)
            if share < 1.0:
                raise ValueError(f"{kind} text rows are not the dump's rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--lm", default="gpt2-large")
    parser.add_argument("--max-oof", type=int, default=40000, help="labelled out-of-fold rows lam is tuned on")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    experiment = Experiment.build(root, args.folds, args.base, torch.device("cpu"))
    dump = experiment.dump
    oof_take = Fusion.pick(dump.oof.y, args.max_oof, args.seed)
    val_take = Fusion.pick(dump.val.y, len(dump.val.y), args.seed)
    oof_contexts, val_contexts = experiment.oof_text.contexts(oof_take), experiment.val_text.contexts(val_take)
    wanted = [*oof_contexts, *val_contexts]
    path = Priors.path(root, args.lm)
    lm = LanguageModel.load(root, args.lm, torch.device("cuda"))
    unigram = lm.score((), CANDIDATES)[:, 0].cpu()
    priors = (Priors.load(path) if path.exists() else Priors.empty(unigram)).extended(lm, wanted)
    priors.save(path)
    del lm
    tuning = Fusion.of(dump.oof, oof_contexts, oof_take, priors)
    scoring = Fusion.of(dump.val, val_contexts, val_take, priors)
    base = experiment.study.official(dump.val.base)
    results: list[dict[str, object]] = [{"arm": "base", "official": base, "oof": tuning.accuracy(ARMS[0], 0.0)}]
    LOG.info("base official %.4f oof(sub) %.4f", base, results[0]["oof"])
    for arm in ARMS:
        oof = {lam: tuning.accuracy(arm, lam) for lam in GRID}
        official = {lam: experiment.study.official(scoring.fused(arm, lam)) for lam in GRID}
        chosen = max(oof, key=oof.__getitem__)
        LOG.info(
            "%s lam* %.2f oof %.4f official %.4f | official by lam %s",
            arm.name,
            chosen,
            oof[chosen],
            official[chosen],
            official,
        )
        results.append({"arm": arm.name, "lam": chosen, "oof": oof, "official": official, "at_lam": official[chosen]})
    (path.parent / f"lmfusion-{args.lm}-results.json").write_text(json.dumps(results, indent=1))
    LOG.info("%d distinct contexts", len(priors.keys))


if __name__ == "__main__":
    main()
