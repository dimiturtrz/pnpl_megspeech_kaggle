"""Cross-fitted kNN readout (kNN-LM, Khandelwal et al. 2020, arXiv 1911.00172) for the Deep track (megspeech-d6r.6).

Datastore of fold `f`: every labelled word of the training sessions fold model `f` held out, embedded by that model at
the context head's input (the span transformer's output at the word), so each training word is embedded only by the
model that never trained on it. Query: every scored (unseen-span) word of val and the local test, embedded by each fold
model; its `k` nearest keys by cosine vote `softmax(cos / temperature)` per label, the fold votes averaged into
`p_knn`. The base model's context-head log-probabilities are mixed with `p_knn`, linearly or log-linearly at weight
`lam` (1 is `p_knn` alone), then logit-adjusted at `tau` as `megspeech.submission --local` scores. A class-mean
(prototype) readout of the same datastores is the control. Every knob is selected on val; the local test is read once,
at the selected setting.

    uv run python -m megspeech.evaluation.studies pool.knn ema-cd25-xsp10sem05-ac05x-s1-ft-s1 \\
        --folds kf5ac05-f0-ft kf5ac05-f1-ft kf5ac05-f2-ft kf5ac05-f3-ft kf5ac05-f4-ft [--grouping kaggle]
"""

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.reader import Reader
from megspeech.evaluation.metric.ruler import Graded, Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.scoring.score import ADJUSTMENTS
from megspeech.evaluation.studies.readout.ioi import Scored
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.layers.centring import RunCentring

LOG = logging.getLogger(__name__)

OUT = "knn"
FLOOR = 1e-6
"""Probability floor of a label no neighbour votes for: keeps its log finite and below every voted label."""
QUERY_CHUNK = 1024
KS = (8, 16, 32, 64, 128, 256)
"""Neighbours voting, from a few to a fifth of a mean class's ~1.4k keys per fold datastore."""
TEMPERATURES = (0.02, 0.05, 0.1, 0.2)
"""Softmax temperatures over cosine similarities in [-1, 1]: near-argmax to near-uniform over the k."""
LAMBDAS = (0.1, 0.2, 0.3, 0.5, 1.0)
"""Weights of the readout in the mix; kNN-LM's best is 0.25, 1 is the readout alone."""
SCORED = (Partition.VALIDATION, Partition.TEST)


@dataclass(frozen=True)
class Embedder:
    """A sentence model read at its context head's input, for every word of a split, as `Evaluator.sweep` reads it."""

    net: SentenceNet

    def embed(self, split: Split) -> Float[Tensor, "n h"]:
        """Every word of `split`'s refined position, in word order; run means gathered first as `sweep` gathers."""
        self.net.eval()
        with RunCentring.calibrated(self.net, lambda: self.swept(split)):
            return self.swept(split)

    def swept(self, split: Split) -> Float[Tensor, "n h"]:
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        out = torch.zeros(words + 1, self.net.width, device=split.y.device)
        with torch.no_grad(), Device.autocast():
            for spans in torch.arange(len(split.index), device=split.y.device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                out[batch.words_at(words)] = self.net.heard(batch.inputs())[0].float()
        return out[:words]


class Space(StrEnum):
    """Where cosine is read: the raw embeddings, or centred on the datastore's mean first."""

    RAW = "raw"
    CENTRED = "centred"

    def unit(self, x: Float[Tensor, "n h"], mean: Float[Tensor, " h"]) -> Float[Tensor, "n h"]:
        return functional.normalize(x - mean * (self is Space.CENTRED), dim=-1)


@dataclass(frozen=True)
class Neighbours:
    """Each query's `K` most similar keys, most similar first: their cosines and labels."""

    sims: Float[Tensor, "q k"]
    labels: Int[Tensor, "q k"]

    def vote(self, k: int, temperature: float) -> Float[Tensor, "q v"]:
        """`p_knn`: the first `k` neighbours' `softmax(cos / temperature)` summed per label, floored."""
        weights = (self.sims[:, :k] / temperature).softmax(dim=-1)
        votes = torch.zeros(len(self.sims), VOCAB_SIZE).scatter_add_(1, self.labels[:, :k], weights)
        return votes.clamp(min=FLOOR)


@dataclass(frozen=True)
class Store:
    """One fold's datastore: the embeddings of the labelled words it held out, and their labels."""

    x: Float[Tensor, "m h"]
    y: Int[Tensor, " m"]

    def neighbours(self, queries: Float[Tensor, "q h"], k: int, space: Space) -> Neighbours:
        mean = self.x.mean(dim=0)
        keys = space.unit(self.x, mean)
        k = min(k, len(keys))
        found = [(space.unit(chunk, mean) @ keys.T).topk(k, dim=-1) for chunk in queries.split(QUERY_CHUNK)]
        sims = torch.cat([sims for sims, _ in found])
        return Neighbours(sims, self.y[torch.cat([at for _, at in found])])

    def prototypes(self, queries: Float[Tensor, "q h"], space: Space) -> Float[Tensor, "q v"]:
        """Each query's cosine to every label's mean unit key; 0 for a label the store lacks."""
        mean = self.x.mean(dim=0)
        keys = space.unit(self.x, mean)
        sums = torch.zeros(VOCAB_SIZE, keys.shape[1]).index_add_(0, self.y, keys)
        return space.unit(queries, mean) @ functional.normalize(sums, dim=-1).T


class Mixing(StrEnum):
    """How the base's log-probabilities and a readout's probabilities combine at weight `lam` of the readout."""

    LINEAR = "linear"
    LOGLINEAR = "loglinear"

    def mix(self, base: Float[Tensor, "n v"], readout: Float[Tensor, "n v"], lam: float) -> Float[Tensor, "n v"]:
        """Log-probabilities (log-linear: unnormalised, which top-k and the adjustment ignore)."""
        if self is Mixing.LINEAR:
            return torch.log((1 - lam) * base.exp() + lam * readout)
        return (1 - lam) * base + lam * readout.log()


class Readout(StrEnum):
    KNN = "knn"
    PROTOTYPE = "prototype"


@dataclass(frozen=True)
class Setting:
    readout: Readout
    space: Space
    k: int
    temperature: float
    mixing: Mixing
    lam: float
    tau: float


@dataclass(frozen=True)
class Rows:
    """One scored domain's unseen-span words: the base's log-probabilities, the labels and each fold's embeddings."""

    base: Float[Tensor, "n v"]
    y: Int[Tensor, " n"]
    folds: tuple[Float[Tensor, "n h"], ...]


@dataclass(frozen=True)
class Dump:
    """Everything the study scores, read once from the models: the datastores, each partition's domains, the base's
    prior and val tau."""

    stores: tuple[Store, ...]
    partitions: dict[Partition, dict[str, Rows]]
    log_prior: Float[Tensor, " v"]
    tau: float

    @staticmethod
    def path(root: DataRoot, base: str, grouping: str) -> Path:
        return root.models() / OUT / f"{base}-{grouping}.pt"

    def save(self, path: Path) -> None:  # devtools-ignore: test-mirror
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> "Dump":  # devtools-ignore: test-mirror
        saved = torch.load(path, weights_only=False)
        return cls(
            tuple(Store(**store) for store in saved["stores"]),
            {
                Partition(p): {name: Rows(r["base"], r["y"], tuple(r["folds"])) for name, r in rows.items()}
                for p, rows in saved["partitions"].items()
            },
            saved["log_prior"],
            saved["tau"],
        )


@dataclass(frozen=True)
class Builder:
    """Reads the dump from the models: the base as `megspeech.submission --local` loads it, each fold as it trained."""

    root: DataRoot
    scoring: Scoring
    device: torch.device
    sessions: int = 0
    """Held-out sessions read per fold datastore; 0 reads them all (fewer is a development subset)."""

    def fold(self, name: str) -> tuple[Embedder, list[Run]]:  # devtools-ignore: test-mirror
        """Fold `name`'s model and the training sessions it held out (`DataConfig.carved`)."""
        directory = self.root.models() / name
        ensemble = cast("Ensemble", self.scoring.network([directory], DataConfig.saved(directory).rate, self.device))
        net = cast("SentenceNet", ensemble.members[0])
        data = DataConfig.shared([directory], self.scoring.grouping)
        held = data.carved(Run.discover(self.root.raw()))
        return Embedder(net), held[: self.sessions or len(held)]

    def store(self, embedder: Embedder, name: str, held: list[Run]) -> Store:  # devtools-ignore: test-mirror
        """The labelled words of `held`, run by run, embedded by `embedder`."""
        directory = self.root.models() / name
        data = DataConfig.shared([directory], self.scoring.grouping)
        reader = Reader(self.root, data, self.device, self.scoring.reading, data.geometry(self.root, self.device))
        xs: list[Float[Tensor, "m h"]] = []
        ys: list[Int[Tensor, " m"]] = []
        for run in held:
            _, split = reader.splits([run])[run]
            labelled = split.labelled()
            xs.append(embedder.embed(split)[labelled].cpu())
            ys.append(split.y[labelled].cpu())
        LOG.info("%s: %d held-out sessions, %d labelled words", name, len(held), sum(len(y) for y in ys))
        return Store(torch.cat(xs), torch.cat(ys))

    def build(self, base: str, folds: list[str]) -> Dump:  # devtools-ignore: test-mirror
        """The base's val and local-test domains as `LocalTest` scores them, each fold's datastore and embeddings."""
        loaded = Loader.of(Group(self.root, self.scoring, self.device), [base])
        runs = Run.discover(self.root.raw())
        tau = loaded.report(runs).tau
        partitions = {partition: self.scored(loaded, partition, runs, tau) for partition in SCORED}
        embedders = [self.fold(name) for name in folds]
        Builder.disjoint([[run.name for run in held] for _, held in embedders], Builder.ruled(loaded, runs))
        stores = tuple(self.store(e, name, held) for (e, held), name in zip(embedders, folds, strict=True))
        rows = {
            partition: {
                name: self.rows(split, context, loaded, [e for e, _ in embedders])
                for name, (split, context) in domains.items()
            }
            for partition, domains in partitions.items()
        }
        return Dump(stores, rows, loaded.log_prior.cpu(), tau)

    def scored(  # devtools-ignore: test-mirror
        self, loaded: ExportedModel, partition: Partition, runs: list[Run], tau: float
    ) -> dict[str, tuple[Split, Float[Tensor, "n v"]]]:
        """`partition`'s eligible domains with the base's context logits, checked against `LocalTest`."""
        held = replace(loaded, partition=partition)
        domains, _ = held.domains(held.ruled(runs))
        counted = {name: (split, held.context(split)) for name, split in held.evaluator.eligible(domains).items()}
        self.check(held, runs, tau, counted)
        return counted

    @staticmethod
    def ruled(loaded: ExportedModel, runs: list[Run]) -> set[str]:  # devtools-ignore: test-mirror
        """Every session the base scores on, over all `SCORED` partitions."""
        return {
            run.name for p in SCORED for group in replace(loaded, partition=p).ruled(runs).values() for run in group
        }

    @staticmethod
    def rows(  # devtools-ignore: test-mirror
        split: Split, context: Float[Tensor, "n v"], loaded: ExportedModel, embedders: list[Embedder]
    ) -> Rows:
        novel = split.novel(loaded.evaluator.seen)
        return Rows(context[novel].cpu(), split.y[novel].cpu(), tuple(e.embed(split)[novel].cpu() for e in embedders))

    @staticmethod
    def check(  # devtools-ignore: test-mirror
        held: ExportedModel, runs: list[Run], tau: float, domains: dict[str, tuple[Split, Float[Tensor, "n v"]]]
    ) -> None:
        """The base read here grades exactly as `LocalTest` grades it."""
        theirs = held.graded(runs, tau)
        mine = held.evaluator.grade(
            {n: c for n, (_, c) in domains.items()}, {n: s for n, (s, _) in domains.items()}, held.calibration(tau)
        )
        if Ruler().point(theirs) != Ruler().point(mine):
            raise ValueError(f"base rows grade {Ruler().point(mine)}, LocalTest {Ruler().point(theirs)}")

    @staticmethod
    def disjoint(folds: list[list[str]], scored: set[str]) -> None:
        """No session is held out by two folds, and none is one the base scores on."""
        names = [name for held in folds for name in held]
        if len(names) != len(set(names)) or scored & set(names):
            raise ValueError("fold datastores overlap each other or the scored sessions")


@dataclass(frozen=True)
class Study:
    """Every setting of the readout scored over a dump's partitions."""

    dump: Dump
    ruler: Ruler = Ruler()

    def reads(self, partition: Partition, space: Space) -> dict[str, tuple[list[Neighbours], list[Tensor]]]:
        """Each domain's per-fold neighbours (the most any `KS` asks for) and prototype cosines."""
        return {
            name: (
                [s.neighbours(q, max(KS), space) for s, q in zip(self.dump.stores, rows.folds, strict=True)],
                [s.prototypes(q, space) for s, q in zip(self.dump.stores, rows.folds, strict=True)],
            )
            for name, rows in self.dump.partitions[partition].items()
        }

    @staticmethod
    def readout(
        reads: tuple[list[Neighbours], list[Tensor]], readout: Readout, k: int, temperature: float
    ) -> Float[Tensor, "n v"]:
        """The fold readouts' mean probabilities."""
        neighbours, cosines = reads
        if readout is Readout.KNN:
            return torch.stack([n.vote(k, temperature) for n in neighbours]).mean(0)
        return torch.stack([(c / temperature).softmax(dim=-1) for c in cosines]).mean(0)

    def graded(self, partition: Partition, scores: dict[str, Float[Tensor, "n v"]], tau: float) -> Graded:
        calibration = Calibration(self.dump.log_prior, tau)
        rows = self.dump.partitions[partition]
        return Graded({n: (self.ruler.hits(calibration.adjusted(s), rows[n].y), rows[n].y) for n, s in scores.items()})

    def base(self, partition: Partition) -> Graded:
        return self.graded(partition, {n: r.base for n, r in self.dump.partitions[partition].items()}, self.dump.tau)

    def settings(self, partition: Partition) -> dict[Setting, Graded]:
        """Every setting's grades on `partition`: each readout, space, k, temperature, mixing, weight and tau."""
        out = {}
        rows = self.dump.partitions[partition]
        for space in Space:
            reads = self.reads(partition, space)
            for readout, k, temperature in itertools.product(Readout, KS, TEMPERATURES):
                if readout is Readout.PROTOTYPE and k != KS[0]:
                    continue
                p = {name: Study.readout(reads[name], readout, k, temperature) for name in rows}
                for mixing, lam, tau in itertools.product(Mixing, LAMBDAS, ADJUSTMENTS):
                    mixed = {name: mixing.mix(rows[name].base, p[name], lam) for name in rows}
                    out[Setting(readout, space, k, temperature, mixing, lam, tau)] = self.graded(partition, mixed, tau)
        return out

    def select(self, graded: dict[Setting, Graded], readout: Readout, *, alone: bool) -> Setting:
        """The best-scoring setting of `readout`, alone (`lam` 1) or mixed."""
        chosen = [s for s in graded if s.readout is readout and (s.lam == 1.0) == alone]
        return max(chosen, key=lambda s: self.ruler.point(graded[s]))

    def at(self, partition: Partition, setting: Setting) -> Graded:
        """`setting`'s grades on `partition` alone, without the rest of the grid."""
        rows = self.dump.partitions[partition]
        reads = self.reads(partition, setting.space)
        return self.graded(
            partition,
            {
                n: setting.mixing.mix(
                    rows[n].base, Study.readout(reads[n], setting.readout, setting.k, setting.temperature), setting.lam
                )
                for n in rows
            },
            setting.tau,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="the full-data model whose probabilities are mixed")
    parser.add_argument("--folds", nargs="+", required=True, help="cross-fit fold models of the base's recipe")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sessions", type=int, default=0, help="held-out sessions per fold datastore (0: all)")
    parser.add_argument("--redump", action="store_true", help="read the dump from the models again")
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    scoring = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    path = Dump.path(root, args.base + (f"-s{args.sessions}" if args.sessions else ""), scoring.grouping)
    if args.redump or not path.exists():
        Builder(root, scoring, torch.device(args.device), args.sessions).build(args.base, args.folds).save(path)
    study = Study(Dump.load(path))
    graded = study.settings(Partition.VALIDATION)
    base = {p: study.base(p) for p in SCORED}
    LOG.info("base val tau %s: %s", study.dump.tau, Scored(base[Partition.VALIDATION]).score())
    record: dict[str, object] = {"grouping": str(scoring.grouping), "tau": study.dump.tau}
    for readout, alone in itertools.product(Readout, (True, False)):
        setting = study.select(graded, readout, alone=alone)
        LOG.info("val %s alone=%s %s: %s", readout, alone, setting, Scored(graded[setting]).score())
        record[f"{readout}-{'alone' if alone else 'mix'}"] = {
            "setting": asdict(setting),
            "val": study.ruler.point(graded[setting]),
        }
    selected = study.select(graded, Readout.KNN, alone=False)
    test = study.at(Partition.TEST, selected)
    delta = study.ruler.interval(study.ruler.delta(test, base[Partition.TEST]))
    LOG.info("base local test tau %s: %s", study.dump.tau, Scored(base[Partition.TEST]).score())
    LOG.info("kNN mix local test %s: %s delta [%.4f, %.4f]", selected, Scored(test).score(), delta.low, delta.high)
    record["test"] = {"base": study.ruler.point(base[Partition.TEST]), "knn": study.ruler.point(test)}
    record["base_val"] = study.ruler.point(base[Partition.VALIDATION])
    path.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
