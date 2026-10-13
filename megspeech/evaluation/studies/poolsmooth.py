"""Transductive repeat-aggregation proxy (megspeech-d6r.8): rows of one scored pool that are the same word should agree,
so each row's scores are mixed with its `k` nearest other rows of the pool, weighted `softmax(cos / temp)` over the
base model's embedding at the context head's input. The log mix is `s_i + lam * sum_j w_ij log_softmax(s_j)`; the
probability mix is `Mixing.LINEAR` over `sum_j w_ij p_j`. Pools are Kaggle-sized draws of a domain's scored rows (and
the whole domain); every knob is selected on val pools alone, and the local test is read at that setting and,
labelled as an oracle, at its own best.

    uv run python -m megspeech.evaluation.studies.poolsmooth ema-cd25-xsp10sem05-ac05x-s1-ft-s1 --device cpu
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.studies.knnreadout import FLOOR, SCORED, Builder, Dump, Embedder, Mixing, Rows, Space, Study
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.model.sentence import SentenceNet

LOG = logging.getLogger(__name__)

SCRATCH = "zz-tns"
POOL = 960
"""The Deep Kaggle test's size."""
DRAWS = 30
SEED = 0
KS = (5, 10, 20, 50)
TEMPERATURES = (0.02, 0.05, 0.1)
LAMBDAS = (0.1, 0.25, 0.5, 1.0)
QUERY_CHUNK = 256
BASE_SCORES = {Partition.VALIDATION: 0.6386, Partition.TEST: 0.7088}
"""The macro of the base this study was specified for, to reproduce before anything else."""
TOLERANCE = 0.002
RULER = Ruler()


class Aggregation(StrEnum):
    """Where the neighbours' scores are summed: in log-probabilities, or in probabilities."""

    LOG = "log"
    PROB = "prob"

    def combine(
        self,
        base: Float[Tensor, "n v"],
        log_readout: Float[Tensor, "n v"],
        prob_readout: Float[Tensor, "n v"],
        lam: float,
    ) -> Float[Tensor, "n v"]:
        if self is Aggregation.LOG:
            return base + lam * log_readout
        return Mixing.LINEAR.mix(base, prob_readout.clamp(min=FLOOR), lam)


@dataclass(frozen=True)
class Setting:
    space: Space
    k: int
    temperature: float
    aggregation: Aggregation
    lam: float


@dataclass(frozen=True)
class Draw:
    """Seeded uniform draws without replacement of `size` rows from a domain (the whole domain when `size` is 0)."""

    size: int = POOL
    count: int = DRAWS
    seed: int = SEED

    def indices(self, n: int) -> Int[Tensor, "d p"]:
        if not self.size:
            return torch.arange(n)[None]
        if n <= self.size:
            return torch.arange(n).repeat(self.count, 1)
        return torch.stack(
            [
                torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + i))[: self.size]
                for i in range(self.count)
            ]
        )


@dataclass(frozen=True)
class Near:
    """Each pool row's `K` nearest other rows, most similar first: their cosines and row numbers."""

    sims: Float[Tensor, "d p k"]
    at: Int[Tensor, "d p k"]

    def weights(self, k: int, temperature: float) -> Float[Tensor, "d p k"]:
        return (self.sims[..., :k] / temperature).softmax(dim=-1)

    def purity(self, y: Int[Tensor, "d p"], k: int) -> float:
        """Mean fraction of the first `k` neighbours that share their row's label."""
        labels = y.gather(1, self.at[..., :k].flatten(1)).view(self.at[..., :k].shape)
        return float((labels == y[..., None]).float().mean())


@dataclass(frozen=True)
class Pool:
    """`d` pools of `p` scored rows each: the base's log-probabilities, the embeddings, the labels."""

    base: Float[Tensor, "d p v"]
    emb: Float[Tensor, "d p h"]
    y: Int[Tensor, "d p"]

    @classmethod
    def drawn(cls, rows: Rows, draw: Draw) -> "Pool":
        at = draw.indices(len(rows.y))
        return cls(rows.base.log_softmax(-1)[at], rows.folds[0][at], rows.y[at])

    def near(self, space: Space, depth: int) -> Near:
        """The `depth` most cosine-similar other rows of each row, in `space` centred on its own pool's mean."""
        unit = torch.stack([space.unit(e, e.mean(dim=0)) for e in self.emb])
        rows = unit.shape[1]
        found = []
        for start in range(0, rows, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, rows)
            sims = unit[:, start:stop] @ unit.transpose(1, 2)
            sims[:, torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
            found.append(sims.topk(min(depth, rows - 1), dim=-1))
        return Near(torch.cat([s for s, _ in found], dim=1), torch.cat([a for _, a in found], dim=1))

    def readouts(self, near: Near, k: int, temperature: float) -> tuple[Float[Tensor, "d p v"], Float[Tensor, "d p v"]]:
        """The weighted neighbours' log-probabilities and probabilities."""
        weights = near.weights(k, temperature)
        gathered = self.base[torch.arange(len(self.base))[:, None, None], near.at[..., :k]]
        return (
            torch.einsum("dpk,dpkv->dpv", weights, gathered),
            torch.einsum("dpk,dpkv->dpv", weights, gathered.exp()),
        )

    def bacc(self, scores: Float[Tensor, "d p v"], calibration: Calibration, ruler: Ruler) -> Float[Tensor, " d"]:
        """BAcc@10 of each pool's `scores`, logit-adjusted as the scored set is."""
        adjusted = torch.stack([calibration.adjusted(s) for s in scores])
        hits = ruler.hits(adjusted.flatten(0, 1), self.y.flatten()).view(self.y.shape)
        return ruler.score(hits, self.y)

    def settings(
        self, calibration: Calibration, ruler: Ruler = RULER, ks: tuple[int, ...] = KS
    ) -> dict[Setting, Float[Tensor, " d"]]:
        """Every setting's BAcc@10 gain over the base on each pool."""
        base = self.bacc(self.base, calibration, ruler)
        out = {}
        for space in Space:
            near = self.near(space, max(ks))
            for k, temperature in itertools.product(ks, TEMPERATURES):
                log_readout, prob_readout = self.readouts(near, k, temperature)
                for aggregation, lam in itertools.product(Aggregation, LAMBDAS):
                    mixed = aggregation.combine(self.base, log_readout, prob_readout, lam)
                    out[Setting(space, k, temperature, aggregation, lam)] = self.bacc(mixed, calibration, ruler) - base
        return out


@dataclass(frozen=True)
class Gains:
    """Per-domain gains of every setting on every pool of a partition, and the macro over domains."""

    domains: dict[str, dict[Setting, Float[Tensor, " d"]]]

    def macro(self) -> dict[Setting, Float[Tensor, " d"]]:
        first = next(iter(self.domains.values()))
        return {s: torch.stack([g[s] for g in self.domains.values()]).mean(dim=0) for s in first}

    def best(self) -> Setting:
        macro = self.macro()
        return max(macro, key=lambda s: float(macro[s].mean()))

    def at(self, setting: Setting) -> dict[str, tuple[float, float]]:
        """Each domain's and the macro's gain mean and SD over pools at `setting`."""
        rows = {n: g[setting] for n, g in self.domains.items()} | {"macro": self.macro()[setting]}
        return {n: (float(v.mean()), float(v.std()) if len(v) > 1 else 0.0) for n, v in rows.items()}


@dataclass(frozen=True)
class Smoothing:
    """The dump's domains scored on pools of one draw."""

    dump: Dump

    def calibration(self) -> Calibration:
        return Calibration(self.dump.log_prior, self.dump.tau)

    def gains(self, partition: Partition, draw: Draw) -> Gains:
        calibration = self.calibration()
        return Gains({n: Pool.drawn(r, draw).settings(calibration) for n, r in self.dump.partitions[partition].items()})

    def purity(self, partition: Partition, draw: Draw) -> dict[str, dict[Space, dict[int, float]]]:
        """Neighbour purity per domain, space and `k`."""
        out: dict[str, dict[Space, dict[int, float]]] = {}
        for name, rows in self.dump.partitions[partition].items():
            pool = Pool.drawn(rows, draw)
            out[name] = {s: {k: pool.near(s, max(KS)).purity(pool.y, k) for k in KS} for s in Space}
        return out

    def chance(self, partition: Partition) -> dict[str, float]:
        shares = {
            n: torch.bincount(r.y, minlength=VOCAB_SIZE).float() / len(r.y)
            for n, r in self.dump.partitions[partition].items()
        }
        return {n: float((s**2).sum()) for n, s in shares.items()}


@dataclass(frozen=True)
class Collector:
    """Reads the base's val and local-test domains once: its context log-probabilities and its embeddings."""

    root: DataRoot
    scoring: Scoring
    device: torch.device

    @staticmethod
    def path(root: DataRoot, base: str) -> Path:
        return root.models() / SCRATCH / f"{base}.pt"

    def build(self, base: str) -> Dump:  # devtools-ignore: test-mirror
        loaded = Loader.of(Group(self.root, self.scoring, self.device), [base])
        net = cast("SentenceNet", cast("Ensemble", loaded.model).members[0])
        runs = Run.discover(self.root.raw())
        tau = loaded.report(runs).tau
        partitions = {}
        for partition in SCORED:
            held = replace(loaded, partition=partition)
            domains, _ = held.domains(held.ruled(runs))
            counted = held.evaluator.eligible(domains)
            contexts = {n: (s, held.context(s)) for n, s in counted.items()}
            Builder.check(held, runs, tau, contexts)
            partitions[partition] = {n: Builder.rows(s, c, held, [Embedder(net)]) for n, (s, c) in contexts.items()}
        return Dump((), partitions, loaded.log_prior.cpu(), tau)

    @staticmethod
    def reproduces(dump: Dump, expected: dict[Partition, float]) -> dict[Partition, float]:
        """The base's macro on each partition, which must sit within `TOLERANCE` of `expected`."""
        study = Study(dump)
        got = {p: study.ruler.point(study.base(p)) for p in SCORED}
        if any(abs(got[p] - expected[p]) > TOLERANCE for p in SCORED):
            raise ValueError(f"base scores {got}, expected {expected}")
        return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val", type=float, default=BASE_SCORES[Partition.VALIDATION])
    parser.add_argument("--test", type=float, default=BASE_SCORES[Partition.TEST])
    parser.add_argument("--redump", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    path = Collector.path(root, args.base)
    if args.redump or not path.exists():
        Collector(root, Scoring(fallback=Fallback(Unseen.DEEP)), torch.device(args.device)).build(args.base).save(path)
    dump = Dump.load(path)
    LOG.info(
        "base reproduces %s", Collector.reproduces(dump, {Partition.VALIDATION: args.val, Partition.TEST: args.test})
    )
    study = Smoothing(dump)
    pooled, whole = Draw(), Draw(size=0)
    val = study.gains(Partition.VALIDATION, pooled)
    test = study.gains(Partition.TEST, pooled)
    chosen, oracle = val.best(), test.best()
    record: dict[str, object] = {"chosen": str(chosen), "oracle": str(oracle)}
    for label, setting in (("val-chosen", chosen), ("oracle", oracle)):
        record[f"test-{label}"] = test.at(setting)
        LOG.info("local test pools %s %s: %s", label, setting, test.at(setting))
    record["val-at-chosen"] = val.at(chosen)
    record["test-full-pool-at-chosen"] = study.gains(Partition.TEST, whole).at(chosen)
    record["val-full-pool-at-chosen"] = study.gains(Partition.VALIDATION, whole).at(chosen)
    LOG.info("full pool at chosen: %s", record["test-full-pool-at-chosen"])
    for partition in SCORED:
        record[f"purity-{partition}"] = study.purity(partition, pooled)
        record[f"chance-{partition}"] = study.chance(partition)
        LOG.info("purity %s %s chance %s", partition, record[f"purity-{partition}"], record[f"chance-{partition}"])
    path.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
