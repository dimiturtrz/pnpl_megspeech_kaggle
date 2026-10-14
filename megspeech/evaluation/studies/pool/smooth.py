"""Transductive repeat-aggregation proxy (megspeech-d6r.8): rows of one scored pool that are the same word should agree,
so each row's scores are mixed with its `k` nearest other rows of the pool, weighted `softmax(cos / temp)` over the
base model's embedding at the context head's input. The log mix is `s_i + lam * sum_j w_ij log_softmax(s_j)`; the
probability mix is `Mixing.LINEAR` over `sum_j w_ij p_j`. Pools are Kaggle-sized draws of a domain's scored rows (and
the whole domain); every knob is selected on val pools alone, and the local test is read at that setting and,
labelled as an oracle, at its own best.

    uv run python -m megspeech.evaluation.studies.pool.smooth ema-cd25-xsp10sem05-ac05x-s1-ft-s1 --device cpu
"""

import argparse
import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.studies.pool.knn import FLOOR, SCORED, Builder, Dump, Embedder, Mixing, Rows, Space, Study
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
ORACLE_KS = (1, 3, 5, 10, 0)
"""Same-label neighbours mixed in; 0 is every one the pool holds."""
MATCH_K = 5
PURITIES = (0.2, 0.4, 0.6)


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
    grouped: bool = False
    """Draw whole spans (sentences) rather than single rows, as a Kaggle pool of complete sentences holds them."""

    def indices(self, n: int, span: Int[Tensor, " n"] | None = None) -> Int[Tensor, "d p"]:
        if not self.size:
            return torch.arange(n)[None]
        if n <= self.size:
            return torch.arange(n).repeat(self.count, 1)
        return torch.stack([self.pick(n, span, self.seed + i) for i in range(self.count)])

    def pick(self, n: int, span: Int[Tensor, " n"] | None, seed: int) -> Int[Tensor, " p"]:
        generator = torch.Generator().manual_seed(seed)
        if not self.grouped or span is None:
            return torch.randperm(n, generator=generator)[: self.size]
        rank = torch.randperm(int(span.max()) + 1, generator=generator)[span]
        return (rank * n + torch.arange(n)).argsort()[: self.size]


@dataclass(frozen=True)
class Near:
    """Each pool row's `K` nearest other rows, most similar first: their cosines and row numbers."""

    sims: Float[Tensor, "d p k"]
    at: Int[Tensor, "d p k"]

    def weights(self, k: int, temperature: float) -> Float[Tensor, "d p k"]:
        return (self.sims[..., :k] / temperature).softmax(dim=-1)

    def hits(self, y: Int[Tensor, "d p"], k: int) -> Float[Tensor, "d p"]:
        """Each row's fraction of its first `k` neighbours that share its label."""
        labels = y.gather(1, self.at[..., :k].flatten(1)).view(self.at[..., :k].shape)
        return (labels == y[..., None]).float().mean(dim=-1)

    def purity(self, y: Int[Tensor, "d p"], k: int) -> float:
        """Mean fraction of the first `k` neighbours that share their row's label."""
        return float(self.hits(y, k).mean())

    def by_label(self, y: Int[Tensor, "d p"], k: int) -> Float[Tensor, " v"]:
        """Purity per 50-way label, NaN for a label no pool row holds: a contrastive embedding trained on the corpus'
        frequencies can be pure on the frequent words alone, which the row mean of near-balanced pools hides."""
        counts = torch.bincount(y.flatten(), minlength=VOCAB_SIZE).float()
        sums = torch.zeros(VOCAB_SIZE).index_add_(0, y.flatten(), self.hits(y, k).flatten())
        return torch.where(counts > 0, sums / counts.clamp(min=1), torch.nan)


@dataclass(frozen=True)
class Pool:
    """`d` pools of `p` scored rows each: the base's log-probabilities, the embeddings, the labels."""

    base: Float[Tensor, "d p v"]
    emb: Float[Tensor, "d p h"]
    y: Int[Tensor, "d p"]

    @classmethod
    def drawn(cls, rows: Rows, draw: Draw, span: Int[Tensor, " n"] | None = None) -> "Pool":
        at = draw.indices(len(rows.y), span)
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

    def posterior(self) -> "Pool":
        """The same pool with each row embedded as the model's own probability vector."""
        return replace(self, emb=self.base.exp())

    def same(self) -> Bool[Tensor, "d p p"]:
        """Whether row `q` is another row of row `p`'s pool with its true label."""
        equal = self.y[:, :, None] == self.y[:, None, :]
        return equal & ~torch.eye(self.y.shape[1], dtype=torch.bool)

    @staticmethod
    def subset(mask: Bool[Tensor, "d p p"], count: int, generator: torch.Generator) -> Bool[Tensor, "d p p"]:
        """Up to `count` random true entries of each row of `mask`; every true entry when `count` is 0."""
        if not count:
            return mask
        scores = torch.rand(mask.shape, generator=generator).masked_fill(~mask, -torch.inf)
        top = scores.topk(min(count, mask.shape[-1]), dim=-1)
        return torch.zeros_like(mask).scatter(-1, top.indices, top.values > -torch.inf)

    def chosen(
        self, chosen: Bool[Tensor, "d p p"], calibration: Calibration, ruler: Ruler = RULER
    ) -> dict[tuple[Aggregation, float], Float[Tensor, " d"]]:
        """Gains of mixing each row with its `chosen` neighbours equally weighted; a row with none stays as it was."""
        present = chosen.any(dim=-1, keepdim=True)
        weights = chosen.float() / chosen.sum(dim=-1, keepdim=True).clamp(min=1)
        log_readout, prob_readout = weights @ self.base, weights @ self.base.exp()
        base = self.bacc(self.base, calibration, ruler)
        out = {}
        for aggregation, lam in itertools.product(Aggregation, LAMBDAS):
            mixed = torch.where(present, aggregation.combine(self.base, log_readout, prob_readout, lam), self.base)
            out[aggregation, lam] = self.bacc(mixed, calibration, ruler) - base
        return out


@dataclass(frozen=True)
class Gains[K]:
    """Per-domain gains of every setting `K` on every pool of a partition, and the macro over domains."""

    domains: dict[str, dict[K, Float[Tensor, " d"]]]

    def macro(self) -> dict[K, Float[Tensor, " d"]]:
        first = next(iter(self.domains.values()))
        return {s: torch.stack([g[s] for g in self.domains.values()]).mean(dim=0) for s in first}

    def best(self, keep: Callable[[K], bool] | None = None) -> K:
        """The setting of best mean macro gain among those `keep` admits (all by default)."""
        macro = {s: g for s, g in self.macro().items() if keep is None or keep(s)}
        return max(macro, key=lambda s: float(macro[s].mean()))

    def at(self, setting: K) -> dict[str, tuple[float, float]]:
        """Each domain's and the macro's gain mean and SD over pools at `setting`."""
        rows = {n: g[setting] for n, g in self.domains.items()} | {"macro": self.macro()[setting]}
        return {n: (float(v.mean()), float(v.std()) if len(v) > 1 else 0.0) for n, v in rows.items()}


@dataclass(frozen=True)
class Ceiling:
    """Macro gains per pool of mixes under a controlled neighbour source, by source setting and (mix, lam)."""

    gains: dict[int, dict[tuple[Aggregation, float], Float[Tensor, " d"]]]

    @staticmethod
    def macro(
        domains: list[dict[tuple[Aggregation, float], Float[Tensor, " d"]]],
    ) -> dict[tuple[Aggregation, float], Float[Tensor, " d"]]:
        return {m: torch.stack([d[m] for d in domains]).mean(dim=0) for m in domains[0]}

    def best(self) -> dict[int, tuple[tuple[Aggregation, float], float, float]]:
        """Per source setting, its best (mix, lam) by mean gain with that gain's mean and SD over pools."""
        out = {}
        for source, mixes in self.gains.items():
            mix = max(mixes, key=lambda m: float(mixes[m].mean()))
            out[source] = (mix, float(mixes[mix].mean()), float(mixes[mix].std()) if len(mixes[mix]) > 1 else 0.0)
        return out


@dataclass(frozen=True)
class Smoothing:
    """The dump's domains scored on pools of one draw."""

    dump: Dump

    def calibration(self) -> Calibration:
        return Calibration(self.dump.log_prior, self.dump.tau)

    def pools(self, partition: Partition, draw: Draw, *, posterior: bool = False) -> dict[str, Pool]:
        pools = {n: Pool.drawn(r, draw) for n, r in self.dump.partitions[partition].items()}
        return {n: p.posterior() if posterior else p for n, p in pools.items()}

    def gains(self, partition: Partition, draw: Draw, *, posterior: bool = False) -> Gains[Setting]:
        calibration = self.calibration()
        return Gains({n: p.settings(calibration) for n, p in self.pools(partition, draw, posterior=posterior).items()})

    def available(self, partition: Partition, draw: Draw) -> dict[str, float]:
        """Mean number of same-label other rows in a row's pool, per domain."""
        return {n: float(p.same().sum(dim=-1).float().mean()) for n, p in self.pools(partition, draw).items()}

    def ceiling(self, partition: Partition, draw: Draw, counts: tuple[int, ...] = ORACLE_KS) -> Ceiling:
        """Gains of mixing in `count` random same-label rows (0: all of them), the macro over domains."""
        calibration, generator = self.calibration(), torch.Generator().manual_seed(draw.seed)
        per_domain = [
            {c: p.chosen(Pool.subset(p.same(), c, generator), calibration) for c in counts}
            for p in self.pools(partition, draw).values()
        ]
        return Ceiling({c: Ceiling.macro([d[c] for d in per_domain]) for c in counts})

    def matched(self, partition: Partition, draw: Draw, purities: tuple[float, ...] = PURITIES) -> Ceiling:
        """Gains with `MATCH_K` neighbours of whom a `purity` share are random same-label rows, the rest random
        other-label rows, keyed by the number of same-label ones."""
        calibration, generator = self.calibration(), torch.Generator().manual_seed(draw.seed)
        wanted = {round(p * MATCH_K) for p in purities}
        per_domain = []
        for pool in self.pools(partition, draw).values():
            same, different = pool.same(), ~pool.same() & ~torch.eye(pool.y.shape[1], dtype=torch.bool)
            per_domain.append(
                {
                    n: pool.chosen(
                        Pool.subset(same, n, generator) | Pool.subset(different, MATCH_K - n, generator), calibration
                    )
                    for n in wanted
                }
            )
        return Ceiling({n: Ceiling.macro([d[n] for d in per_domain]) for n in wanted})

    def purity(
        self, partition: Partition, draw: Draw, *, posterior: bool = False
    ) -> dict[str, dict[Space, dict[int, float]]]:
        """Neighbour purity per domain, space and `k`."""
        out: dict[str, dict[Space, dict[int, float]]] = {}
        for name, pool in self.pools(partition, draw, posterior=posterior).items():
            out[name] = {s: {k: pool.near(s, max(KS)).purity(pool.y, k) for k in KS} for s in Space}
        return out

    def by_label(self, partition: Partition, draw: Draw) -> dict[str, dict[Space, list[float]]]:
        """Purity at `MATCH_K` per domain, space and 50-way label (`Near.by_label`)."""
        return {
            name: {s: pool.near(s, max(KS)).by_label(pool.y, MATCH_K).tolist() for s in Space}
            for name, pool in self.pools(partition, draw).items()
        }

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
    def path(root: DataRoot, base: str, reader: str = "") -> Path:
        return root.models() / SCRATCH / (f"{base}--{reader}.pt" if reader else f"{base}.pt")

    def build(self, base: str, reader: str = "") -> Dump:  # devtools-ignore: test-mirror
        """The base's rows; with a `reader` model, that model's scores and embeddings of the base's very rows."""
        group = Group(self.root, self.scoring, self.device)
        loaded = Loader.of(group, [base])
        reading = Loader.of(group, [reader]) if reader else loaded
        net = cast("SentenceNet", cast("Ensemble", reading.model).members[0])
        runs = Run.discover(self.root.raw())
        tau = loaded.report(runs).tau
        partitions = {}
        for partition in SCORED:
            held = replace(loaded, partition=partition)
            domains, _ = held.domains(held.ruled(runs))
            counted = held.evaluator.eligible(domains)
            viewing = replace(reading, partition=partition)
            contexts = {n: (s, viewing.context(s)) for n, s in counted.items()}
            if not reader:
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
        record[f"purity-by-label-{partition}"] = study.by_label(partition, pooled)
        LOG.info("purity %s %s chance %s", partition, record[f"purity-{partition}"], record[f"chance-{partition}"])
    record["available"] = {p: study.available(p, pooled) for p in SCORED}
    record["oracle-label"] = {p: study.ceiling(p, pooled).best() for p in SCORED}
    record["purity-matched"] = {p: study.matched(p, pooled).best() for p in SCORED}
    LOG.info("available %s", record["available"])
    LOG.info("oracle-label ceiling (best mix by test, k 0=all) %s", record["oracle-label"])
    LOG.info("purity-matched (key = same-label of %d) %s", MATCH_K, record["purity-matched"])
    val_post, test_post = (
        study.gains(Partition.VALIDATION, pooled, posterior=True),
        study.gains(Partition.TEST, pooled, posterior=True),
    )
    chosen_post = val_post.best()
    record["posterior-chosen"] = str(chosen_post)
    record["posterior-val-at-chosen"] = val_post.at(chosen_post)
    record["posterior-test-at-chosen"] = test_post.at(chosen_post)
    record["posterior-test-macro-at-chosen"] = float(test_post.macro()[chosen_post].mean())
    record["posterior-oracle"] = str(test_post.best())
    record["posterior-test-at-oracle"] = test_post.at(test_post.best())
    for partition in SCORED:
        record[f"posterior-purity-{partition}"] = study.purity(partition, pooled, posterior=True)
    LOG.info("posterior chosen %s test %s", chosen_post, record["posterior-test-at-chosen"])
    LOG.info("posterior purity %s", {p: record[f"posterior-purity-{p}"] for p in SCORED})
    path.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
