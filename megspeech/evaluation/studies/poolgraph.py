"""Round 3 of transductive pool smoothing (megspeech-d6r.8): raise the purity of the neighbours a row is smoothed with.

Four ways to a purer pool graph, each the same mix over the same 960-row pools as `poolsmooth`: mutual kNN (an edge
needs both rows to list each other), two views (embedding and posterior cosine, z-scored and summed), propagation
(the smoothing applied for several rounds over the one graph), and an ensemble (the base and the cross-fit fold models,
which never trained on a scored session, vote on who the neighbours are; the scores mixed stay the base's). Settings
are chosen on VAL pools, the local test reports; then tau is re-selected on VAL under each arm's chosen setting.

    uv run python -m megspeech.evaluation.studies.poolgraph <base> [--folds kf5ac05-f0-ft ...]
"""

import argparse
import itertools
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Calibration
from megspeech.evaluation.studies.crowding import TAUS
from megspeech.evaluation.studies.knnreadout import Dump
from megspeech.evaluation.studies.poolconfuser import Confusers
from megspeech.evaluation.studies.poollayout import Layout
from megspeech.evaluation.studies.poolsmooth import (
    BASE_SCORES,
    LAMBDAS,
    RULER,
    TEMPERATURES,
    Aggregation,
    Collector,
    Draw,
    Gains,
    Pool,
    Smoothing,
)
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

NEIGHBOURS = (5, 10, 20)
PROPAGATED_NEIGHBOURS = (5, 10)
PROPAGATION_ROUNDS = (2, 3)
PURITY_K = 5
FOLDS = tuple(f"kf5ac05-f{i}-ft" for i in range(5))
OUT = "poolgraph.json"


class Source(StrEnum):
    """What two rows' similarity is read from."""

    EMBEDDING = "embedding"
    POSTERIOR = "posterior"
    TWO_VIEWS = "two-views"
    ENSEMBLE_EMBEDDING = "ensemble-embedding"
    ENSEMBLE_POSTERIOR = "ensemble-posterior"
    ENSEMBLE_BOTH = "ensemble-both"


@dataclass(frozen=True)
class Views:
    """A pool as the base sees it, and the same rows as each cross-fit fold model sees them."""

    pool: Pool
    folds: tuple[Pool, ...] = ()
    span: Int[Tensor, "d p"] | None = None
    place: Int[Tensor, "d p"] | None = None

    @staticmethod
    def cosine(emb: Float[Tensor, "d p h"]) -> Float[Tensor, "d p p"]:
        """Cosine of every row pair in each pool, centred on the pool's mean."""
        unit = functional.normalize(emb - emb.mean(dim=1, keepdim=True), dim=-1)
        return unit @ unit.transpose(1, 2)

    @staticmethod
    def zscored(sims: Float[Tensor, "d p p"]) -> Float[Tensor, "d p p"]:
        """Each pool's similarities standardised over its off-diagonal pairs."""
        off = ~torch.eye(sims.shape[1], dtype=torch.bool)
        count = off.sum()
        mean = (sims * off).sum(dim=(1, 2), keepdim=True) / count
        var = (((sims - mean) * off) ** 2).sum(dim=(1, 2), keepdim=True) / count
        return (sims - mean) / var.sqrt()

    def embedding(self) -> Float[Tensor, "d p p"]:
        return self.cosine(self.pool.emb)

    def posterior(self) -> Float[Tensor, "d p p"]:
        return self.cosine(self.pool.base.exp())

    def two_views(self) -> Float[Tensor, "d p p"]:
        return self.zscored(self.embedding()) + self.zscored(self.posterior())

    def ensemble_embedding(self) -> Float[Tensor, "d p p"]:
        """The mean over the base and the folds of each model's own embedding cosine."""
        return torch.stack([self.cosine(p.emb) for p in (self.pool, *self.folds)]).mean(dim=0)

    def ensemble_posterior(self) -> Float[Tensor, "d p p"]:
        """The cosine of the base's and the folds' mean probability vector."""
        return self.cosine(torch.stack([p.base.exp() for p in (self.pool, *self.folds)]).mean(dim=0))

    def ensemble_both(self) -> Float[Tensor, "d p p"]:
        return self.zscored(self.ensemble_embedding()) + self.zscored(self.ensemble_posterior())

    def similarity(self, source: Source, *, apart: bool = False) -> Float[Tensor, "d p p"]:
        """Every row pair's similarity under `source`, minus infinity on a row with itself, and on a row of its own
        span when `apart`."""
        sims = {
            Source.EMBEDDING: self.embedding,
            Source.POSTERIOR: self.posterior,
            Source.TWO_VIEWS: self.two_views,
            Source.ENSEMBLE_EMBEDDING: self.ensemble_embedding,
            Source.ENSEMBLE_POSTERIOR: self.ensemble_posterior,
            Source.ENSEMBLE_BOTH: self.ensemble_both,
        }[source]()
        blocked = torch.eye(sims.shape[1], dtype=torch.bool)
        if apart:
            blocked = blocked | (self.span[:, :, None] == self.span[:, None, :])
        return sims.masked_fill(blocked, -torch.inf)


@dataclass(frozen=True)
class Graph:
    """Which rows each row listens to, and how much: `weights` rows sum to 1, or to 0 for a row with no neighbour."""

    keep: Bool[Tensor, "d p p"]
    weights: Float[Tensor, "d p p"]

    @classmethod
    def of(cls, sims: Float[Tensor, "d p p"], k: int, temperature: float, *, mutual: bool) -> "Graph":
        """Each row's `k` most similar others, only the ones that list the row back when `mutual`."""
        nearest = sims.topk(k, dim=-1).indices
        keep = torch.zeros_like(sims, dtype=torch.bool).scatter(-1, nearest, value=True) & sims.isfinite()
        if mutual:
            keep = keep & keep.transpose(1, 2)
        weights = (sims / temperature).masked_fill(~keep, -torch.inf).softmax(dim=-1).nan_to_num(nan=0.0)
        return cls(keep, weights)

    def purity(self, y: Tensor) -> tuple[float, float]:
        """The share of kept edges joining rows of one label, and the share of rows with an edge."""
        same = y[:, :, None] == y[:, None, :]
        return float((self.keep & same).sum() / self.keep.sum().clamp(min=1)), float(self.keep.any(-1).float().mean())

    def mixed(
        self, base: Float[Tensor, "d p v"], aggregation: Aggregation, lams: tuple[float, ...]
    ) -> Float[Tensor, "d p v"]:
        """`base` smoothed over this graph once per weight in `lams`; a row with no neighbour stays as it was."""
        present = self.keep.any(dim=-1, keepdim=True)
        state = base
        for lam in lams:
            combined = aggregation.combine(state, self.weights @ state, self.weights @ state.exp(), lam)
            state = torch.where(present, combined, state).log_softmax(dim=-1)
        return state


@dataclass(frozen=True)
class Config:
    source: Source
    mutual: bool
    k: int
    temperature: float
    aggregation: Aggregation
    lams: tuple[float, ...]
    apart: bool = False

    @property
    def rounds(self) -> int:
        return len(self.lams)


@dataclass(frozen=True)
class Arm:
    """A named family of configs, the one a round-3 idea searches over."""

    name: str
    sources: tuple[Source, ...]
    rounds: tuple[int, ...] = (1,)
    mutual: tuple[bool, ...] = (False,)
    apart: bool = False
    twin: str = ""
    """The non-ensemble arm whose chosen setting this arm is also read at."""

    @property
    def leaked(self) -> bool:
        """Whether a source is a fold model whose epoch was selected on VAL rows, so VAL selection is not clean."""
        return bool(ENSEMBLES & set(self.sources))

    def admits(self, config: Config) -> bool:
        return (
            config.source in self.sources
            and config.rounds in self.rounds
            and config.mutual in self.mutual
            and config.apart is self.apart
        )

    def configs(self) -> Iterable[Config]:
        ks = NEIGHBOURS if self.rounds == (1,) else PROPAGATED_NEIGHBOURS
        paths = [p for r in self.rounds for p in itertools.product(LAMBDAS, repeat=r)]
        for source, mutual, k, temperature, aggregation, lams in itertools.product(
            self.sources, self.mutual, ks, TEMPERATURES, Aggregation, paths
        ):
            yield Config(source, mutual, k, temperature, aggregation, lams, self.apart)


ENSEMBLES = {Source.ENSEMBLE_EMBEDDING, Source.ENSEMBLE_POSTERIOR, Source.ENSEMBLE_BOTH}
EITHER = (False, True)
ARMS = (
    Arm("ref: kNN embedding", (Source.EMBEDDING,)),
    Arm("ref: kNN posterior", (Source.POSTERIOR,)),
    Arm("1 mutual kNN, embedding", (Source.EMBEDDING,), mutual=(True,)),
    Arm("1 mutual kNN, posterior", (Source.POSTERIOR,), mutual=(True,)),
    Arm("2 two views", (Source.TWO_VIEWS,), mutual=EITHER),
    Arm("3 propagation, embedding", (Source.EMBEDDING,), PROPAGATION_ROUNDS, EITHER),
    Arm("3 propagation, two views", (Source.TWO_VIEWS,), PROPAGATION_ROUNDS, EITHER),
    Arm("4 ensemble, embedding", (Source.ENSEMBLE_EMBEDDING,), mutual=EITHER, twin="ref: kNN embedding"),
    Arm("4 ensemble, posterior", (Source.ENSEMBLE_POSTERIOR,), mutual=EITHER, twin="ref: kNN posterior"),
    Arm("4 ensemble, both", (Source.ENSEMBLE_BOTH,), mutual=EITHER, twin="2 two views"),
    Arm(
        "4 ensemble, both, propagation",
        (Source.ENSEMBLE_BOTH,),
        PROPAGATION_ROUNDS,
        EITHER,
        twin="3 propagation, two views",
    ),
    Arm("7 other spans only, embedding", (Source.EMBEDDING,), mutual=EITHER, apart=True),
    Arm("7 other spans only, posterior", (Source.POSTERIOR,), mutual=EITHER, apart=True),
    Arm("7 other spans only, two views", (Source.TWO_VIEWS,), mutual=EITHER, apart=True),
)


@dataclass(frozen=True)
class Study:
    """The base's pools, and the fold models' views of the same pools, over one dump each."""

    base: Smoothing
    folds: tuple[Smoothing, ...] = ()
    layouts: dict[Partition, dict[str, Layout]] = field(default_factory=dict)

    def views(self, partition: Partition, draw: Draw) -> dict[str, Views]:
        out = {}
        for name, rows in self.base.dump.partitions[partition].items():
            layout = self.layouts.get(partition, {}).get(name, Layout.alone(len(rows.y)))
            at = draw.indices(len(rows.y), layout.span)
            folds = tuple(Pool.drawn(f.dump.partitions[partition][name], draw, layout.span) for f in self.folds)
            out[name] = Views(Pool.drawn(rows, draw, layout.span), folds, layout.span[at], layout.place[at])
        return out

    def calibration(self, tau: float | None = None) -> Calibration:
        return Calibration(self.base.dump.log_prior, self.base.dump.tau if tau is None else tau)

    @staticmethod
    def grouped(configs: Iterable[Config]) -> dict[tuple[Source, bool], dict[tuple[bool, int, float], list[Config]]]:
        """The configs by the similarity and the graph they share, so that each is built once."""
        out: dict[tuple[Source, bool], dict[tuple[bool, int, float], list[Config]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for c in configs:
            out[c.source, c.apart][c.mutual, c.k, c.temperature].append(c)
        return out

    def gains(self, partition: Partition, draw: Draw, configs: Sequence[Config], ruler: Ruler = RULER) -> Gains[Config]:
        """Every config's BAcc@10 gain over the base on each pool of each domain."""
        calibration, grouped = self.calibration(), self.grouped(configs)
        domains = {}
        for name, views in self.views(partition, draw).items():
            base = views.pool.bacc(views.pool.base, calibration, ruler)
            out = {}
            for (source, apart), graphs in grouped.items():
                sims = views.similarity(source, apart=apart)
                for (mutual, k, temperature), members in graphs.items():
                    graph = Graph.of(sims, k, temperature, mutual=mutual)
                    for c in members:
                        mixed = graph.mixed(views.pool.base, c.aggregation, c.lams)
                        out[c] = views.pool.bacc(mixed, calibration, ruler) - base
            domains[name] = out
        return Gains(domains)

    def purity(
        self, partition: Partition, draw: Draw, config: Config, k: int = PURITY_K
    ) -> dict[str, tuple[float, float]]:
        """Per domain, the share of `k`-neighbour edges of `config`'s similarity and mutuality that join one label and
        the share of rows with any edge."""
        return {
            name: Graph.of(views.similarity(config.source, apart=config.apart), k, 1.0, mutual=config.mutual).purity(
                views.pool.y
            )
            for name, views in self.views(partition, draw).items()
        }

    def confusers(self, partition: Partition, draw: Draw, source: Source, k: int = PURITY_K) -> Confusers:
        """Where the `k` nearest neighbours sit, averaged over the domains."""
        parts = []
        for views in self.views(partition, draw).values():
            keep = Graph.of(views.similarity(source), k, 1.0, mutual=False).keep
            parts.append(Confusers.of(keep, views.span, views.place, views.pool.y))
        return Confusers.mean(parts)

    def retuned(
        self, partition: Partition, draw: Draw, config: Config, taus: Sequence[float], ruler: Ruler = RULER
    ) -> dict[float, tuple[float, float]]:
        """Macro BAcc@10 by tau of `config`'s smoothed scores, and of the base's, over the draws."""
        smoothed: dict[float, list[Tensor]] = defaultdict(list)
        plain: dict[float, list[Tensor]] = defaultdict(list)
        for views in self.views(partition, draw).values():
            graph = Graph.of(
                views.similarity(config.source, apart=config.apart), config.k, config.temperature, mutual=config.mutual
            )
            mixed = graph.mixed(views.pool.base, config.aggregation, config.lams)
            for tau in taus:
                calibration = self.calibration(tau)
                smoothed[tau].append(views.pool.bacc(mixed, calibration, ruler))
                plain[tau].append(views.pool.bacc(views.pool.base, calibration, ruler))
        return {
            tau: (
                float(torch.stack(smoothed[tau]).mean()),
                float(torch.stack(plain[tau]).mean()),
            )
            for tau in taus
        }

    def taus(self) -> tuple[float, ...]:
        return tuple(sorted({*TAUS, self.base.dump.tau}))


@dataclass(frozen=True)
class Report:
    """One arm's VAL-chosen config, how it did on the local test, and how pure its graph was."""

    arm: str
    chosen: Config
    val_gain: float
    test: dict[str, tuple[float, float]]
    purity: dict[str, tuple[float, float]]
    tau: dict[str, float]
    leaked: bool = False
    twin_test: dict[str, tuple[float, float]] | None = None
    """The local test at the twin arm's chosen setting with this arm's source, for a leaked arm."""

    @staticmethod
    def macro(per_domain: dict[str, tuple[float, float]]) -> tuple[float, float]:
        values = torch.tensor(list(per_domain.values()))
        return float(values[:, 0].mean()), float(values[:, 1].mean())

    def line(self) -> str:
        purity, coverage = self.macro(self.purity)
        twin = "" if self.twin_test is None else f" | at twin setting {self.twin_test['macro'][0]:+.4f}"
        return (
            f"{self.arm:34s}{' [VAL LEAKED]' if self.leaked else ''} val {self.val_gain:+.4f} chosen {self.chosen} "
            f"| test {self.test['macro'][0]:+.4f} +- {self.test['macro'][1]:.4f} "
            f"| purity@{PURITY_K} {purity:.3f} (rows with an edge {coverage:.2f}) | tau {self.tau}{twin}"
        )


@dataclass(frozen=True)
class Experiment:
    """Selects each arm on VAL pools and reports it on the local test's."""

    study: Study
    draw: Draw = Draw()

    def retune(self, config: Config) -> dict[str, float]:
        """Re-select tau on VAL under `config`; the local test's smoothed score against the base at both taus."""
        taus = self.study.taus()
        val = self.study.retuned(Partition.VALIDATION, self.draw, config, taus)
        tau = max(val, key=lambda t: val[t][0])
        test = self.study.retuned(Partition.TEST, self.draw, config, [self.study.base.dump.tau, tau])
        fixed, base = test[self.study.base.dump.tau]
        return {
            "tau": tau,
            "gain at fixed tau": fixed - base,
            "gain at retuned tau, same-tau base": test[tau][0] - test[tau][1],
            "gain at retuned tau, fixed-tau base": test[tau][0] - base,
        }

    def report(self, arm: Arm, val: Gains[Config]) -> Report:
        chosen = val.best(arm.admits)
        test = self.study.gains(Partition.TEST, self.draw, [chosen]).at(chosen)
        purity = self.study.purity(Partition.TEST, self.draw, chosen)
        return Report(arm.name, chosen, val.at(chosen)["macro"][0], test, purity, self.retune(chosen), arm.leaked)

    def twinned(self, report: Report, reports: dict[str, Report]) -> Report:
        """`report` of a leaked arm, with the local test read at its twin's chosen setting and this arm's source."""
        twin = next((a.twin for a in ARMS if a.name == report.arm), "")
        if not twin:
            return report
        setting = replace(reports[twin].chosen, source=report.chosen.source)
        return replace(report, twin_test=self.study.gains(Partition.TEST, self.draw, [setting]).at(setting))

    def run(self) -> list[Report]:
        configs = list(dict.fromkeys(c for arm in ARMS for c in arm.configs()))
        val = self.study.gains(Partition.VALIDATION, self.draw, configs)
        reports = [self.report(arm, val) for arm in ARMS]
        named = {r.arm: r for r in reports}
        return [self.twinned(r, named) for r in reports]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--folds", nargs="*", default=list(FOLDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grouped", action="store_true", help="draw whole spans into each pool")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    collector = Collector(root, Scoring(fallback=Fallback(Unseen.DEEP)), torch.device(args.device))
    dumps = []
    for reader in ("", *args.folds):
        path = Collector.path(root, args.base, reader)
        if not path.exists():
            collector.build(args.base, reader).save(path)
        dumps.append(Dump.load(path))
    LOG.info("base reproduces %s", Collector.reproduces(dumps[0], BASE_SCORES))
    layout = Layout.path(root, args.base)
    if not layout.exists():
        Layout.save(Layout.build(root, collector.scoring, collector.device, args.base), layout)
    study = Study(Smoothing(dumps[0]), tuple(Smoothing(d) for d in dumps[1:]), Layout.load(layout))
    draw = Draw(grouped=args.grouped)
    for source, drawn in itertools.product(
        (Source.EMBEDDING, Source.POSTERIOR), (Draw(), Draw(grouped=True), Draw(size=0))
    ):
        LOG.info(
            "confusers k=%d %s %s test: %s", PURITY_K, source, drawn, study.confusers(Partition.TEST, drawn, source)
        )
    reports = Experiment(study, draw).run()
    for report in reports:
        LOG.info(report.line())
    Collector.path(root, args.base).with_name(f"grouped-{OUT}" if args.grouped else OUT).write_text(
        json.dumps([vars(r) for r in reports], indent=1, default=str)
    )


if __name__ == "__main__":
    main()
