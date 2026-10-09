"""Whether a finish fit on labelled validation words beats the tau-only finish: a crowding penalty, a finer tau, a
head mix.

The current finish ranks `context - tau * log_prior`, tau one of `ADJUSTMENTS`, picked on validation. The error anatomy
(interpretations/deep/2026-10-08_error_anatomy.md) found a handful of frequent words in the top 10 of 40% of all rows,
while the metric rewards each class's own recall. The crowding finish takes a class's top-10 inclusion rate on the
FITTING words (label-free, but read on labelled validation, never on the Kaggle test) and penalises it:
`- gamma * log(rate / (TOP / VOCAB))`. Each arm carries at most three global parameters, never a free per-class vector.

Protocol (pre-registered): validation's novel words are cut into two halves, each domain on its own, in alternating
`BLOCK`-word blocks. Every arm is fit on one half and graded on the other, then swapped, so every word is graded by
parameters that never saw it. The delta to the tau-only finish carries a paired block bootstrap. The local test only
reports, with every arm refit on all of validation. A candidate needs >= +0.010 on the held-out halves with its
interval's low end above 0, and the same sign on the local test.

The lone arm grades the words alone in their span (the Kaggle test's isolated rows, which take the word head) and mixes
the context head into the word head there.

    uv run python -m megspeech.evaluation.studies.crowding m11xsp7hub6d2v3
"""

import argparse
import itertools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.metric.ruler import Graded, Ruler
from megspeech.evaluation.scoring.probabilities import META, PROBS, Probabilities, Words
from megspeech.evaluation.scoring.score import ADJUSTMENTS, MIN_NOVEL
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

TOP = 10
TAUS = tuple(round(0.1 * step, 1) for step in range(13))
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MIXES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
BLOCK = 32
"""Consecutive words that go to the same half, so a sentence rarely straddles the cut."""
RESAMPLE_BLOCK = 10
VAL = "val"
TEST = "local-test"
OUT = "crowding.json"


@dataclass(frozen=True)
class Rows:
    """One domain's scored words: the ensemble's mean log-probabilities from the graded head and the other head."""

    head: Float[Tensor, "n v"]
    other: Float[Tensor, "n v"]
    labels: Int[Tensor, " n"]

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, keep: Bool[Tensor, " n"]) -> "Rows":
        return Rows(self.head[keep], self.other[keep], self.labels[keep])

    def swapped(self) -> "Rows":
        """The same words graded on the other head."""
        return Rows(self.other, self.head, self.labels)

    def halves(self, seed: int) -> Int[Tensor, " n"]:
        """Each word's half (0 or 1): `BLOCK`-word blocks, a seeded half of the blocks to each side."""
        blocks = -(-len(self) // BLOCK)
        side = torch.randperm(blocks, generator=torch.Generator().manual_seed(seed)) % 2
        return side.repeat_interleave(BLOCK)[: len(self)]

    @classmethod
    def mean(cls, words: Sequence[Words], keep: Bool[Tensor, " n"]) -> "Rows":
        """The members' `words` averaged in log-probability space (the ensemble's geometric mean), `keep` rows."""
        head = torch.stack([member.logits.float().log_softmax(-1) for member in words]).mean(0)
        other = torch.stack([member.word_logits.float().log_softmax(-1) for member in words]).mean(0)
        return cls(head[keep], other[keep], words[0].labels[keep])


@dataclass(frozen=True)
class Finish:
    """`head + mix * other - tau * log_prior - gamma * crowding`; `crowding` is read on the words it was fit on."""

    tau: float
    gamma: float = 0.0
    mix: float = 0.0
    crowding: Float[Tensor, " v"] | None = None

    def scores(self, rows: Rows, log_prior: Float[Tensor, " v"]) -> Float[Tensor, "n v"]:
        penalty = torch.zeros(VOCAB_SIZE) if self.crowding is None else self.crowding
        return rows.head + self.mix * rows.other - self.tau * log_prior - self.gamma * penalty

    def inclusion(self, domains: Sequence[Rows], log_prior: Float[Tensor, " v"]) -> Float[Tensor, " v"]:
        """Log of each class's add-one top-`TOP` rate over `domains`' words relative to the even share `TOP/VOCAB`,
        under this finish without its penalty."""
        plain = replace(self, gamma=0.0, crowding=None)
        top = torch.cat([plain.scores(rows, log_prior).topk(TOP, dim=1).indices.flatten() for rows in domains])
        words = sum(len(rows) for rows in domains)
        rate = (torch.bincount(top, minlength=VOCAB_SIZE).float() + 1) / (words + VOCAB_SIZE / TOP)
        return (rate * VOCAB_SIZE / TOP).log()

    def fitted(self, domains: Sequence[Rows], log_prior: Float[Tensor, " v"]) -> "Finish":
        """This finish with its crowding read on `domains`."""
        return replace(self, crowding=self.inclusion(domains, log_prior))

    def label(self) -> str:
        return f"tau {self.tau:.1f} gamma {self.gamma:.2f} mix {self.mix:.2f}"


@dataclass(frozen=True)
class Arm:
    """A family of finishes; fitting picks the member with the best macro over the fitting domains."""

    name: str
    grid: tuple[Finish, ...]

    @classmethod
    def all(cls) -> tuple["Arm", ...]:
        """The tau-only baseline first, then the arms graded against it."""
        return (
            cls("tau3", tuple(Finish(tau) for tau in ADJUSTMENTS)),
            cls("tau", tuple(Finish(tau) for tau in TAUS)),
            cls("crowd", tuple(Finish(tau, gamma) for tau, gamma in itertools.product(TAUS, GAMMAS))),
            cls("headmix", tuple(Finish(tau, 0.0, mix) for tau, mix in itertools.product(TAUS, MIXES))),
        )

    def fit(self, domains: dict[str, Rows], log_prior: Float[Tensor, " v"], ruler: Ruler) -> Finish:
        fitted = [finish.fitted(list(domains.values()), log_prior) for finish in self.grid]
        return max(fitted, key=lambda finish: Grading.point(finish, domains, log_prior, ruler))


@dataclass(frozen=True)
class Grading:
    """Every word's hit under some finish, by domain."""

    hits: dict[str, Bool[Tensor, " n"]]

    @staticmethod
    def point(finish: Finish, domains: dict[str, Rows], log_prior: Float[Tensor, " v"], ruler: Ruler) -> float:
        hits = {name: ruler.hits(finish.scores(rows, log_prior), rows.labels) for name, rows in domains.items()}
        return ruler.point(Graded({name: (hits[name], rows.labels) for name, rows in domains.items()}))

    @classmethod
    def crossfit(cls, arm: Arm, domains: dict[str, Rows], log_prior: Float[Tensor, " v"], seed: int) -> "Grading":
        """Every word graded by `arm` fit on the other half of its domain."""
        ruler = Ruler(k=TOP)
        side = {name: rows.halves(seed) for name, rows in domains.items()}
        hits = {name: torch.zeros(len(rows), dtype=torch.bool) for name, rows in domains.items()}
        for half in (0, 1):
            fit = {name: rows.take(side[name] != half) for name, rows in domains.items()}
            finish = arm.fit(fit, log_prior, ruler)
            LOG.info("%s half %d fit %s", arm.name, half, finish.label())
            for name, rows in domains.items():
                held = side[name] == half
                hits[name][held] = ruler.hits(finish.scores(rows.take(held), log_prior), rows.labels[held])
        return cls(hits)

    @classmethod
    def transfer(
        cls, arm: Arm, fit: dict[str, Rows], graded: dict[str, Rows], log_prior: Float[Tensor, " v"]
    ) -> "Grading":
        """`graded`'s words under `arm` fit on all of `fit`."""
        ruler = Ruler(k=TOP)
        finish = arm.fit(fit, log_prior, ruler)
        LOG.info("%s full fit %s", arm.name, finish.label())
        return cls({name: ruler.hits(finish.scores(rows, log_prior), rows.labels) for name, rows in graded.items()})

    def graded(self, domains: dict[str, Rows]) -> Graded:
        return Graded({name: (self.hits[name], rows.labels) for name, rows in domains.items()})


@dataclass(frozen=True)
class Delta:
    """An arm's paired gain over the baseline: macro and per domain, each with its bootstrap interval."""

    point: float
    low: float
    high: float
    domains: dict[str, tuple[float, float, float]]

    @classmethod
    def of(cls, arm: Grading, base: Grading, domains: dict[str, Rows]) -> "Delta":
        ruler = Ruler(k=TOP, block=RESAMPLE_BLOCK)
        a, b = arm.graded(domains), base.graded(domains)
        macro = ruler.interval(ruler.delta(a, b))
        per: dict[str, tuple[float, float, float]] = {}
        for name, rows in domains.items():
            draws = ruler.draws(arm.hits[name], rows.labels) - ruler.draws(base.hits[name], rows.labels)
            interval = ruler.interval(draws)
            gain = ruler.balanced(arm.hits[name], rows.labels) - ruler.balanced(base.hits[name], rows.labels)
            per[name] = (gain, interval.low, interval.high)
        return cls(ruler.point(a) - ruler.point(b), macro.low, macro.high, per)

    def record(self) -> dict[str, object]:
        return {"delta": self.point, "low": self.low, "high": self.high, "domains": self.domains}


@dataclass(frozen=True)
class Section:
    """One saved section of every member: the novel words of the domains that count, the lone ones among them."""

    domains: dict[str, Rows]
    lone: dict[str, Rows]
    log_prior: Float[Tensor, " v"]

    @classmethod
    def load(cls, directories: Sequence[Path], section: str, lone: dict[str, Bool[Tensor, " n"]]) -> "Section":
        saved = [Probabilities.load(directory / PROBS, section) for directory in directories]
        names = [name for name, words in saved[0].domains.items() if int(words.novel.sum()) >= MIN_NOVEL]
        for name in names:
            if any(not torch.equal(member.domains[name].labels, saved[0].domains[name].labels) for member in saved):
                raise ValueError(f"members disagree on the words of {name} in {section}")
        words = {name: [member.domains[name] for member in saved] for name in saved[0].domains}
        domains = {name: Rows.mean(words[name], saved[0].domains[name].novel) for name in names}
        alone = {
            name: Rows.mean(words[name], mask).swapped() for name, mask in lone.items() if int(mask.sum()) >= MIN_NOVEL
        }
        return cls(domains, alone, saved[0].log_prior.float())


@dataclass(frozen=True)
class Study:
    """Every arm against the tau-only baseline, cross-fit on validation, then refit and reported on the local test."""

    val: Section
    test: Section | None

    def run(self, seed: int) -> dict[str, object]:
        arms = Arm.all()
        ruler = Ruler(k=TOP)
        val = self.val
        result: dict[str, object] = {
            "val-by-tau": {tau: Grading.point(Finish(tau), val.domains, val.log_prior, ruler) for tau in TAUS},
            "val": self.crossfit(arms, self.val.domains, seed),
        }
        lone = (arms[0], arms[3])
        if self.val.lone:
            result["lone-val"] = self.crossfit(lone, self.val.lone, seed)
        if self.test is not None:
            result["test"] = self.transfer(arms, self.val.domains, self.test.domains)
            if self.val.lone and self.test.lone:
                result["lone-test"] = self.transfer(lone, self.val.lone, self.test.lone)
        return result

    def crossfit(self, arms: Sequence[Arm], domains: dict[str, Rows], seed: int) -> dict[str, object]:
        graded = [Grading.crossfit(arm, domains, self.val.log_prior, seed) for arm in arms]
        return self.compared(arms, graded, domains)

    def transfer(self, arms: Sequence[Arm], fit: dict[str, Rows], graded: dict[str, Rows]) -> dict[str, object]:
        gradings = [Grading.transfer(arm, fit, graded, self.val.log_prior) for arm in arms]
        return self.compared(arms, gradings, graded)

    @staticmethod
    def compared(arms: Sequence[Arm], graded: Sequence[Grading], domains: dict[str, Rows]) -> dict[str, object]:
        ruler = Ruler(k=TOP)
        base = graded[0]
        sizes = {name: len(rows) for name, rows in domains.items()}
        out: dict[str, object] = {"base": ruler.point(base.graded(domains)), "n": sizes}
        for arm, grading in zip(arms[1:], graded[1:], strict=True):
            delta = Delta.of(grading, base, domains)
            LOG.info("%s: %+.4f [%+.4f, %+.4f] %s", arm.name, delta.point, delta.low, delta.high, delta.domains)
            out[arm.name] = delta.record()
        return out

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble (or run) under models/ whose members saved probs.pt")
        parser.add_argument("--members", nargs="*", help="these runs instead of the ensemble's record")
        parser.add_argument("--lone", type=Path, help="a lone-word mask file {section: {domain: bool[n]}}")
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        directories = (
            [root.models() / member for member in args.members]
            if args.members
            else Ensemble.trained(root.models(), args.name)
        )
        masks: dict[str, dict[str, Tensor]] = torch.load(args.lone) if args.lone else {}
        val = Section.load(directories, VAL, masks.get(VAL, {}))
        held = all(f"{TEST}{META}" in Probabilities.stored(directory / PROBS) for directory in directories)
        test = Section.load(directories, TEST, masks.get(TEST, {})) if held else None
        result = Study(val, test).run(args.seed)
        out = root.models() / args.name / OUT
        out.write_text(json.dumps(result, indent=1))
        LOG.info("wrote %s", out)


if __name__ == "__main__":
    Study.main()
