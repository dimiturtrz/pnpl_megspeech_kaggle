"""LM fusion from what the Kaggle test gives at inference: each span's row order and the ensemble's posteriors.

`lmbound` showed GPT-2 lifts the context head when it sees a word's TRUE in-vocabulary neighbours. Here the neighbours
are the model's own beliefs. A Kaggle sentence epoch holds every word heard, out-of-vocabulary ones included (its rows'
onsets are 0.21 s apart at the median, as all words of the training text are, against 0.53 s for the 50 words alone), so
row adjacency is known and each row's state is one of the 50 words or `OOV`. A neighbour's belief is
`share * softmax(head)` over the words and `1 - share` on `OOV`, `share` the in-vocabulary share of validation's rows:
the head has no out-of-vocabulary class, so the scalar is read off labels, never tuned.

An `OOV` neighbour ends the context (`lmbound`'s `backoff`). Each row's context is marginalised over its neighbours'
states, `WIDTHS` candidates deep: `pmi` over the left configurations, `pmi-right` over the `JOINT` most likely left and
right ones crossed. The fused channel is `log sum_c P(c) p_LM(w | c) - log p_LM(w)` over the 50, a first-order
(one-row-at-a-time) reading of the joint decode, which leaves each neighbour's belief unrefined. The `oracle` belief
(true word, or `OOV` where the label is unknown) re-derives `lmbound`'s `backoff` as the plumbing check. A word alone in
its span has only the empty context, so its channel is exactly 0.

Arms are cross-fit on validation halves and transferred to the local test as `lmbound`'s are. The Kaggle rows' channels
and the all-validation fit are saved to `KAGGLE_OUT`, so a later finish step needs no GPU rerun.

    uv run python -m megspeech.evaluation.studies.lmsoft m11xsp7hub6d2v3
"""

import argparse
import heapq
import itertools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.device import Placed
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.evaluation.studies.crowding import TAUS, TEST, TOP, VAL, Arm, Finish, Rows, Section
from megspeech.evaluation.studies.crowding import Study as Crossfit
from megspeech.evaluation.studies.lmbound import CANDIDATES, LAMS, RIGHT, Causal, Context, Table
from megspeech.evaluation.studies.lmrows import KAGGLE, ROWS, Posteriors
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

OOV = VOCAB_SIZE
"""A row's state outside the 50 words."""
WIDTHS = (4, 2)
"""Word candidates kept at the nearest neighbour, then the next; `RIGHT` deep, as `lmbound`'s right channel."""
JOINT = (3, 5)
"""Left, then right configurations crossed in `pmi-right`."""
CHANNELS = {"pmi": 0, "pmi-right": RIGHT}
BATCH = 8
"""Contexts per LM forward, each carrying every candidate."""
LOG_EVERY = 500
OUT = "lmsoft.json"
KAGGLE_OUT = "lmsoft-kaggle.pt"
EMPTY: Context = ((), ())
SOFT = "soft"
FUSED = Arm("lm", tuple(Finish(tau, 0.0, lam) for tau, lam in itertools.product(TAUS, LAMS)))
"""The context head plus `lam` times the LM channel, less `tau` times the log prior."""

States = tuple[int, ...]


@dataclass(frozen=True)
class Belief:
    """Every row's probability over the 50 words and `OOV`."""

    q: Float[Tensor, "n s"]

    @classmethod
    def soft(cls, head: Float[Tensor, "n v"], share: float) -> "Belief":
        words = share * head.float().softmax(-1)
        return cls(torch.cat([words, torch.full((len(head), 1), 1 - share)], dim=1))

    @classmethod
    def oracle(cls, labels: Int[Tensor, " n"]) -> "Belief":
        states = torch.where(labels >= 0, labels, OOV)
        return cls(torch.nn.functional.one_hot(states, VOCAB_SIZE + 1).float())

    def outward(self, rows: Sequence[int], widths: Sequence[int]) -> list[tuple[States, float]]:
        """The in-vocabulary runs starting at `rows[0]` going outward, nearest first, each with its probability: an
        `OOV` row ends a run, and so does the last of `widths`."""
        if not rows or not widths:
            return [((), 1.0)]
        belief = self.q[rows[0]]
        configs = [((), float(belief[OOV]))]
        values, words = belief[:OOV].topk(widths[0])
        for word, weight in zip(words.tolist(), values.tolist(), strict=True):
            configs += [((word, *rest), weight * tail) for rest, tail in self.outward(rows[1:], widths[1:])]
        return [(states, weight) for states, weight in configs if weight > 0]


@dataclass(frozen=True)
class Weighted:
    """One row's contexts and their probabilities."""

    contexts: tuple[Context, ...]
    weights: Float[Tensor, " k"]

    @classmethod
    def of(cls, pairs: Sequence[tuple[Context, float]]) -> "Weighted":
        weights = torch.tensor([weight for _, weight in pairs], dtype=torch.float64)
        return cls(tuple(key for key, _ in pairs), (weights / weights.sum()).float())

    def other(self, table: Table, column: int) -> Float[Tensor, " c"]:
        """The candidates' LM log-probability at `column` mixed over the contexts, less the unigram's."""
        scores = torch.stack([table.scores[key][:, column] for key in self.contexts]).log_softmax(-1)
        return torch.logsumexp(scores + self.weights.log()[:, None], dim=0) - table.unigram.log_softmax(-1)


@dataclass(frozen=True)
class Neighbours:
    """A domain's beliefs and every row's span and place in it."""

    belief: Belief
    place: dict[int, tuple[list[int], int]]

    @classmethod
    def of(cls, belief: Belief, index: Int[Tensor, "g w"]) -> "Neighbours":
        place: dict[int, tuple[list[int], int]] = {}
        for span in index.tolist():
            members = [row for row in span if row != PAD]
            place.update({row: (members, at) for at, row in enumerate(members)})
        return cls(belief, place)

    def channels(self, row: int) -> dict[str, Weighted]:
        """`row`'s contexts in each channel; a row in no span stands alone."""
        members, at = self.place.get(row, ([row], 0))
        outward = self.belief.outward
        left = [
            (self.spelled(states[::-1], at - len(states)), weight)
            for states, weight in outward(members[:at][::-1], WIDTHS)
        ]
        right = [(self.spelled(states, at + 1), weight) for states, weight in outward(members[at + 1 :], WIDTHS)]
        first, second = (heapq.nlargest(k, side, itemgetter(1)) for k, side in zip(JOINT, (left, right), strict=True))
        return {
            "pmi": Weighted.of([((words, ()), weight) for words, weight in left]),
            "pmi-right": Weighted.of(
                [((before, after), wb * wa) for (before, wb), (after, wa) in itertools.product(first, second)]
            ),
        }

    @staticmethod
    def spelled(states: States, start: int) -> tuple[str, ...]:
        """`states` as words, the first capitalised when it opens its span at `start`."""
        words = [CANDIDATES[state] for state in states]
        if words and start == 0:
            words[0] = words[0][:1].upper() + words[0][1:]
        return tuple(words)


@dataclass(frozen=True)
class Batched:
    """`Causal.score` for many contexts per forward: every context's candidates right-padded into one batch."""

    causal: Causal

    def score(self, keys: Sequence[Context]) -> list[Float[Tensor, "c j"]]:
        causal = self.causal
        own = causal.pieces(CANDIDATES)
        rows: list[list[int]] = []
        starts: list[int] = []
        ends: list[list[int]] = []
        for left, right in keys:
            head = [causal.bos, *(token for piece in causal.pieces(left) for token in piece)]
            after = causal.pieces(right)
            flat = [token for piece in after for token in piece]
            reach = list(itertools.accumulate(map(len, after)))
            for piece in own:
                stops = [len(piece), *(len(piece) + length for length in reach)]
                rows.append([*head, *piece, *flat])
                starts.append(len(head))
                ends.append(stops + stops[-1:] * (RIGHT + 1 - len(stops)))
        steps = self.steps(rows, torch.tensor(starts))
        picked = steps.gather(1, torch.tensor(ends, device=steps.device) - 1).cpu()
        return list(picked.view(len(keys), len(own), RIGHT + 1))

    def steps(self, rows: list[list[int]], starts: Int[Tensor, " s"]) -> Float[Tensor, "s t"]:
        """Each row's summed token log-probabilities from its `starts` token on, one column per token."""
        model = self.causal.model
        output = model.get_output_embeddings()
        if output is None:
            raise ValueError("the LM has no output embedding")
        device = model.device
        width = max(map(len, rows))
        lengths = torch.tensor([len(row) for row in rows], device=device)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows], device=device)
        mask = (torch.arange(width, device=device) < lengths[:, None]).long()
        span = int((lengths - starts.to(device)).max())
        at = (starts.to(device)[:, None] - 1 + torch.arange(span, device=device)).clamp(max=width - 2)
        with torch.no_grad():
            hidden = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            chosen = hidden.gather(1, at[:, :, None].expand(-1, -1, hidden.shape[-1]))
            logits = output(chosen).float().log_softmax(-1)
        return logits.gather(2, ids.gather(1, at + 1)[:, :, None]).squeeze(2).cumsum(dim=1)

    def table(self, wanted: Sequence[Context], known: dict[Context, Tensor]) -> Table:
        """`known` with every missing context of `wanted` scored, shortest first so a batch pads little."""
        missing = sorted(set(wanted) - set(known), key=lambda key: (len(key[0]) + len(key[1]), key))
        scores = dict(known)
        for count, start in enumerate(range(0, len(missing), BATCH)):
            batch = missing[start : start + BATCH]
            scores.update(zip(batch, self.score(batch), strict=True))
            if count % LOG_EVERY == 0:
                LOG.info("scored %d of %d contexts", start, len(missing))
        return Table(scores, scores[EMPTY][:, 0])


@dataclass(frozen=True)
class Fusion:
    """One section's graded words and every belief's and channel's contexts for each of them."""

    section: Section
    weighted: dict[str, dict[str, dict[str, list[Weighted]]]]
    """By belief, then channel, then domain."""

    @classmethod
    def of(
        cls, section: Section, neighbours: dict[str, dict[str, Neighbours]], ids: dict[str, Int[Tensor, " n"]]
    ) -> "Fusion":
        """`neighbours` by belief, then domain; `ids` each domain's graded rows."""
        weighted: dict[str, dict[str, dict[str, list[Weighted]]]] = {}
        for belief, domains in neighbours.items():
            per = {name: [domains[name].channels(row) for row in rows.tolist()] for name, rows in ids.items()}
            weighted[belief] = {
                channel: {name: [row[channel] for row in rows] for name, rows in per.items()} for channel in CHANNELS
            }
        return cls(section, weighted)

    def wanted(self) -> list[Context]:
        return [
            key
            for channels in self.weighted.values()
            for domains in channels.values()
            for rows in domains.values()
            for row in rows
            for key in row.contexts
        ]

    def arms(self) -> list[tuple[str, str]]:
        return [(belief, channel) for belief, channels in self.weighted.items() for channel in channels]

    def fused(self, table: Table, belief: str, channel: str) -> dict[str, Rows]:
        """Each domain's words with the LM channel in place of the other head."""
        column = CHANNELS[channel]
        rows = self.weighted[belief][channel]
        domains = self.section.domains
        return {
            name: Rows(words.head, torch.stack([row.other(table, column) for row in rows[name]]), words.labels)
            for name, words in domains.items()
        }


@dataclass(frozen=True)
class Study:
    """Every belief and channel of LM fusion against the tau-only finish."""

    val: Fusion
    test: Fusion

    def run(self, table: Table, seed: int) -> dict[str, object]:
        arms = (Arm.all()[0], FUSED)
        result: dict[str, object] = {}
        for belief, channel in self.val.arms():
            LOG.info("%s %s", belief, channel)
            val = self.val.fused(table, belief, channel)
            crossfit = Crossfit(Section(val, {}, self.val.section.log_prior), None)
            result[f"{belief}/{channel}"] = {
                "val": crossfit.crossfit(arms, val, seed),
                "test": crossfit.transfer(arms, val, self.test.fused(table, belief, channel)),
            }
        return result

    def fitted(self, table: Table, channel: str) -> Finish:
        """The fused arm fit on all of validation under the `soft` belief: the finish the Kaggle export applies."""
        section = self.val.section
        return FUSED.fit(self.val.fused(table, SOFT, channel), section.log_prior, Ruler(k=TOP))

    def fits(self, table: Table) -> dict[str, dict[str, float]]:
        """Each channel's `fitted` tau and lam."""
        finishes = {channel: self.fitted(table, channel) for channel in CHANNELS}
        return {channel: {"tau": finish.tau, "lam": finish.mix} for channel, finish in finishes.items()}

    @staticmethod
    def aligned(rows: Posteriors, graded: Rows, ids: Int[Tensor, " n"]) -> float:
        """The dumped rows' head top-1 agreement with the graded words', after their labels are checked equal."""
        if not torch.equal(rows.labels[ids], graded.labels):
            raise ValueError("the dumped rows' labels differ from the graded words'")
        return float((rows.head[ids].argmax(-1) == graded.head.argmax(-1)).float().mean())

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/ with probs.pt per member and lmrows.pt")
        parser.add_argument("--lm", default="gpt2-large")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        directories = Ensemble.trained(root.models(), args.name)
        out = root.models() / args.name
        dumped = Dumped(out / ROWS, Dumped.shared(Posteriors.load(str(out / ROWS), VAL)))
        val, val_aligned = dumped.fusion(directories, VAL)
        test, test_aligned = dumped.fusion(directories, TEST)
        kaggle = dumped.kaggle()
        LOG.info("share %.4f aligned %s %s", dumped.share, val_aligned, test_aligned)
        wanted = [EMPTY, *val.wanted(), *test.wanted(), *kaggle.contexts()]
        cache = out / f"lmsoft-{args.lm}.pt"
        known: dict[Context, Tensor] = torch.load(cache) if cache.exists() else {}
        table = Batched(Causal.load(Placed(root, args.lm, torch.device(args.device)))).table(wanted, known)
        torch.save(table.scores, cache)
        channels = kaggle.others(table)
        study = Study(val, test)
        fits = study.fits(table)
        saved = {"share": dumped.share, "log_prior": val.section.log_prior, "channels": channels, "fits": fits}
        torch.save(saved, out / KAGGLE_OUT)
        result = {
            "share": dumped.share,
            "aligned": {VAL: val_aligned, TEST: test_aligned},
            "kaggle-lone-max": kaggle.lone(channels),
            "fits": fits,
            **study.run(table, args.seed),
        }
        (out / OUT).write_text(json.dumps(result, indent=1))
        LOG.info("wrote %s", out / OUT)


@dataclass(frozen=True)
class Kaggle:
    """The Kaggle rows' contexts under the `soft` belief, by channel."""

    weighted: dict[str, list[Weighted]]

    def contexts(self) -> list[Context]:
        return [key for rows in self.weighted.values() for row in rows for key in row.contexts]

    def others(self, table: Table) -> dict[str, Float[Tensor, "n c"]]:
        """Each channel's LM channel per row."""
        return {
            channel: torch.stack([row.other(table, CHANNELS[channel]) for row in rows])
            for channel, rows in self.weighted.items()
        }

    def lone(self, others: dict[str, Float[Tensor, "n c"]]) -> float:
        """The largest channel magnitude over the rows alone in their span: 0 when the channel is sound."""
        alone = [row.contexts == (EMPTY,) for row in self.weighted["pmi-right"]]
        return max(float(other[torch.tensor(alone)].abs().max()) for other in others.values())


@dataclass(frozen=True)
class Dumped:
    """An ensemble's dumped rows and the share of validation's words inside the vocabulary."""

    path: Path
    share: float

    @staticmethod
    def shared(rows: dict[str, Posteriors]) -> float:
        """The share of `rows`' words inside the vocabulary."""
        labels = torch.cat([posteriors.labels for posteriors in rows.values()])
        return float((labels >= 0).float().mean())

    def fusion(  # devtools-ignore: test-mirror
        self, directories: Sequence[Path], section: str
    ) -> tuple[Fusion, dict[str, float]]:
        graded = Section.load(directories, section, {})
        saved = Probabilities.load(directories[0] / PROBS, section).domains
        ids = {name: saved[name].ids[saved[name].novel] for name in graded.domains}
        dumped = Posteriors.load(str(self.path), section)
        agreement = {name: Study.aligned(dumped[name], graded.domains[name], ids[name]) for name in graded.domains}
        beliefs = {
            "oracle": {name: Belief.oracle(dumped[name].labels) for name in graded.domains},
            SOFT: {name: Belief.soft(dumped[name].head, self.share) for name in graded.domains},
        }
        neighbours = {
            belief: {name: Neighbours.of(domains[name], dumped[name].index) for name in domains}
            for belief, domains in beliefs.items()
        }
        return Fusion.of(graded, neighbours, ids), agreement

    def kaggle(self) -> Kaggle:  # devtools-ignore: test-mirror
        dumped = Posteriors.load(str(self.path), KAGGLE)[KAGGLE]
        neighbours = Neighbours.of(Belief.soft(dumped.head, self.share), dumped.index)
        per = [neighbours.channels(row) for row in range(len(dumped.labels))]
        return Kaggle({channel: [row[channel] for row in per] for channel in CHANNELS})


if __name__ == "__main__":
    Study.main()
