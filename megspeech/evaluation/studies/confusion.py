"""Which target words a timing-only model confuses, and where real MEG separates them (megspeech-7gf.71).

The timing-only model is the trained model scored with every word's MEG window swapped for another word's of the same
domain, onsets untouched (`timing.Shuffled`), its probabilities averaged over `PERMUTATIONS` shuffles; the real model is
the same weights unpermuted. Both are read on the official validation novel-span words, the ruler's own words. Per
target: top-1 recall and top-10 hit rate; the 50x50 top-1 confusion; its most-confused pairs; clusters of targets the
timing model cannot separate (average-linkage on the symmetrised confusion, cut where a cluster's mean pairwise
confusion falls under five times the uniform rate); whether confusion tracks word length (letters, phonemes from the
events'
forced alignment, mean gap to the next onset); and where the real model's top-10 gain over timing lands.

    uv run python -m megspeech.evaluation.studies.confusion sent40-ioi-ema9998-s0 --device cpu
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.labels.phonemes import EVENTS, WORD
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.metric.ruler import TOP
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.scored import Scored
from megspeech.evaluation.studies.timing import PERMUTATIONS

LOG = logging.getLogger(__name__)

CONTESTED = 5.0
"""A cluster is contested when its mean pairwise confusion is at least this many times the uniform rate 1/V."""
PAIRS = 15
"""Most-confused pairs reported."""
CONFUSIONS = 5
"""Top confusions listed for a single target."""
NON_WORD_PHONEMES = ("oov", "sil")
OUT = "confusion"
CELL = 0.17
"""Inches per heatmap row, so the 50 tick labels stay legible."""
LABEL_SIZE = 5


@dataclass(frozen=True)
class Structure:
    """What a top-1 confusion matrix says about which targets are confused."""

    counts: Int[np.ndarray, "v v"]

    def rates(self) -> Float[np.ndarray, "v v"]:
        """Row-normalised confusion; a target with no word has a zero row."""
        return self.counts / np.maximum(self.counts.sum(axis=1, keepdims=True), 1)

    def affinity(self) -> Float[np.ndarray, "v v"]:
        """Symmetrised off-diagonal confusion rate, zero on the diagonal."""
        rates = self.rates()
        both = (rates + rates.T) / 2
        return both - np.diag(np.diag(both))

    def tree(self) -> Float[np.ndarray, "k 4"]:
        """Average-linkage tree over distance 1 - affinity."""
        return linkage(squareform(1 - self.affinity(), checks=False), method="average")

    def order(self) -> Int[np.ndarray, " v"]:
        """The tree's leaf order, similar targets adjacent."""
        return leaves_list(self.tree())

    def pairs(self, k: int) -> list[tuple[int, int, int, int]]:
        """The `k` unordered pairs with most confusions, as (i, j, i read as j, j read as i)."""
        both = self.counts + self.counts.T
        rows, cols = np.triu_indices(VOCAB_SIZE, 1)
        top = np.argsort(-both[rows, cols], kind="stable")[:k]
        return [
            (int(rows[p]), int(cols[p]), int(self.counts[rows[p], cols[p]]), int(self.counts[cols[p], rows[p]]))
            for p in top
        ]

    def clusters(self) -> list[list[int]]:
        """Groups of at least two targets whose mean pairwise confusion is `CONTESTED` times the uniform rate."""
        cut = 1 - CONTESTED / VOCAB_SIZE
        groups = fcluster(self.tree(), cut, criterion="distance")
        found = [np.flatnonzero(groups == g).tolist() for g in np.unique(groups)]
        return sorted((members for members in found if len(members) > 1), key=len, reverse=True)

    def pair_values(self) -> Float[np.ndarray, " p"]:
        """Affinity of every unordered pair, in `np.triu_indices` order."""
        return self.affinity()[np.triu_indices(VOCAB_SIZE, 1)]


@dataclass(frozen=True)
class Spellings:
    """The target words, measured by their letters and by their sounds in the forced alignment."""

    words: Sequence[str]

    def letters(self) -> Float[np.ndarray, " v"]:
        return np.array([sum(char.isalpha() for char in word) for word in self.words], dtype=np.float64)

    def phonemes(self, events: Sequence[Path]) -> Float[np.ndarray, " v"]:
        """Each word's most common phoneme count in the events' forced alignment (NaN if it never occurs)."""
        pieces = []
        for path in events:
            table = pd.read_csv(path, sep="\t", usecols=["kind", "segment", *WORD], dtype={"segment": str})
            sounds = table[(table["kind"] == "phoneme") & ~table["segment"].str.startswith(NON_WORD_PHONEMES)]
            counts = sounds.groupby(WORD).size().rename("count").reset_index()
            named = table.loc[table["kind"] == "word", ["segment", *WORD]].merge(counts, on=WORD)
            pieces.append(named.assign(word=Vocabulary.spelled(named["segment"]))[["word", "count"]])
        merged = pd.concat(pieces, ignore_index=True)
        common = merged.groupby("word")["count"].agg(lambda counted: counted.mode().iloc[0])
        return np.array([float(common.get(word, np.nan)) for word in self.words])


@dataclass(frozen=True)
class Lengths:
    """Per-target size measures: letters, phonemes and mean seconds to the next onset."""

    letters: Float[np.ndarray, " v"]
    phonemes: Float[np.ndarray, " v"]
    duration: Float[np.ndarray, " v"]

    @staticmethod
    def durations(split: Split, novel: Int[Tensor, " k"]) -> Float[np.ndarray, " k"]:
        """Seconds from each `novel` word's onset to the next onset of its span, NaN for a span's last word."""
        index, onset = split.index.cpu().numpy(), split.onset.cpu().numpy()
        gap = np.full(len(split.y), np.nan)
        valid = (index[:, :-1] != PAD) & (index[:, 1:] != PAD)
        gap[index[:, :-1][valid]] = (onset[:, 1:] - onset[:, :-1])[valid]
        return gap[novel.cpu().numpy()]

    @staticmethod
    def mean_by_target(labels: Int[np.ndarray, " n"], values: Float[np.ndarray, " n"]) -> Float[np.ndarray, " v"]:
        known = ~np.isnan(values)
        total = np.bincount(labels[known], weights=values[known], minlength=VOCAB_SIZE)
        count = np.bincount(labels[known], minlength=VOCAB_SIZE)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def spearman(self, pair_affinity: Float[np.ndarray, " p"]) -> dict[str, tuple[float, float, int]]:
        """Spearman rho, p and pair count of pairwise confusion against each |difference| in size."""
        rows, cols = np.triu_indices(VOCAB_SIZE, 1)
        out = {}
        for name, size in (("letters", self.letters), ("phonemes", self.phonemes), ("duration", self.duration)):
            gap = np.abs(size[rows] - size[cols])
            known = ~np.isnan(gap)
            result = spearmanr(pair_affinity[known], gap[known])
            out[name] = (float(result.statistic), float(result.pvalue), int(known.sum()))
        return out


@dataclass(frozen=True)
class Collected:
    """The real and timing-only models' probabilities on the same words, and each word's duration."""

    real: Scored
    timing: Scored
    duration: Float[np.ndarray, " n"]

    @staticmethod
    def _probabilities(
        loaded: LoadedModel, split: Split, novel: Int[Tensor, " k"], tau: float
    ) -> Float[np.ndarray, "k v"]:
        evaluator = loaded.evaluator
        logits = evaluator.logits(loaded.model, split)[Head.CONTEXT][novel]
        return torch.softmax(evaluator.adjusted(logits, loaded.log_prior, tau), dim=-1).cpu().numpy()

    @classmethod
    def of(cls, loaded: LoadedModel, runs: list[Run]) -> "Collected":  # devtools-ignore: test-mirror
        """Every counted domain's novel words, scored real and permuted (`PERMUTATIONS` shuffles averaged)."""
        tau = loaded.report(runs).tau
        domains, _ = loaded.domains(loaded.ruled(runs))
        evaluator = loaded.evaluator
        labels: list[Int[np.ndarray, " k"]] = []
        real: list[Float[np.ndarray, "k v"]] = []
        timing: list[Float[np.ndarray, "k v"]] = []
        names: list[np.ndarray] = []
        gaps: list[Float[np.ndarray, " k"]] = []
        for name, split in domains.items():
            novel = split.novel(evaluator.seen)
            if len(novel) < evaluator.min_novel:
                continue
            labels.append(split.y[novel].cpu().numpy())
            real.append(cls._probabilities(loaded, split, novel, tau))
            shuffles = [cls._probabilities(loaded, split.permuted(seed), novel, tau) for seed in range(PERMUTATIONS)]
            timing.append(np.mean(shuffles, axis=0))
            names.append(np.full(len(novel), name))
            gaps.append(Lengths.durations(split, novel))
        y, where = np.concatenate(labels), np.concatenate(names)
        return cls(
            Scored(y, np.concatenate(real), where), Scored(y, np.concatenate(timing), where), np.concatenate(gaps)
        )


@dataclass(frozen=True)
class Heatmap:
    """Row-normalised confusions side by side, targets in `order`."""

    words: Sequence[str]
    order: Int[np.ndarray, " v"]

    def save(self, path: Path, panels: dict[str, Float[np.ndarray, "v v"]]) -> Path:
        side = CELL * VOCAB_SIZE
        figure = Figure(figsize=(side * len(panels) + 2, side + 1), layout="constrained")
        names = [self.words[i] for i in self.order]
        axes = figure.subplots(1, len(panels))
        for axis, (title, rates) in zip(np.atleast_1d(axes), panels.items(), strict=True):
            image = axis.imshow(rates[np.ix_(self.order, self.order)], cmap="magma_r", vmin=0, vmax=1)
            axis.set_xticks(range(VOCAB_SIZE), names, rotation=90, fontsize=LABEL_SIZE)
            axis.set_yticks(range(VOCAB_SIZE), names, fontsize=LABEL_SIZE)
            axis.set_title(title)
            axis.set_xlabel("predicted")
            axis.set_ylabel("true")
            figure.colorbar(image, ax=axis, shrink=0.6, label="row-normalised top-1")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        return path


@dataclass(frozen=True)
class Gains:
    """Real minus timing-only top-`TOP` hit rate per target, and whether it concentrates in contested clusters."""

    gain: Float[np.ndarray, " v"]
    clustered: Bool[np.ndarray, " v"]
    mass: Float[np.ndarray, " v"]

    @classmethod
    def of(cls, gain: Float[np.ndarray, " v"], shape: Structure) -> "Gains":
        member = np.zeros(VOCAB_SIZE, dtype=bool)
        member[[i for group in shape.clusters() for i in group]] = True
        return cls(gain, member, shape.affinity().sum(axis=1))

    def concentration(self) -> dict[str, float]:
        """Mean gain inside and outside the clusters, their share of the total gain, and a rank-sum p-value."""
        inside, outside = self.gain[self.clustered], self.gain[~self.clustered]
        known = ~np.isnan(self.gain)
        test = mannwhitneyu(inside[~np.isnan(inside)], outside[~np.isnan(outside)])
        return {
            "n-inside": int(self.clustered.sum()),
            "n-outside": int((~self.clustered).sum()),
            "mean-inside": float(np.nanmean(inside)),
            "mean-outside": float(np.nanmean(outside)),
            "share-of-total-inside": float(np.nansum(inside) / np.nansum(self.gain[known])),
            "mann-whitney-p": float(test.pvalue),
            "rho-gain-vs-timing-confusion-mass": float(spearmanr(self.gain[known], self.mass[known]).statistic),
        }


class Report:
    """Logs the study's findings and saves its artefacts."""

    def __init__(self, collected: Collected, words: Sequence[str], lengths: Lengths, out: Path) -> None:
        self.collected, self.words, self.lengths, self.out = collected, words, lengths, out

    def run(self) -> None:
        real, timing = self.collected.real, self.collected.timing
        for name, scored in (("real", real), ("timing", timing)):
            LOG.info("%s bAcc@%d %s (n=%d)", name, TOP, scored.macro(), len(scored.labels))
        support = real.support()
        table = pd.DataFrame(
            {
                "n": support,
                "real-top1": real.recall(),
                "timing-top1": timing.recall(),
                "real-top10": real.hit_rate(),
                "timing-top10": timing.hit_rate(),
            },
            index=list(self.words),
        )
        LOG.info("per-target (n, top-1 recall, top-%d hit rate)\n%s", TOP, table.round(3).to_string())
        shape = Structure(timing.confusion())
        self._save(shape, Structure(real.confusion()))
        for i, j, ij, ji in shape.pairs(PAIRS):
            LOG.info(
                "pair %s/%s: %s->%s %d, %s->%s %d",
                self.words[i],
                self.words[j],
                self.words[i],
                self.words[j],
                ij,
                self.words[j],
                self.words[i],
                ji,
            )
        clusters = shape.clusters()
        for members in clusters:
            LOG.info("cluster (%d): %s", len(members), " ".join(self.words[m] for m in members))
        LOG.info("%d targets in %d clusters of %d", sum(map(len, clusters)), len(clusters), VOCAB_SIZE)
        for name, (rho, p, n) in self.lengths.spearman(shape.pair_values()).items():
            LOG.info("spearman confusion vs |d %s| rho=%.3f p=%.2g pairs=%d", name, rho, p, n)
        self._target(table, shape, Structure(real.confusion()), "a")
        gain = real.hit_rate() - timing.hit_rate()
        ranked = table.assign(gain=gain, cluster=Gains.of(gain, shape).clustered).sort_values("gain", ascending=False)
        LOG.info("top-%d gain real - timing\n%s", TOP, ranked.round(3).to_string())
        LOG.info("gain concentration %s", Gains.of(gain, shape).concentration())

    def _target(self, table: pd.DataFrame, shape: Structure, real: Structure, word: str) -> None:
        i = PRIMARY.positions[word]
        LOG.info("%s: %s", word, table.loc[word].round(3).to_dict())
        for name, counts in (("timing", shape.counts), ("real", real.counts)):
            row = counts[i]
            top = [(self.words[j], int(row[j])) for j in np.argsort(-row, kind="stable")[: CONFUSIONS + 1]]
            LOG.info("%s top predictions for true %s (n=%d): %s", name, word, row.sum(), top)
        recall_rank = int(np.argsort(np.argsort(-shape.rates().diagonal()))[i])
        LOG.info("%s recall rank among targets (timing, 0=highest): %d", word, recall_rank)
        off = shape.affinity().sum(axis=1)
        LOG.info(
            "%s confusion mass (affinity row sum) rank (0=least confused): %d of %d",
            word,
            int(np.argsort(np.argsort(off))[i]),
            VOCAB_SIZE,
        )

    def _save(self, shape: Structure, real: Structure) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        order = shape.order()
        for name, structure in (("timing", shape), ("real", real)):
            np.save(self.out / f"{name}_counts.npy", structure.counts)
            np.save(self.out / f"{name}_confusion.npy", structure.rates())
        np.save(self.out / "order.npy", order)
        (self.out / "words.json").write_text(json.dumps(list(self.words)))
        png = Heatmap(self.words, order).save(
            self.out / "confusion.png", {"timing-only": shape.rates(), "real MEG": real.rates()}
        )
        LOG.info("saved %s", png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--device", default=Device.CPU)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(root, [args.name], Scoring(), torch.device(args.device))
    runs = Run.discover(root.raw())
    collected = Collected.of(loaded, runs)
    words = PRIMARY.words
    events = [path for path in sorted(root.raw().rglob(EVENTS)) if path.stat().st_size]
    spellings = Spellings(words)
    lengths = Lengths(
        spellings.letters(),
        spellings.phonemes(events),
        Lengths.mean_by_target(collected.real.labels, collected.duration),
    )
    Report(collected, words, lengths, root.models() / OUT).run()


if __name__ == "__main__":
    main()
