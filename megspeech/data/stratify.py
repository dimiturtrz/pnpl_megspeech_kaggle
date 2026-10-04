"""Stratified train/validation/test split of one source's subject-0 training sessions (`DataConfig.split`)."""

from functools import cache

import numpy as np
from jaxtyping import Float, Int

from megspeech.data.cache.run_cache import NO_SENTENCE, RunCache
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY, VOCAB_SIZE
from megspeech.data.raw.runs import Run

Picked = tuple[list[Run], list[Run]]
"""A source's sessions picked for validation, then for test."""

WORDS, SENTENCES, SENTENCED, LOG_IOI, IOIS, KNOWN = range(6)
"""A session's additive counts: words, sentences, words in a sentence, summed log inter-onset interval within a
sentence and their count, words in the vocabulary; its per-label counts follow, then its words per book."""
LABELS = slice(KNOWN + 1, KNOWN + 1 + VOCAB_SIZE)
BOOKS = slice(LABELS.stop, None)
RATIOS = (np.array([LOG_IOI, SENTENCED, KNOWN]), np.array([IOIS, SENTENCES, WORDS]))
"""Numerators and denominators of the per-word means compared: log word rate, sentence length, in-vocabulary share."""

CRITERIA = 6
"""Gaps a held partition is balanced on: word volume, the three `RATIOS`, label mix and book mix."""
DRAWS = 512
"""Random assignments each criterion's gap is scaled by: its median over them is one unit, so criteria add on one scale
without hand-set weights."""
RESTARTS = 8
SEED = 0
TRAIN, VALIDATION, TEST = range(3)


class Stratify:
    """The validation and test sessions whose words best match training's on every criterion at once: word volume,
    word rate, sentence length, in-vocabulary share, label mix and book mix (megspeech-7gf.81)."""

    @staticmethod
    def profile(run: Run, books: tuple[str, ...]) -> Float[np.ndarray, " d"]:
        """`run`'s additive counts (`WORDS`...), its labels' and its words under its book in `books`."""
        words = RunCache.words(run)
        labels = PRIMARY.labels(words["word"])
        known = labels[labels != ABSENT]
        sentenced = words[words["sentence"] != NO_SENTENCE]
        intervals = sentenced.groupby("sentence")["onset"].diff().dropna().to_numpy()
        intervals = intervals[intervals > 0]
        book = np.eye(len(books))[books.index(run.key[0])] * len(words)
        counts = [len(words), sentenced["sentence"].nunique(), len(sentenced), np.log(intervals).sum(), len(intervals)]
        return np.concatenate([counts, [len(known)], np.bincount(known, minlength=VOCAB_SIZE), book])

    @staticmethod
    def gaps(
        held: Float[np.ndarray, "k d"], rest: Float[np.ndarray, "k d"], fraction: float
    ) -> Float[np.ndarray, "k c"]:
        """Each assignment's `held` counts against training's `rest`: its word share off `fraction` (its session share),
        each of `RATIOS` apart, the labels' Jensen-Shannon divergence, the books' total variation."""
        volume = np.abs(held[:, WORDS] / (held[:, WORDS] + rest[:, WORDS]) - fraction)
        numerator, denominator = RATIOS
        means = [counts[:, numerator] / np.maximum(counts[:, denominator], 1) for counts in (held, rest)]
        p, q = (counts[:, LABELS] + 1e-9 for counts in (held, rest))
        p, q = p / p.sum(1, keepdims=True), q / q.sum(1, keepdims=True)
        m = (p + q) / 2
        divergence = 0.5 * (p * np.log(p / m) + q * np.log(q / m)).sum(1)
        books = [counts[:, BOOKS] / counts[:, BOOKS].sum(1, keepdims=True) for counts in (held, rest)]
        variation = 0.5 * np.abs(books[0] - books[1]).sum(1)
        return np.column_stack([volume, np.abs(means[0] - means[1]), divergence, variation])

    @classmethod
    def cost(
        cls, totals: Float[np.ndarray, "k 3 d"], fraction: float, scale: Float[np.ndarray, " c"]
    ) -> Float[np.ndarray, " k"]:
        """Each assignment's summed scaled gaps, validation's and test's against training's."""
        rest = totals[:, TRAIN]
        held = np.stack([cls.gaps(totals[:, part], rest, fraction) for part in (VALIDATION, TEST)])
        return (held / scale).sum((0, 2))

    @staticmethod
    def totals(profiles: Float[np.ndarray, "n d"], assigned: Int[np.ndarray, "k n"]) -> Float[np.ndarray, "k 3 d"]:
        """Each assignment's summed counts per partition (`TRAIN`, `VALIDATION`, `TEST`)."""
        return np.einsum("knp,nd->kpd", np.eye(3)[assigned], profiles)

    @classmethod
    def search(
        cls, profiles: Float[np.ndarray, "n d"], start: Int[np.ndarray, " n"], scale: Float[np.ndarray, " c"]
    ) -> tuple[float, Int[np.ndarray, " n"]]:
        """`start` improved by the best swap of two sessions' partitions until none lowers the cost."""
        fraction = float((start == VALIDATION).mean())
        assigned, best = start.copy(), float(cls.cost(cls.totals(profiles, start[None]), fraction, scale)[0])
        i, j = np.triu_indices(len(start), 1)
        while True:
            moved = (np.eye(3)[assigned[j]] - np.eye(3)[assigned[i]])[:, :, None] * (profiles[i] - profiles[j])[:, None]
            costs = cls.cost(cls.totals(profiles, assigned[None]) + moved, fraction, scale)
            costs[assigned[i] == assigned[j]] = np.inf
            pick = int(costs.argmin())
            if costs[pick] >= best - 1e-12:
                return best, assigned
            assigned[[i[pick], j[pick]]] = assigned[[j[pick], i[pick]]]
            best = float(costs[pick])

    @classmethod
    @cache
    def split(cls, source: tuple[Run, ...], count: int) -> Picked:
        """`count` of `source`'s sessions each to validation and test: the assignment of least `cost` over swap
        searches from `RESTARTS` seeded random ones, each gap scaled by its median over `DRAWS` random ones."""
        if not count:
            return [], []
        books = tuple(sorted({run.key[0] for run in source}))
        profiles = np.stack([cls.profile(run, books) for run in source])
        base = np.array([TRAIN] * (len(source) - 2 * count) + [VALIDATION] * count + [TEST] * count)
        rng = np.random.default_rng(SEED)
        draws = np.stack([rng.permutation(base) for _ in range(DRAWS)])
        totals = cls.totals(profiles, draws)
        held = np.concatenate(
            [cls.gaps(totals[:, part], totals[:, TRAIN], count / len(source)) for part in (VALIDATION, TEST)]
        )
        median = np.median(held, axis=0)
        scale = np.where(median > 0, median, 1.0)
        _, assigned = min(
            (cls.search(profiles, start, scale) for start in draws[:RESTARTS]), key=lambda found: found[0]
        )
        val, test = (
            [run for run, at in zip(source, assigned, strict=True) if at == part] for part in (VALIDATION, TEST)
        )
        return val, test
