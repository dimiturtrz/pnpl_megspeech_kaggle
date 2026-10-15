"""The viewer's samples: whole sentences drawn per source the way `Stratify` draws val's sessions, and their text.

`Stratify` assigns whole sessions; here the unit is one sentence, whose profile is its words' sums in the same
additive layout (`WORDS`...`KNOWN`, label counts, book counts) a session's sums to, and the pick is the draw of least
summed scaled `Stratify.gaps` against the rest of its source. Its exhaustive pair-swap search is quadratic in the units,
so over thousands of words it is replaced by rounds of sampled swaps.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import NO_SENTENCE, Column, Run
from megspeech.data.sources import Sources
from megspeech.data.stratify import DRAWS, IOIS, KNOWN, LOG_IOI, SEED, SENTENCED, SENTENCES, WORDS, Stratify

ROUNDS = 64
"""Swap rounds after the best random draw; a round that lowers nothing ends the search."""
PROPOSALS = 4096
"""Random (held, rest) word pairs each swap round prices at once."""
LENGTHS = ("short", "medium", "long")
"""Sentence-length strata: the pool's terciles of the length of each word's sentence."""
SOURCE, LENGTH = "source", "length"
STRATA = (SOURCE, LENGTH)


@dataclass(frozen=True)
class Profiles:
    """Each word's additive counts in `Stratify.profile`'s layout, so `Stratify.gaps` compares word sets as sessions."""

    @staticmethod
    def of(words: pd.DataFrame, books: tuple[str, ...]) -> Float[np.ndarray, "n d"]:
        """`words` (labelled; `run`, `sentence`, `onset`, `target`) as rows of one word's counts each."""
        sentenced = (words[Column.SENTENCE] != NO_SENTENCE).to_numpy()
        keys = [words[Column.RUN], words[Column.SENTENCE]]
        size = words.groupby(keys)[Column.ONSET].transform("size").to_numpy()
        interval = words.groupby(keys)[Column.ONSET].diff().to_numpy()
        timed = sentenced & (np.nan_to_num(interval) > 0)
        counts = np.zeros((len(words), KNOWN + 1))
        counts[:, WORDS] = 1
        counts[:, SENTENCES] = np.where(sentenced, 1 / size, 0)
        counts[:, SENTENCED] = sentenced
        counts[:, LOG_IOI] = np.where(timed, np.log(np.where(timed, interval, 1)), 0)
        counts[:, IOIS] = timed
        counts[:, KNOWN] = 1
        labels = np.eye(VOCAB_SIZE)[words[Column.TARGET].to_numpy()]
        tasks = Run.tasks(words[Column.RUN]).to_numpy()
        book = np.stack([tasks == name for name in books], axis=1).astype(float)
        return np.concatenate([counts, labels, book], axis=1)


@dataclass(frozen=True)
class Picker:
    """`n` words split over sources by their share, each source's the least-cost of `draws` random sets improved by
    sampled swaps."""

    seed: int = SEED
    draws: int = DRAWS
    rounds: int = ROUNDS
    proposals: int = PROPOSALS

    @staticmethod
    def quotas(sources: "pd.Series[str]", n: int) -> dict[str, int]:
        """`n` shared over `sources`' values by their counts, largest remainders rounded up."""
        counts = sources.value_counts().sort_index()
        exact = counts.to_numpy() * min(n, len(sources)) / len(sources)
        floor = np.floor(exact).astype(int)
        floor[np.argsort(floor - exact)[: round(exact.sum()) - floor.sum()]] += 1
        return dict(zip(counts.index, floor.tolist(), strict=True))

    def cost(
        self,
        held: Float[np.ndarray, "k d"],
        total: Float[np.ndarray, " d"],
        fraction: float,
        scale: Float[np.ndarray, " c"],
    ) -> Float[np.ndarray, " k"]:
        """Each held set's summed `Stratify.gaps` against the rest of `total`, each criterion in `scale` units."""
        return (Stratify.gaps(held, total - held, fraction) / scale).sum(1)

    def chosen(self, profiles: Float[np.ndarray, "m d"], count: int) -> Int[np.ndarray, " k"]:
        """The `count` rows of one source whose summed profile best matches the rest's."""
        rows = len(profiles)
        if count >= rows:
            return np.arange(rows)
        rng = np.random.default_rng(self.seed)
        masks = np.argsort(rng.random((self.draws, rows)), axis=1) < count
        total, fraction = profiles.sum(0), count / rows
        held = masks.astype(float) @ profiles
        gaps = Stratify.gaps(held, total - held, fraction)
        median = np.median(gaps, axis=0)
        scale = np.where(median > 0, median, 1.0)
        best = int(self.cost(held, total, fraction, scale).argmin())
        return np.flatnonzero(self.swapped(profiles, masks[best].copy(), scale, rng))

    def swapped(
        self,
        profiles: Float[np.ndarray, "m d"],
        mask: Bool[np.ndarray, " m"],
        scale: Float[np.ndarray, " c"],
        rng: np.random.Generator,
    ) -> Bool[np.ndarray, " m"]:
        """`mask` improved by the best of each round's sampled swaps until a round lowers nothing."""
        total, fraction = profiles.sum(0), mask.mean()
        current = float(self.cost(profiles[mask].sum(0)[None], total, fraction, scale)[0])
        for _ in range(self.rounds):
            inside, outside = np.flatnonzero(mask), np.flatnonzero(~mask)
            i = rng.choice(inside, self.proposals)
            j = rng.choice(outside, self.proposals)
            held = profiles[mask].sum(0) + profiles[j] - profiles[i]
            costs = self.cost(held, total, fraction, scale)
            pick = int(costs.argmin())
            if costs[pick] >= current - 1e-12:
                break
            mask[[i[pick], j[pick]]] = False, True
            current = float(costs[pick])
        return mask

    def pick(self, profiles: Float[np.ndarray, "n d"], sources: "pd.Series[str]", n: int) -> Int[np.ndarray, " k"]:
        """`n` rows, each source's `quotas` share `chosen` within it, in row order."""
        groups = sources.to_numpy()
        picked = [
            np.flatnonzero(groups == source)[self.chosen(profiles[groups == source], count)]
            for source, count in self.quotas(sources, n).items()
        ]
        return np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)

    def whole(
        self, profiles: Float[np.ndarray, "n d"], keys: "pd.Series[str]", sources: "pd.Series[str]", n: int
    ) -> Int[np.ndarray, " k"]:
        """Every row of the `n` units (rows sharing a `keys` value) `pick` draws, a unit's profile its rows' sum."""
        codes, uniques = pd.factorize(keys)
        units = np.zeros((len(uniques), profiles.shape[1]))
        np.add.at(units, codes, profiles)
        first = np.unique(codes, return_index=True)[1]
        chosen = self.pick(units, pd.Series(sources.to_numpy()[first]), n)
        return np.flatnonzero(np.isin(codes, chosen))


@dataclass(frozen=True)
class Sentences:
    """Every word of the runs the samples come from (`run`, `sentence`, `onset`, `word`), read for text."""

    words: pd.DataFrame

    @staticmethod
    def keys(rows: pd.DataFrame) -> "pd.Series[str]":
        """Each of `rows`' sentence as `run/sentence`; a word outside any sentence is its own, `run@onset`."""
        run = rows[Column.RUN].astype(str)
        sentence = run + "/" + rows[Column.SENTENCE].astype(str)
        alone = run + "@" + rows[Column.ONSET].astype(str)
        return sentence.where(rows[Column.SENTENCE] != NO_SENTENCE, alone)

    def lengths(self, rows: pd.DataFrame) -> Int[np.ndarray, " n"]:
        """Each of `rows`' sentence length in words, 1 for a word outside any sentence."""
        sizes = self.words.groupby([Column.RUN, Column.SENTENCE]).size().rename("size").reset_index()
        merged = rows[[Column.RUN, Column.SENTENCE]].merge(sizes, how="left", on=[Column.RUN, Column.SENTENCE])
        sentenced = (rows[Column.SENTENCE] != NO_SENTENCE).to_numpy()
        return np.where(sentenced, merged["size"].to_numpy(), 1).astype(np.int64)

    def text(self, row: pd.Series) -> tuple[str, tuple[int, int]]:
        """`row`'s sentence as text, words in onset order, and the character span of `row`'s word in it."""
        if row[Column.SENTENCE] == NO_SENTENCE:
            return str(row[Column.WORD]), (0, len(str(row[Column.WORD])))
        same = (self.words[Column.RUN] == row[Column.RUN]) & (self.words[Column.SENTENCE] == row[Column.SENTENCE])
        sentence = self.words[same].sort_values(Column.ONSET)
        spoken = [str(word) for word in sentence[Column.WORD]]
        at = int(np.flatnonzero(sentence[Column.ONSET].to_numpy() == row[Column.ONSET])[0])
        start = sum(len(word) + 1 for word in spoken[:at])
        return " ".join(spoken), (start, start + len(spoken[at]))


@dataclass(frozen=True)
class Strata:
    """The strata a sample is browsed by: its source and its sentence's length tercile over the pool."""

    @staticmethod
    def of(rows: pd.DataFrame, lengths: Int[np.ndarray, " n"], pool: Int[np.ndarray, " p"]) -> list[dict[str, str]]:
        """Each of `rows`' source and `LENGTHS` tercile, the cuts at `pool`'s sentence-length terciles."""
        cuts = np.quantile(pool, [1 / 3, 2 / 3])
        bins = np.searchsorted(cuts, lengths, side="right")
        sources = Run.tasks(rows[Column.RUN]).map(Sources.stratum).to_numpy()
        return [{SOURCE: str(s), LENGTH: LENGTHS[b]} for s, b in zip(sources, bins, strict=True)]
