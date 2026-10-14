"""Why one occurrence of a word is decoded and another is not (megspeech-7gf.106).

perword's scene, both heads: the blend's context head and its word head (each word's own window, no context), each
finished at the tau val picks for it, on the local test's unseen-span words. Each occurrence is described by the GPT-2
surprisal of the heard word given the words before it in its run, its forced-alignment duration, its lexical stress
(CMUdict, the variant the alignment realised), the log training prior of its word and its place in its span. Top-10
hit and top-1 hit are regressed on these per head, pooled and with word identity as a fixed effect, with CIs from a
bootstrap over spans; and for `the` and `a` alone, top-1 and the P(the) - P(a) margin by surprisal, duration, vowel
form and whether the next word starts with a vowel.

    uv run python -m megspeech.evaluation.studies errors.cause m11xsp7hub6d2v3
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import cmudict
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import PRIMARY, Vocabulary
from megspeech.data.raw.events import Event, Events, Kind
from megspeech.data.raw.runs import Column
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.errors.perword import Attributes, Position, Scene
from megspeech.evaluation.studies.errors.scored import Scored

LOG = logging.getLogger(__name__)

WINDOW = 1024
"""GPT-2's context: a word is scored with up to this many tokens before it, never fewer than half of it."""
VOWELS = frozenset({"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"})
SCHWA = "ah"
RANK = {"0": 0, "2": 1, "1": 2}
"""CMUdict stress digit to rank: none, secondary, primary."""
PRIMARY_STRESS = 2
TOLERANCE = 0.005
"""Seconds an alignment onset may sit from the scored word's onset and still be that word."""
CONTINUOUS = ("surprisal", "log_duration", "log_prior")
POOLED = ("surprisal", "log_duration", "stressed", "log_prior", "first", "last")
WITHIN = ("surprisal", "log_duration", "stressed", "first", "last")
ARTICLE = ("surprisal", "log_duration", "schwa", "next_vowel")
OUTCOMES = ("top10", "top1")
ARTICLES = ("the", "a")
DRAWS = 200
STRENGTH = 10.0
"""Inverse L2 strength: weak, so it moves no estimable coefficient, but keeps an all-hit word's fixed effect finite."""
QUANTILES = 3
LEVEL = 0.95
OUT = "cause.json"


@dataclass(frozen=True)
class Surprisal:
    """A causal LM read word by word: each word's summed token surprisal, in bits, given every word before it."""

    logits: Callable[[Int[Tensor, " t"]], Float[Tensor, "t v"]]
    tokenize: Callable[[list[str]], list[list[int]]]
    bos: int
    window: int = WINDOW

    @classmethod
    def load(cls, root: DataRoot, name: str) -> "Surprisal":  # devtools-ignore: test-mirror
        path = root.foreign(name)
        model = AutoModelForCausalLM.from_pretrained(path).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, token=False)
        if tokenizer is None or tokenizer.bos_token_id is None:
            raise FileNotFoundError(name)
        return cls(
            lambda ids: model(ids[None]).logits[0],
            lambda texts: [list(ids) for ids in tokenizer(texts)["input_ids"]],
            tokenizer.bos_token_id,
        )

    def bits(self, words: Sequence[str]) -> Float[np.ndarray, " n"]:
        """Each of `words`' surprisal, the text starting at the end-of-text token and read in half-window strides."""
        pieces = self.tokenize([words[0], *(f" {word}" for word in words[1:])])
        owner = np.repeat(np.arange(len(words)), [len(piece) for piece in pieces])
        ids = torch.tensor([self.bos, *chain.from_iterable(pieces)])
        nats = torch.zeros(len(ids))
        stride = self.window // 2
        for start in range(1, len(ids), stride):
            end = min(start + stride, len(ids))
            begin = max(0, end - self.window)
            with torch.no_grad():
                read = self.logits(ids[begin:end]).float().log_softmax(-1)
            chosen = read[start - 1 - begin : end - 1 - begin].gather(1, ids[start:end, None]).squeeze(1)
            nats[start:end] = -chosen
        return np.bincount(owner, weights=nats[1:].numpy(), minlength=len(words)) / np.log(2)


@dataclass(frozen=True)
class Variant:
    """One CMUdict pronunciation of a spelling: its phones, each vowel carrying a stress digit."""

    phones: list[str]

    def rank(self) -> int:
        """The strongest stress (`RANK`) on any of its vowels."""
        return max((RANK[phone[-1]] for phone in self.phones if phone[-1] in RANK), default=0)

    def sounds(self) -> tuple[str, ...]:
        """Its phones as the forced alignment names them: lower case, no stress digit."""
        return tuple(phone.rstrip("012").lower() for phone in self.phones)


@dataclass(frozen=True)
class Stress:
    """Lexical stress from CMUdict's pronunciations of each spelling."""

    variants: dict[str, list[list[str]]]

    @classmethod
    def load(cls) -> "Stress":  # devtools-ignore: test-mirror
        return cls(cmudict.dict())

    def of(self, word: str, phones: tuple[str, ...], *, function: bool) -> float:
        """The stress rank (`RANK`) of the variant the alignment realised. Where several variants match (`the` as
        DH AH0 or DH AH1) or none does, a function word takes the least stressed (its reduced form, as connected
        speech has it) and a content word the most; NaN for a word CMUdict lacks."""
        variants = [Variant(spelled) for spelled in self.variants.get(word.lower().replace("’", "'"), [])]
        if not variants:
            return float("nan")
        ranks = [variant.rank() for variant in variants]
        matched = [variant.rank() for variant in variants if variant.sounds() == phones] or ranks
        return float(min(matched) if function else max(matched))


@dataclass(frozen=True)
class Aligned:
    """One run's forced-aligned words in order: `word`, `onset`, `duration`, `phones`, `next_vowel`, `surprisal`."""

    words: pd.DataFrame

    @classmethod
    def read(cls, path: Path) -> "Aligned":
        """`path`'s word rows, each with its phonemes (position suffix dropped) and whether the next word's first
        phoneme is a vowel."""
        table = Events(path).read(Event.SENTENCE, Event.WORD, Event.ONSET, Event.DURATION)
        phones = table[table[Event.KIND] == Kind.PHONEME]
        sounds = (
            phones.assign(phone=phones[Event.SEGMENT].str.split("_").str[0])
            .groupby([Event.SENTENCE, Event.WORD], sort=False)["phone"]
            .agg(tuple)
        )
        words = (
            table[table[Event.KIND] == Kind.WORD].join(sounds, on=[Event.SENTENCE, Event.WORD]).reset_index(drop=True)
        )
        spoken = words["phone"].map(lambda sound: sound if isinstance(sound, tuple) else ())
        first = spoken.map(lambda sound: sound[0] if sound else "")
        return cls(
            pd.DataFrame(
                {
                    "word": words[Event.SEGMENT],
                    "onset": words[Event.ONSET],
                    "duration": words[Event.DURATION],
                    "phones": spoken,
                    "next_vowel": first.shift(-1).isin(VOWELS),
                }
            )
        )

    def surprised(self, surprisal: Surprisal) -> "Aligned":
        return Aligned(self.words.assign(surprisal=surprisal.bits(self.words["word"].tolist())))

    def at(self, onset: Float[np.ndarray, " n"], word: pd.Series) -> pd.DataFrame:
        """The aligned row of each (`onset`, `word`), NaN where none lies within `TOLERANCE` with that spelling."""
        order = np.argsort(self.words["onset"].to_numpy(), kind="stable")
        times = self.words["onset"].to_numpy()[order]
        slot = np.clip(np.searchsorted(times, onset), 1, len(times) - 1)
        nearer = np.where(np.abs(times[slot - 1] - onset) <= np.abs(times[slot] - onset), slot - 1, slot)
        rows = self.words.iloc[order[nearer]].reset_index(drop=True)
        same = (np.abs(times[nearer] - onset) <= TOLERANCE) & (
            Vocabulary.spelled(rows["word"]).to_numpy() == Vocabulary.spelled(word).to_numpy()
        )
        return rows.where(pd.Series(same), other=np.nan)


@dataclass(frozen=True)
class Described:
    """One row per scored word: its features, and per head its hits and its P(the) - P(a) margin."""

    table: pd.DataFrame

    @classmethod
    def of(
        cls,
        scene: Scene,
        aligned: dict[str, Aligned],
        stress: Stress,
        attributes: Attributes,
    ) -> "Described":
        spoken = scene.spoken
        if spoken is None or scene.word is None:
            raise ValueError("the scene carries no word head or spoken words")
        found = pd.concat(
            [
                aligned[str(run)].at(rows["onset"].to_numpy(), rows["word"]).set_index(rows.index)
                for run, rows in spoken.groupby(Column.RUN)
            ]
        ).sort_index()
        scored = scene.occurrences.scored
        function = attributes.function()[scored.labels]
        rank = np.array(
            [
                stress.of(word, sounds, function=bool(flag)) if isinstance(sounds, tuple) else np.nan
                for word, sounds, flag in zip(spoken["word"], found["phones"], function, strict=True)
            ]
        )
        stressed = np.where(np.isnan(rank), np.nan, rank >= PRIMARY_STRESS)
        position = scene.occurrences.place.position
        table = pd.DataFrame(
            {
                "word": scored.labels,
                "span": scene.occurrences.place.span,
                "surprisal": found["surprisal"].astype(float),
                "log_duration": np.log(found["duration"].astype(float)),
                "stressed": stressed,
                "schwa": found["phones"].map(lambda sounds: float(isinstance(sounds, tuple) and SCHWA in sounds)),
                "next_vowel": found["next_vowel"].astype(float),
                "log_prior": np.log(scene.prior[scored.labels]),
                "first": (position == Position.FIRST).astype(float),
                "last": (position == Position.LAST).astype(float),
            }
        )
        for head, result in ((Head.CONTEXT, scored), (Head.WORD, scene.word)):
            table[f"top10-{head}"] = result.hits().astype(float)
            table[f"top1-{head}"] = (result.top1() == result.labels).astype(float)
            table[f"margin-{head}"] = cls.margin(result)
        return cls(table)

    @staticmethod
    def margin(scored: Scored) -> Float[np.ndarray, " n"]:
        """P(the) - P(a) under the softmax of each word's finished scores."""
        probs = torch.from_numpy(scored.probs).float().softmax(-1).numpy()
        return probs[:, PRIMARY.positions["the"]] - probs[:, PRIMARY.positions["a"]]

    def articles(self, head: str) -> pd.DataFrame:
        """For `the` and `a`: n, top-1 rate and mean margin, overall and by within-word tertile of surprisal and
        duration, by vowel form and by whether the next word starts with a vowel."""
        rows = self.table[self.table["word"].isin([PRIMARY.positions[word] for word in ARTICLES])]
        named = rows.assign(word=rows["word"].map(dict(enumerate(PRIMARY.words))))
        pieces = [self._summary(named, head, "all", pd.Series("all", index=named.index))]
        for column in ("surprisal", "log_duration"):
            third = named.groupby("word")[column].transform(
                lambda values: pd.qcut(values, QUANTILES, labels=["low", "mid", "high"])
            )
            pieces.append(self._summary(named, head, column, third.astype(object)))
        pieces.extend(
            self._summary(named, head, column, named[column].map({0.0: "no", 1.0: "yes"}))
            for column in ("schwa", "next_vowel")
        )
        return pd.concat(pieces)

    @staticmethod
    def _summary(rows: pd.DataFrame, head: str, by: str, value: pd.Series) -> pd.DataFrame:
        grouped = rows.assign(value=value).dropna(subset="value").groupby(["word", "value"], observed=True)
        frame = pd.DataFrame(
            {
                "n": grouped.size(),
                "top1": grouped[f"top1-{head}"].mean(),
                "margin": grouped[f"margin-{head}"].mean(),
            }
        )
        return frame.assign(by=by).set_index("by", append=True).reorder_levels(["word", "by", "value"])

    def subset(self, words: Sequence[str]) -> "Described":
        return Described(self.table[self.table["word"].isin([PRIMARY.positions[word] for word in words])])


@dataclass(frozen=True)
class Regression:
    """Logistic regression of an outcome on `columns` (continuous ones z-scored, so a coefficient is log-odds per SD;
    binary ones per unit), with word identity as a fixed effect when `fixed`."""

    columns: tuple[str, ...]
    fixed: bool

    def design(self, table: pd.DataFrame) -> Float[np.ndarray, "n f"]:
        parts = [table[list(self.columns)].to_numpy(dtype=float)]
        if self.fixed:
            parts.append(pd.get_dummies(table["word"]).to_numpy(dtype=float))
        return np.concatenate(parts, axis=1)

    def held(self, table: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """`table`'s rows with every column and the outcome known, continuous columns z-scored over them."""
        rows = table.dropna(subset=[*self.columns, outcome]).copy()
        for column in set(self.columns) & set(CONTINUOUS):
            rows[column] = (rows[column] - rows[column].mean()) / rows[column].std()
        return rows

    def coefficients(self, rows: pd.DataFrame, outcome: str) -> Float[np.ndarray, " f"]:
        model = LogisticRegression(C=STRENGTH, max_iter=2000)
        model.fit(self.design(rows), rows[outcome].to_numpy())
        return model.coef_[0, : len(self.columns)]

    def fit(self, table: pd.DataFrame, outcome: str, draws: int, seed: int) -> pd.DataFrame:
        """Each column's coefficient and its `LEVEL` percentile interval over `draws` resamples of whole spans (the
        words of one span share its MEG and its context, so they are not independent)."""
        rows = self.held(table, outcome)
        point = self.coefficients(rows, outcome)
        spans = rows.groupby("span").indices
        keys = np.array(list(spans))
        generator = np.random.default_rng(seed)
        boot = np.stack(
            [
                self.coefficients(
                    rows.iloc[np.concatenate([spans[key] for key in generator.choice(keys, len(keys))])], outcome
                )
                for _ in range(draws)
            ]
        )
        tail = (1 - LEVEL) / 2
        return pd.DataFrame(
            {
                "coef": point,
                "lo": np.quantile(boot, tail, axis=0),
                "hi": np.quantile(boot, 1 - tail, axis=0),
                "n": len(rows),
                "rate": rows[outcome].mean(),
            },
            index=list(self.columns),
        )


class Study:
    """Fits the regressions and the article table, logs and saves them."""

    def __init__(self, described: Described, out: Path, draws: int = DRAWS) -> None:
        self.described, self.out, self.draws = described, out, draws

    def run(self) -> dict[str, object]:
        table = self.described.table
        LOG.info("occurrences %d, with every feature %d", len(table), len(table.dropna(subset=list(POOLED))))
        record: dict[str, object] = {"n": len(table)}
        fits: dict[str, object] = {}
        specs = {"pooled": Regression(POOLED, fixed=False), "word-fixed": Regression(WITHIN, fixed=True)}
        for head in Head:
            for outcome in OUTCOMES:
                for name, spec in specs.items():
                    frame = spec.fit(table, f"{outcome}-{head}", self.draws, seed=0)
                    LOG.info("%s head, %s, %s\n%s", head, outcome, name, frame.round(3).to_string())
                    fits[f"{head}/{outcome}/{name}"] = json.loads(frame.to_json(orient="index"))
        articles = self.described.subset(ARTICLES)
        spec = Regression(ARTICLE, fixed=True)
        for head in Head:
            frame = spec.fit(articles.table, f"top1-{head}", self.draws, seed=0)
            LOG.info("the/a top-1, %s head\n%s", head, frame.round(3).to_string())
            fits[f"{head}/the-a-top1"] = json.loads(frame.to_json(orient="index"))
            buckets = self.described.articles(head)
            LOG.info("the/a by feature, %s head\n%s", head, buckets.round(3).to_string())
            record[f"articles-{head}"] = json.loads(buckets.rename(index=str).reset_index().to_json(orient="records"))
        record["fits"] = fits
        self.out.write_text(json.dumps(record, indent=1))
        return record

    @staticmethod
    def _events(root: DataRoot, run: str) -> Path:
        return next(path for path in sorted(root.raw().rglob(f"{run}_events.tsv")) if path.stat().st_size)

    @classmethod
    def main(cls) -> None:  # devtools-ignore: test-mirror
        parser = Scene.parser(__doc__)
        parser.add_argument("--lm", default="gpt2-large")
        parser.add_argument("--draws", type=int, default=DRAWS)
        root, args, scene = Scene.opened(parser)
        if scene.spoken is None:
            raise ValueError("the scene carries no spoken words")
        surprisal = Surprisal.load(root, args.lm)
        aligned = {
            run: Aligned.read(cls._events(root, run)).surprised(surprisal) for run in scene.spoken[Column.RUN].unique()
        }
        described = Described.of(scene, aligned, Stress.load(), Attributes.of(root))
        Study(described, root.models() / args.name / OUT, args.draws).run()


if __name__ == "__main__":
    Study.main()
