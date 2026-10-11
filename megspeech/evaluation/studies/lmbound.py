"""Whether a causal LM that sees only the in-vocabulary neighbours of a word can add to the finished context head: the
upper bound of LM fusion on the Kaggle test, where any neighbour the model could name is one of the 50 words.

Every novel word of validation and the local test gets its span's words as the model groups them, shown three ways
(`MASKS`): `full` (every true neighbour: the oracle the in-vocabulary bound sits under), `backoff` (only the unbroken
run of in-vocabulary words next to it, either side) and `drop` (every in-vocabulary word of the span in order, the
others left out). GPT-2-large scores the 50 candidates given the left words (`pmi`) and given the left and the next
`RIGHT` right words (`pmi-right`); the unigram is taken out, so the channel is context evidence without the frequency
prior the finish already has. The fused finish is `context - tau * log_prior + lam * pmi`, cross-fit on validation
halves against the tau-only finish exactly as `crowding` does, then refit on all of validation and reported on the
local test. An instrument: the `backoff` or `drop` arm reaching +0.03 keeps LM fusion alive.

    uv run python -m megspeech.evaluation.studies.lmbound m11xsp7hub6d2v3
"""

import argparse
import itertools
import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import PRIMARY, Vocabulary
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Placed
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.probabilities import PROBS, SECTIONS, Probabilities
from megspeech.evaluation.studies.crowding import TAUS, TEST, VAL, Arm, Finish, Rows, Section
from megspeech.evaluation.studies.crowding import Study as Crossfit
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

RIGHT = 2
"""Right neighbours the `pmi-right` channel reads."""
LAMS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CANDIDATES = tuple(word.replace("’", "'") for word in PRIMARY.words)
LOG_EVERY = 1000
OUT = "lmbound.json"

Context = tuple[tuple[str, ...], tuple[str, ...]]
"""The left words shown, then the right words shown."""


class Mask:
    """Which of a word's span neighbours the LM is shown: every one."""

    name = "full"

    def shown(self, words: Sequence[str], at: int) -> Context:
        return tuple(words[:at]), tuple(words[at + 1 : at + 1 + RIGHT])

    @staticmethod
    def known(word: str) -> bool:
        return Vocabulary.spell(word) in PRIMARY.positions


class Backoff(Mask):
    """The unbroken run of in-vocabulary words beside the word, either side."""

    name = "backoff"

    @override
    def shown(self, words: Sequence[str], at: int) -> Context:
        left = list(itertools.takewhile(self.known, reversed(words[:at])))[::-1]
        right = list(itertools.takewhile(self.known, words[at + 1 :]))[:RIGHT]
        return tuple(left), tuple(right)


class Drop(Mask):
    """Every in-vocabulary word of the span in order; the rest left out."""

    name = "drop"

    @override
    def shown(self, words: Sequence[str], at: int) -> Context:
        left = [word for word in words[:at] if self.known(word)]
        right = [word for word in words[at + 1 :] if self.known(word)][:RIGHT]
        return tuple(left), tuple(right)


MASKS: tuple[Mask, ...] = (Mask(), Backoff(), Drop())


@dataclass(frozen=True)
class Spelling:
    """Each word token's most frequent written form."""

    forms: dict[int, str]

    @classmethod
    def of(cls, words: pd.Series) -> "Spelling":
        counts = Counter(zip(Vocabulary.token(words).tolist(), words.astype(str).tolist(), strict=True))
        forms: dict[int, str] = {}
        for (token, form), _ in counts.most_common():
            forms.setdefault(token, form)
        return cls(forms)

    def spans(self, split: Split, rows: Int[Tensor, " n"]) -> list[tuple[list[str], int]]:
        """Each of `rows`' span text, its first word capitalised, and the row's place in it."""
        token = split.token.cpu().tolist()
        place: dict[int, tuple[list[str], int]] = {}
        for span in split.index.cpu().tolist():
            members = [row for row in span if row != PAD]
            text = [self.forms[token[row]] for row in members]
            text[0] = text[0][:1].upper() + text[0][1:]
            place.update({row: (text, at) for at, row in enumerate(members)})
        return [place[row] for row in rows.tolist()]


@dataclass(frozen=True)
class Causal:
    """A causal LM and its tokenizer; a text starts at the end-of-text token so an empty context is a sentence start."""

    model: PreTrainedModel
    tokenize: Callable[[list[str]], list[list[int]]]
    bos: int

    @classmethod
    def load(cls, placed: Placed) -> "Causal":  # devtools-ignore: test-mirror
        path = placed.root.foreign(placed.name)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(placed.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, token=False)
        if tokenizer is None or tokenizer.bos_token_id is None:
            raise FileNotFoundError(placed.name)
        return cls(model, lambda texts: [list(ids) for ids in tokenizer(texts)["input_ids"]], tokenizer.bos_token_id)

    def pieces(self, words: Sequence[str]) -> list[list[int]]:
        return self.tokenize([f" {word}" for word in words]) if words else []

    def score(self, key: Context) -> Float[Tensor, "c j"]:  # devtools-ignore: test-mirror
        """Each candidate's log-probability given the left words, then with each following right word, the columns
        past the last right word repeating the last."""
        left, right = key
        device = self.model.device
        head = torch.tensor([[self.bos, *(token for piece in self.pieces(left) for token in piece)]], device=device)
        own, after = self.pieces(CANDIDATES), self.pieces(right)
        flat = [token for piece in after for token in piece]
        reach = list(itertools.accumulate(map(len, after)))
        tails = [[*piece, *flat] for piece in own]
        ends = torch.tensor([[len(piece), *(len(piece) + length for length in reach)] for piece in own], device=device)
        width = max(map(len, tails))
        ids = torch.tensor([tail + [0] * (width - len(tail)) for tail in tails], device=device)
        with torch.no_grad():
            opened = self.model(head, use_cache=True)
            first = opened.logits[0, -1].float().log_softmax(-1)
            cache = opened.past_key_values
            cache.batch_repeat_interleave(len(tails))
            mask = torch.ones(len(tails), head.shape[1] + width, device=device, dtype=torch.long)
            steps = self.model(ids, past_key_values=cache, attention_mask=mask).logits.float().log_softmax(-1)
        raw = self.cumulative(first, steps, ids, ends).cpu()
        return torch.cat([raw, raw[:, -1:].expand(-1, RIGHT + 1 - raw.shape[1])], dim=1)

    @staticmethod
    def cumulative(
        first: Float[Tensor, " v"], after: Float[Tensor, "c w v"], ids: Int[Tensor, "c w"], ends: Int[Tensor, "c s"]
    ) -> Float[Tensor, "c s"]:
        """Token log-probabilities summed up to each of `ends` tokens, from the first token's `first` and the rest's
        `after` (position `i` predicts token `i + 1`)."""
        inner = after[:, :-1].gather(2, ids[:, 1:, None]).squeeze(2)
        steps = torch.cat([first[ids[:, :1]], inner], dim=1).cumsum(dim=1)
        return steps.gather(1, ends - 1)


@dataclass(frozen=True)
class Table:
    """The LM's candidate scores per distinct context, and with no context at all."""

    scores: dict[Context, Float[Tensor, "c j"]]
    unigram: Float[Tensor, " c"]

    @classmethod
    def scored(cls, score: Callable[[Context], Tensor], wanted: Sequence[Context]) -> "Table":
        distinct = list(dict.fromkeys(wanted))
        scores: dict[Context, Tensor] = {}
        for count, key in enumerate(distinct):
            scores[key] = score(key)
            if count % LOG_EVERY == 0:
                LOG.info("scored %d of %d contexts", count, len(distinct))
        return cls(scores, score(((), ()))[:, 0])

    def pmi(self, contexts: Sequence[Context], column: int) -> Float[Tensor, "n c"]:
        """Each context's candidate log-probabilities at `column`, normalised over the 50, less the unigram's."""
        rows = torch.stack([self.scores[key][:, column] for key in contexts]).log_softmax(-1)
        return rows - self.unigram.log_softmax(-1)


@dataclass(frozen=True)
class Bound:
    """The fused arms of one section: each mask's contexts for every graded word of every domain."""

    section: Section
    contexts: dict[str, dict[str, list[Context]]]
    """By mask, then domain."""

    @classmethod
    def of(cls, section: Section, text: dict[str, list[tuple[list[str], int]]]) -> "Bound":
        contexts = {
            mask.name: {name: [mask.shown(words, at) for words, at in text[name]] for name in section.domains}
            for mask in MASKS
        }
        return cls(section, contexts)

    def wanted(self) -> list[Context]:
        return [key for domains in self.contexts.values() for keys in domains.values() for key in keys]

    def fused(self, table: Table, mask: str, column: int) -> dict[str, Rows]:
        """Each domain's words with the LM's `pmi` at `column` in place of the other head."""
        domains = self.section.domains
        return {
            name: Rows(rows.head, table.pmi(self.contexts[mask][name], column), rows.labels)
            for name, rows in domains.items()
        }

    def masks(self) -> tuple[str, ...]:
        """The masks this bound holds contexts under."""
        return tuple(self.contexts)


@dataclass(frozen=True)
class Study:
    """Every mask and channel of LM fusion against the tau-only finish."""

    val: Bound
    test: Bound | None

    def run(self, table: Table, seed: int) -> dict[str, object]:
        arms = (Arm.all()[0], Arm("lm", tuple(Finish(tau, 0.0, lam) for tau, lam in itertools.product(TAUS, LAMS))))
        result: dict[str, object] = {}
        for mask, (channel, column) in itertools.product(self.val.masks(), (("pmi", 0), ("pmi-right", RIGHT))):
            val = self.val.fused(table, mask, column)
            LOG.info("%s %s", mask, channel)
            crossfit = Crossfit(Section(val, {}, self.val.section.log_prior), None)
            entry = {"val": crossfit.crossfit(arms, val, seed)}
            if self.test is not None:
                entry["test"] = crossfit.transfer(arms, val, self.test.fused(table, mask, column))
            result[f"{mask}/{channel}"] = entry
        return result

    @staticmethod
    def text(  # devtools-ignore: test-mirror
        root: DataRoot, member: str, section: str, saved: Probabilities
    ) -> dict[str, list[tuple[list[str], int]]]:
        """Each graded domain's novel words' span text, from `member`'s own recorded data config."""
        directories = Ensemble.trained(root.models(), member)
        data = DataConfig.shared(directories, DataConfig.saved(directories[0]).grouping)
        cpu = torch.device("cpu")
        if section == VAL:
            splits = Corpus.load(data, root, cpu).validation
        else:
            partition = next(part for part, name in SECTIONS.items() if name == section)
            reader = Reader(root, data, cpu, partition=partition)
            splits = reader.domains(reader.ruled(Run.discover(root.raw())))[0]
        spelling = Spelling.of(data.run_cache(root).published()["word"])
        return {name: spelling.spans(splits[name], words.ids[words.novel]) for name, words in saved.domains.items()}

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/ whose members saved probs.pt")
        parser.add_argument("--lm", default="gpt2-large")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        directories = Ensemble.trained(root.models(), args.name)
        member = directories[0].name
        bounds: dict[str, Bound] = {}
        for section in (VAL, TEST):
            saved = Probabilities.load(directories[0] / PROBS, section)
            loaded = Section.load(directories, section, {})
            bounds[section] = Bound.of(loaded, Study.text(root, member, section, saved))
        cache = root.models() / args.name / f"lmbound-{args.lm}.pt"
        table = Study.table(root, args, cache, [*bounds[VAL].wanted(), *bounds[TEST].wanted()])
        result = Study(bounds[VAL], bounds[TEST]).run(table, args.seed)
        out = root.models() / args.name / OUT
        out.write_text(json.dumps(result, indent=1))
        LOG.info("wrote %s", out)

    @staticmethod
    def table(  # devtools-ignore: test-mirror
        root: DataRoot, args: argparse.Namespace, cache: Path, wanted: list[Context]
    ) -> Table:
        if cache.exists():
            saved = torch.load(cache)
            return Table(saved["scores"], saved["unigram"])
        lm = Causal.load(Placed(root, args.lm, torch.device(args.device)))
        table = Table.scored(lm.score, wanted)
        torch.save({"scores": table.scores, "unigram": table.unigram}, cache)
        return table


if __name__ == "__main__":
    Study.main()
