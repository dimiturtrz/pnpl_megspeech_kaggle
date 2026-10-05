"""Whether an external masked language model knows the Deep vocabulary's order beyond what the context model learned
from training text: each scored word's context logits fused with the masked LM's log-probability of every vocabulary
word at its slot, the span's other vocabulary words filled with the context head's guesses and every out-of-vocabulary
word a mask (its identity is unknown on the Kaggle test), on a grid of fusion weights.

Sherlock1 is public-domain text a pretrained LM may have memorised; TheMoth is the clean read. A submission fused this
way needs an owner rules decision (megspeech-7gf.60).

    uv run python -m megspeech.evaluation.studies.fusion ens-ema24
"""

import argparse
import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, PRIMARY
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.scoring.score import Head
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

WEIGHTS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
"""Fusion weights on the masked LM's log-probability; 0 is the context head alone."""
QUERIES = 256
"""Masked-LM queries per forward pass."""
MASKED_LM = "bert-base-uncased"


@dataclass(frozen=True)
class Specials:
    """The masked LM's special token ids."""

    cls: int
    sep: int
    mask: int
    pad: int


@dataclass(frozen=True)
class MaskedPrior:
    """A masked LM read at one slot per query over the vocabulary words' first tokens."""

    model: nn.Module
    ids: Int[Tensor, " v"]
    specials: Specials

    @classmethod
    def load(cls, name: str, device: torch.device) -> "MaskedPrior":  # devtools-ignore: test-mirror
        tokenizer = AutoTokenizer.from_pretrained(name, token=False)
        if tokenizer is None:
            raise FileNotFoundError(name)
        model = AutoModelForMaskedLM.from_pretrained(name, token=False).to(device).eval()
        words = [word.replace("’", "'") for word in PRIMARY.words]
        ids = torch.tensor([tokenizer(word, add_special_tokens=False)["input_ids"][0] for word in words], device=device)
        specials = Specials(
            tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id
        )
        return cls(model, ids, specials)

    def queries(
        self, index: Int[Tensor, "g w"], y: Int[Tensor, " n"], guess: Int[Tensor, " n"]
    ) -> tuple[Int[Tensor, "q l"], Int[Tensor, " q"], Int[Tensor, " q"]]:
        """One query per labelled word: its span as tokens, the word masked, other labelled words as their `guess`,
        the rest masked; with each query's mask position and the word it scores."""
        words = index.clamp(min=0)
        present = index != PAD
        labelled = present & (y[words] != IGNORE)
        tokens = torch.where(labelled, self.ids[guess[words]], self.specials.mask)
        tokens = torch.where(present, tokens, self.specials.pad)
        span, slot = torch.nonzero(labelled, as_tuple=True)
        rows = tokens[span].clone()
        rows[torch.arange(len(span), device=rows.device), slot] = self.specials.mask
        length = present.sum(dim=1)[span]
        edge = torch.full_like(rows[:, :1], self.specials.pad)
        rows = torch.cat([edge.clone().fill_(self.specials.cls), rows, edge], dim=1)
        rows[torch.arange(len(span), device=rows.device), length + 1] = self.specials.sep
        return rows, slot + 1, words[span, slot]

    def __call__(
        self, index: Int[Tensor, "g w"], y: Int[Tensor, " n"], guess: Int[Tensor, " n"]
    ) -> Float[Tensor, "n v"]:
        """Every word's log-probability over the vocabulary from the masked LM; zero for unlabelled words."""
        rows, at, word = self.queries(index, y, guess)
        out = torch.zeros(len(y), len(self.ids), device=rows.device)
        with torch.no_grad():
            for part in torch.arange(len(rows), device=rows.device).split(QUERIES):
                tokens = rows[part]
                logits = self.model(input_ids=tokens, attention_mask=tokens != self.specials.pad).logits
                out[word[part]] = logits[torch.arange(len(part), device=rows.device), at[part]][:, self.ids].float()
        return out.log_softmax(-1)


@dataclass(frozen=True)
class Fusion:
    """A loaded model's validation domains read with and without the masked LM."""

    loaded: LoadedModel
    prior: MaskedPrior

    def domain(self, split: Split) -> dict[float, float]:  # devtools-ignore: test-mirror
        """BAcc@10 on `split`'s unseen-span words at every fusion weight."""
        evaluator = self.loaded.evaluator
        context = evaluator.logits(self.loaded.model, split)[Head.CONTEXT]
        lm = self.prior(split.index, split.y, context.argmax(dim=1))
        novel = split.novel(evaluator.seen)
        alone = evaluator.evaluate(lm[novel], split.y[novel], self.loaded.log_prior)
        LOG.info("masked LM alone %.4f over %d words", alone.bacc10, alone.n)
        return {
            weight: evaluator.evaluate((context + weight * lm)[novel], split.y[novel], self.loaded.log_prior).bacc10
            for weight in WEIGHTS
        }

    def table(self, held: dict[str, list[Run]]) -> dict[str, dict[float, float]]:  # devtools-ignore: test-mirror
        """Every counted domain's scores by fusion weight."""
        domains, _ = self.loaded.domains(held)
        evaluator = self.loaded.evaluator
        counted = {name: s for name, s in domains.items() if len(s.novel(evaluator.seen)) >= evaluator.min_novel}
        return {name: self.domain(split) for name, split in counted.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--model", default=MASKED_LM)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    loaded = Loader.of(root, args.names, Scoring(fallback=Fallback(Unseen.DEEP)), device)
    held = loaded.data.held(Run.discover(root.raw()))
    table = Fusion(loaded, MaskedPrior.load(args.model, device)).table(held)
    for name, scores in table.items():
        LOG.info("%s %s", name, {weight: round(score, 4) for weight, score in scores.items()})
    for weight in WEIGHTS:
        LOG.info("weight %s macro %.4f", weight, sum(s[weight] for s in table.values()) / len(table))


if __name__ == "__main__":
    main()
