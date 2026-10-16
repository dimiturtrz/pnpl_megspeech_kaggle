"""A causal LM's log-probability of each candidate word given a left context, optionally followed by the true words
after it (megspeech-7gf.61): the prior the fusion study multiplies a MEG base's word posterior by.

Every candidate is one batch row continuing one cached context pass, so a context costs one pass plus a short
continuation per candidate.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from core.paths import DataRoot


@dataclass(frozen=True)
class LanguageModel:
    """A causal LM and its tokenizer; a text starts at the end-of-text token so an empty context is a sentence start."""

    model: PreTrainedModel
    tokenize: Callable[[list[str]], list[list[int]]]
    bos: int

    @classmethod
    def load(cls, root: DataRoot, name: str, device: torch.device) -> "LanguageModel":  # devtools-ignore: test-mirror
        path = root.foreign(name)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, token=False)
        if tokenizer is None or tokenizer.bos_token_id is None:
            raise FileNotFoundError(name)
        return cls(model, lambda texts: [list(ids) for ids in tokenizer(texts)["input_ids"]], tokenizer.bos_token_id)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def pieces(self, words: Sequence[str]) -> list[list[int]]:
        """Each word's token ids, spelled as it follows a space."""
        return self.tokenize([f" {word}" for word in words]) if words else []

    def opening(self, context: Sequence[str]) -> list[int]:
        """The ids a text of `context` words starts with."""
        return [self.bos, *(token for piece in self.pieces(context) for token in piece)]

    def score(
        self, context: Sequence[str], candidates: Sequence[str], suffix: Sequence[str] = ()
    ) -> Float[Tensor, "c s"]:
        """Column `j` is each candidate's log-probability, then the first `j` words of `suffix` following it, given
        `context`: column 0 the candidate alone, the last the candidate and the whole `suffix`."""
        head = torch.tensor([self.opening(context)], device=self.device)
        own, after_words = self.pieces(candidates), self.pieces(suffix)
        flat = [token for piece in after_words for token in piece]
        reach = list(itertools.accumulate(map(len, after_words)))
        tails = [[*piece, *flat] for piece in own]
        ends = torch.tensor(
            [[len(piece), *(len(piece) + length for length in reach)] for piece in own], device=self.device
        )
        width = max(map(len, tails))
        ids = torch.tensor([tail + [0] * (width - len(tail)) for tail in tails], device=self.device)
        with torch.no_grad():
            opened = self.model(head, use_cache=True)
            first = opened.logits[0, -1].float().log_softmax(-1)
            cache = opened.past_key_values
            cache.batch_repeat_interleave(len(tails))
            mask = torch.ones(len(tails), head.shape[1] + width, device=self.device, dtype=torch.long)
            after = self.model(ids, past_key_values=cache, attention_mask=mask).logits.float().log_softmax(-1)
        return self.cumulative(first, after, ids, ends)

    @staticmethod
    def cumulative(
        first: Float[Tensor, " v"], after: Float[Tensor, "c w v"], ids: Int[Tensor, "c w"], ends: Int[Tensor, "c s"]
    ) -> Float[Tensor, "c s"]:
        """Token log-probabilities summed up to each of `ends` tokens, from the first token's `first` and the rest's
        `after` (position `i` predicts token `i + 1`)."""
        inner = after[:, :-1].gather(2, ids[:, 1:, None]).squeeze(2)
        steps = torch.cat([first[ids[:, :1]], inner], dim=1).cumsum(dim=1)
        return steps.gather(1, ends - 1)
