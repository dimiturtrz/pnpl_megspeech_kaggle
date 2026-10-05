"""Frozen GPT-2 hidden states of every word of each training run's transcript, read in running context: the dense
in-context text target of the semantic auxiliary (megspeech-7gf.88), for every word, in or out of the 50.

A word's target is the middle-layer state of its last token given everything said before it in the run (a sliding
`WINDOW` of tokens, each kept token read with at least half a window of left context), z-scored per dimension over
the run,
a row of `<run>.npy` aligned with the run's word table (`RunCache.words`), NaN where a word has no token. Only trained
runs' transcripts are read; validation and test text never is.

uv run python -m megspeech.objective.targets.context
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoTokenizer, GPT2Model

from core.paths import DataRoot
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import Targets
from megspeech.objective.targets.t5 import Encoder

MODEL = "gpt2-large"
LAYER = 18
"""Hidden state 18 of gpt2-large's 36: the middle of the decoder, where lexical-semantic content peaks."""
WINDOW = 1024
"""Tokens one forward pass reads: gpt2's whole context."""
FLOOR = 1e-6
"""Least per-dimension spread a z-score divides by."""
DEVICE = "cuda"
LOG = logging.getLogger(__name__)

Tokenizer = Callable[[list[str]], dict[str, list[list[int]]]]


@dataclass(frozen=True)
class Transcript:
    """One run's words, in the order they were heard."""

    words: tuple[str, ...]

    def tokens(self, tokenizer: Tokenizer) -> tuple[Int[Tensor, " k"], Int[np.ndarray, " k"]]:
        """The token ids of the whole transcript, each word a space and its own tokens, and the word each belongs to."""
        pieces = tokenizer([f" {word}" for word in self.words])["input_ids"]
        owner = np.repeat(np.arange(len(pieces)), [len(piece) for piece in pieces])
        return torch.tensor([token for piece in pieces for token in piece], dtype=torch.long), owner

    @staticmethod
    def states(ids: Int[Tensor, " k"], encoder: Encoder, layer: int, window: int) -> Float[Tensor, "k d"]:
        """Each token's `layer` state, windows of `window` tokens advancing by half, each later window keeping only
        the tokens past that half."""
        stride = window // 2
        kept: list[Tensor] = []
        start = 0
        with torch.no_grad(), torch.autocast(ids.device.type, dtype=torch.bfloat16):
            while True:
                stop = min(start + window, len(ids))
                hidden = encoder(input_ids=ids[None, start:stop], output_hidden_states=True).hidden_states[layer][0]
                kept.append(hidden[(stride if start else 0) :].float())
                if stop == len(ids):
                    return torch.cat(kept)
                start += stride

    def encode(
        self, tokenizer: Tokenizer, encoder: Encoder, device: torch.device, layer: int = LAYER, window: int = WINDOW
    ) -> Targets:
        """Each word's last token's state, z-scored over the run's words; NaN for a word that has no token."""
        ids, owner = self.tokens(tokenizer)
        if not len(ids):
            return Targets(np.full((len(self.words), 1), np.nan, dtype=np.float16))
        states = self.states(ids.to(device), encoder, layer, window).cpu().numpy()
        last = np.full(len(self.words), -1, dtype=np.int64)
        np.maximum.at(last, owner, np.arange(len(owner)))
        vectors = np.where((last >= 0)[:, None], states[last], np.nan)
        spread = np.nanstd(vectors, axis=0)
        return Targets(((vectors - np.nanmean(vectors, axis=0)) / np.maximum(spread, FLOOR)).astype(np.float16))

    @staticmethod
    def of(run: Run) -> "Transcript":
        return Transcript(tuple(str(word) for word in RunCache.words(run)["word"]))

    @staticmethod
    def write(root: DataRoot, tokenizer: Tokenizer, encoder: Encoder, device: torch.device) -> Path:
        """Every trained run's words' targets under processed/speech/gpt2-large, those already there kept."""
        directory = Targets.directory(root, MODEL)
        for run in DataConfig().trained(Run.discover(root.raw())):
            path = directory / f"{run.name}.npy"
            if path.exists():
                continue
            targets = Transcript.of(run).encode(tokenizer, encoder, device)
            targets.save(path)
            LOG.info("%s: %d of %d words", run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Read every trained run's transcript through gpt2-large on the GPU and write its words' targets."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, device = DataRoot.from_config(), torch.device(DEVICE)
        loaded = AutoTokenizer.from_pretrained(root.foreign(MODEL), token=False)
        if loaded is None:
            raise FileNotFoundError(MODEL)
        encoder = GPT2Model.from_pretrained(root.foreign(MODEL)).to(device).eval()
        LOG.info("wrote %s", Transcript.write(root, lambda words: dict(loaded(words)), encoder, device))


if __name__ == "__main__":
    Transcript.main()
