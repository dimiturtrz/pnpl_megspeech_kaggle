"""Frozen t5-large embeddings of the 50 words: the semantic targets d'Ascoli et al. align MEG to.

uv run python -m megspeech.prototypes
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel

from core.paths import DataRoot
from megspeech.vocabulary import PRIMARY

MODEL = "google-t5/t5-large"
LAYER = 12
"""Hidden state 12 of t5-large's 25: the middle of the encoder, as in d'Ascoli et al. (`layers=0.5`)."""
LOG = logging.getLogger(__name__)


class Encoded(Protocol):
    @property
    def hidden_states(self) -> tuple[Tensor, ...]: ...


Tokenizer = Callable[..., dict[str, Int[Tensor, "1 k"]]]
Encoder = Callable[..., Encoded]


@dataclass(frozen=True)
class Prototypes:
    vectors: Float[Tensor, "v d"]

    @classmethod
    def encode(cls, words: tuple[str, ...], tokenizer: Tokenizer, encoder: Encoder, layer: int) -> "Prototypes":
        """Each word alone, without special tokens, mean-pooled over its tokens at `layer`."""
        rows = []
        with torch.no_grad():
            for word in words:
                ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
                hidden = encoder(input_ids=ids, output_hidden_states=True).hidden_states[layer]
                rows.append(hidden[0].mean(dim=0))
        return cls(torch.stack(rows).float())

    @staticmethod
    def path(root: DataRoot) -> Path:
        return root.processed() / "prototypes" / f"{MODEL.split('/')[-1]}-layer{LAYER}.npy"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors.numpy())
        return path

    @classmethod
    def load(cls, path: Path) -> "Prototypes":
        return cls(torch.from_numpy(np.load(path)))

    @staticmethod
    def main() -> None:
        """Download t5-large once and write the 50 words' prototypes under processed/prototypes."""
        argparse.ArgumentParser(description=__doc__).parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=False)
        if tokenizer is None:
            raise FileNotFoundError(MODEL)
        encoder = T5EncoderModel.from_pretrained(MODEL, token=False).eval()
        prototypes = Prototypes.encode(PRIMARY.words, tokenizer, encoder, LAYER)
        LOG.info("wrote %s", prototypes.save(Prototypes.path(DataRoot.from_config())))


if __name__ == "__main__":
    Prototypes.main()
