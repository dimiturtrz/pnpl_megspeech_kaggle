"""Which per-word encoder the sentence model reads: the CNN trained from scratch, or pretrained MEG-XL."""

from enum import StrEnum

from core.paths import DataRoot
from megspeech.model.megxl_words import Words
from megspeech.model.wordnet import ConvConfig, WordNet


class Encoder(StrEnum):
    CONV = "conv"
    MEGXL = "megxl"

    def build(self, config: ConvConfig) -> WordNet | Words:
        """A fresh CNN, or MEG-XL's trunk with its released weights under the configured data root."""
        if self is Encoder.MEGXL:
            return Words.pretrained(DataRoot.from_config(), config)
        return WordNet(config)
