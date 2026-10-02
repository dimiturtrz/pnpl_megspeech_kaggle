"""Which per-word encoder the sentence model reads: the CNN trained from scratch, pretrained MEG-XL, or MEG-XL's
features computed ahead of training by its frozen trunk."""

from enum import StrEnum

from core.paths import DataRoot
from megspeech.model.cached import Cached
from megspeech.model.megxl_words import Words
from megspeech.model.wordnet import ConvConfig, WordNet


class Encoder(StrEnum):
    CONV = "conv"
    MEGXL = "megxl"
    CACHED = "cached"

    def build(self, config: ConvConfig) -> WordNet | Words | Cached:
        """A fresh CNN, MEG-XL's trunk with its released weights under the configured data root, or a head over
        the features `DataConfig.features` names."""
        builders = {
            Encoder.CONV: lambda: WordNet(config),
            Encoder.MEGXL: lambda: Words.pretrained(DataRoot.from_config(), config),
            Encoder.CACHED: lambda: Cached(config),
        }
        return builders[self]()
