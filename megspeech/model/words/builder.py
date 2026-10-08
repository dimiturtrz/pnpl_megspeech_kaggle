"""The word encoder each `Encoder` names."""

from collections.abc import Callable

from core.paths import DataRoot
from megspeech.data.constants import CHANNELS
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.conv import ConvWords
from megspeech.model.words.experts import ExpertWords
from megspeech.model.words.megxl import POOLED, RESOLVED, FeatureWords
from megspeech.model.words.spanned import SpanWords

BUILDERS: dict[Encoder, Callable[[ContextConfig, int], WordEncoder]] = {
    Encoder.SPANNED: lambda context, channels: SpanWords.pretrained(
        DataRoot.from_config(), context.conv, context.spanned, channels
    ),
    Encoder.CONV: lambda context, channels: ConvWords(context.conv, channels),
    Encoder.CACHED: lambda context, _: FeatureWords(context.conv, POOLED),
    Encoder.RESOLVED: lambda context, _: FeatureWords(context.conv, RESOLVED),
    Encoder.TRAINED: lambda context, _: FeatureWords(context.conv, context.conv.width),
    Encoder.EXPERTS: lambda context, channels: ExpertWords(context.conv, channels, context.crops),
}


class WordBuilder:
    """Builds the word encoder a config names."""

    @staticmethod
    def build(encoder: Encoder, context: ContextConfig, channels: int = CHANNELS) -> WordEncoder:
        """`encoder` for recordings of `channels` rows; a cached encoder reads its features, not them."""
        return BUILDERS[encoder](context, channels)
