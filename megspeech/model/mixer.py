"""The layers that refine word embeddings across a span."""

from torch import nn

from megspeech.model.config import ContextConfig
from megspeech.model.context.droplayer import DropPathLayer
from megspeech.model.context.droppath import DropPath
from megspeech.model.context.span import SpanConformer


class Mixer:
    @staticmethod
    def of(context: ContextConfig) -> nn.Module:
        """The layers that refine word embeddings across the span: a transformer, conformer when `mixing`; with no
        `layers` none, each word read alone."""
        if context.mixing or not context.layers:
            conformer = SpanConformer(context.width, context.heads, context.layers, context.dropout, context.mixing)
            conformer.ramp(context.drop_path)
            return conformer
        layer = SpanConformer.layer(context.width, context.heads, context.dropout)
        norm = nn.LayerNorm(context.width) if context.final_norm else None
        encoder = nn.TransformerEncoder(layer, context.layers, norm=norm, enable_nested_tensor=False)
        DropPath.ramp((block.path for block in encoder.layers if isinstance(block, DropPathLayer)), context.drop_path)
        return encoder
