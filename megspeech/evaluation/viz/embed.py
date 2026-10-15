"""What a model's context head reads for each word, and its 2-D principal-component projection."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.layers.centring import RunCentring

COMPONENTS = 2


@dataclass(frozen=True)
class Embedder:
    """Reads every word's refined span embedding, the context head's input, members' side by side."""

    evaluator: Evaluator

    @staticmethod
    @contextmanager
    def captured(model: nn.Module, into: list[Tensor]) -> Iterator[None]:
        """Within the block, every context head of `model` appends its input to `into`."""
        heads = [member.readouts()[0] for member in model.modules() if isinstance(member, SentenceNet)]
        handles = [head.register_forward_pre_hook(lambda _, args: into.append(args[0])) for head in heads]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def embedded(self, model: nn.Module, split: Split) -> Float[Tensor, "n h"]:
        """Every word of `split`'s embedding, in word order, read as `Evaluator.sweep` reads its logits."""
        model.eval()
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        out: Tensor | None = None
        with RunCentring.calibrated(model, lambda: self.evaluator.swept(model, split)), torch.no_grad():
            for spans in torch.arange(len(split.index), device=self.evaluator.device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                taken: list[Tensor] = []
                with self.captured(model, taken):
                    model(batch.inputs())
                refined = torch.cat(taken, dim=-1).float()
                if out is None:
                    out = torch.zeros(words + 1, refined.shape[-1], device=refined.device)
                out[batch.words_at(words)] = refined
        return torch.zeros(words, 0) if out is None else out[:words]


@dataclass(frozen=True)
class Projection:
    """Points on their first `COMPONENTS` principal components, with the variance share each explains."""

    points: Float[np.ndarray, "n 2"]
    explained: Float[np.ndarray, " 2"]

    @classmethod
    def of(cls, x: Float[np.ndarray, "n h"]) -> "Projection":
        centred = x - x.mean(axis=0, keepdims=True)
        _, values, vectors = np.linalg.svd(centred, full_matrices=False)
        variance = values**2
        share = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
        points = np.zeros((len(x), COMPONENTS))
        explained = np.zeros(COMPONENTS)
        kept = min(COMPONENTS, len(values))
        points[:, :kept] = centred @ vectors[:kept].T
        explained[:kept] = share[:kept]
        return cls(points, explained)
