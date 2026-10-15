"""What a model's context head reads for each word, and its 2-D projections (PCA, UMAP)."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from umap import UMAP

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
    """Points in 2-D, with the variance share each axis explains where the projector defines one (else empty)."""

    points: Float[np.ndarray, "n 2"]
    explained: Float[np.ndarray, " k"]

    @classmethod
    def flat(cls, rows: int) -> "Projection":
        """Every point at the origin: what a projector returns for input it cannot spread."""
        return cls(np.zeros((rows, COMPONENTS)), np.zeros(0))


class Projector(Protocol):
    """Maps embeddings to a 2-D `Projection`, under its own `name` in the export."""

    @property
    def name(self) -> str: ...

    def project(self, x: Float[np.ndarray, "n h"]) -> Projection: ...


@dataclass(frozen=True)
class Principal:
    """The first `COMPONENTS` principal components and the variance share each explains."""

    name: str = "pca"

    def project(self, x: Float[np.ndarray, "n h"]) -> Projection:
        centred = x - x.mean(axis=0, keepdims=True)
        _, values, vectors = np.linalg.svd(centred, full_matrices=False)
        variance = values**2
        share = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
        points = np.zeros((len(x), COMPONENTS))
        explained = np.zeros(COMPONENTS)
        kept = min(COMPONENTS, len(values))
        points[:, :kept] = centred @ vectors[:kept].T
        explained[:kept] = share[:kept]
        return Projection(points, explained)


@dataclass(frozen=True)
class Manifold:
    """UMAP on cosine distance: keeps each word's `neighbours` nearest, so clusters linear axes average away show."""

    name: str = "umap"
    neighbours: int = 15
    spread: float = 0.1
    seed: int = 0

    def project(self, x: Float[np.ndarray, "n h"]) -> Projection:
        if x.shape[1] == 0 or len(x) <= self.neighbours or not np.ptp(x, axis=0).any():
            return Projection.flat(len(x))
        reducer = UMAP(
            n_components=COMPONENTS,
            metric="cosine",
            n_neighbors=self.neighbours,
            min_dist=self.spread,
            random_state=self.seed,
        )
        return Projection(np.asarray(reducer.fit_transform(x), dtype=np.float64), np.zeros(0))


PROJECTORS: tuple[Projector, ...] = (Principal(), Manifold())
