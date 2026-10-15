"""Which training windows a sensor whitening is fit on: the windows themselves, or their residuals about each word's
mean window, whose covariance is the noise's."""

import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.model.config import Covariance
from megspeech.training.regularise.inputs.bgmix import MIN_CELL, BGMix, BGMixKey

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TotalSample:
    """The sample as it is: their covariance holds the evoked signal as well as the noise."""

    def windows(self, train: Split, sample: Float[Tensor, "k c t"]) -> Float[Tensor, "k c t"]:
        return sample


@dataclass(frozen=True)
class NoiseSample:
    """Labelled windows less their word's training template, the grand mean for a word under `MIN_CELL` windows."""

    def windows(self, train: Split, sample: Float[Tensor, "k c t"]) -> Float[Tensor, "k c t"]:
        """As many residuals as `sample` has windows."""
        table, cells = BGMix.templates(train, BGMixKey.POOLED)
        template = table[0]
        grand = (template * cells[:, None, None]).sum(0) / cells.sum()
        words = train.labelled()
        words = words[torch.randperm(len(words), device=words.device)[: len(sample)]]
        mean = torch.where((cells[train.y[words]] < MIN_CELL)[:, None, None], grand, template[train.y[words]])
        LOG.info("noise whitening: %d residuals about %d word templates", len(words), int((cells >= MIN_CELL).sum()))
        return train.windows(words).float() - mean


SAMPLES: dict[Covariance, TotalSample | NoiseSample] = {
    Covariance.TOTAL: TotalSample(),
    Covariance.NOISE: NoiseSample(),
}
