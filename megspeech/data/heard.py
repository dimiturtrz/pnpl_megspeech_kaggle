"""The words training heard: the word table cut to a grouping, and the text and label prior counted from it."""

from collections.abc import Sequence
from itertools import chain

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import SpanText, Split
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping

Text = tuple[frozenset[tuple[int, ...]], Float[Tensor, " v"]]
"""The token sequence of every span training heard, and the label prior of its draws."""


class Heard:
    @classmethod
    def of(cls, cache: RunCache, runs: Sequence[Run], data: DataConfig) -> pd.DataFrame:
        """Every word of the runs training reads (`DataConfig.trained`) that `data.grouping` carries, with its run's
        sampling `weight`, read from the word table without any MEG: the text a novel span must not repeat, and the
        labels the training prior counts."""
        trained = data.trained(runs)
        table = cache.table([r.name for r in trained])
        weights = {r.name: data.weight(r) for r in trained}
        return data.carried(table.assign(weight=table["run"].map(weights)))

    @classmethod
    def under(cls, root: DataRoot, data: DataConfig, device: torch.device) -> Text:
        """The text training under `data` heard, as every span's token sequence, and the label prior of its draws."""
        heard = cls.of(data.run_cache(root), Run.everywhere(root), data)
        return cls.sequences(heard, data.grouping), cls.prior(heard, data).to(device)

    @staticmethod
    def prior(heard: pd.DataFrame, data: DataConfig) -> Float[Tensor, " v"]:
        """The label prior of training's draws (`Split.log_prior` of the train split) counted from `heard` without
        any MEG: each word at its run's weight, a recited span's words down-weighted by `data.repeat_power`."""
        repeats = SpanText(data.grouping.spans(heard).index, heard["token"].to_numpy()).repeats(data.repeat_power)
        weight = torch.tensor(heard["weight"].to_numpy(dtype=np.float64), dtype=torch.float32)
        return Split.prior(torch.tensor(heard["target"].to_numpy()), weight * torch.from_numpy(repeats).float())

    @staticmethod
    def sequences(heard: pd.DataFrame, grouping: Grouping) -> frozenset[tuple[int, ...]]:
        """The text of every span of `heard` as its token sequence, spans cut run by run."""
        runs = heard.groupby("run", sort=False)
        texts = (SpanText(grouping.spans(run).index, run["token"].to_numpy()).sequences() for _, run in runs)
        return frozenset(chain.from_iterable(texts))
