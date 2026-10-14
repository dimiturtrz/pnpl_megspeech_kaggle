import numpy as np
import pandas as pd
import torch

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.contrastive.instance import InstanceTerm
from megspeech.training.sampling.pairing import Pairing
from megspeech.training.sampling.sampler import SamplerConfig

CPU = torch.device("cpu")
SUBJECTS, TOKENS, BATCH, HEARERS = 4, 6, 12, 3


def words() -> pd.DataFrame:
    """`SUBJECTS` subjects each hearing the same `TOKENS` one-word sentences of a MASC story, one span a word."""
    runs = [f"sub-{101 + s}_ses-0_task-MASC1_run-1" for s in range(SUBJECTS) for _ in range(TOKENS)]
    position = np.tile(np.arange(TOKENS), SUBJECTS)
    return pd.DataFrame({"run": runs, "position": position, "sentence": position})


def test_paired_batches_give_every_anchor_its_other_hearers() -> None:
    """The pair sampler's batches, read through the instance term, give each anchor `HEARERS - 1` positives from
    other subjects: the sampler's output is the term's valid input."""
    table, n = words(), SUBJECTS * TOKENS
    train = Split.from_arrays(np.zeros((n, 4, 4), np.float32), np.arange(n) % 3, Spans.by_gap(np.arange(n) * 9.0), CPU)
    pairing = Pairing.of(SamplerConfig(pairs=1.0, hearers=HEARERS), table, train, BATCH)
    assert isinstance(pairing, Pairing)
    term = InstanceTerm.of(1.0, table, CPU, apart=True)
    torch.manual_seed(0)
    batches = pairing.drawn(torch.zeros(5 * BATCH, dtype=torch.long), torch.ones(n), BATCH).view(5, BATCH)
    for spans in batches:
        _, positive = term.pairs(train.index[spans, 0])
        assert (positive[: pairing.anchors].sum(dim=1) >= HEARERS - 1).all()
