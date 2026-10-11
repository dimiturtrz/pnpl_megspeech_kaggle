from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch import nn

from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.model.config import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)), run_centred=True)
CPU = torch.device("cpu")
WORDS = 12
SAMPLES = 50
OFFSET = 3.0


def split() -> Split:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(WORDS, CHANNELS, SAMPLES)).astype(np.float32)
    x[WORDS // 2 :] += OFFSET
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(WORDS - DENSE) * 2.0]
    held = Split.from_arrays(x, np.arange(WORDS) % 3, Spans.by_gap(onset), CPU)
    return held.numbered(pd.Series(["a"] * (WORDS // 2) + ["b"] * (WORDS - WORDS // 2)))


def test_evaluation_centres_on_each_runs_own_mean() -> None:
    """A split's batches through a run-centred model, as evaluation reads them: the two-pass read equals one pass
    with every run's mean over all its windows subtracted, and differs from the uncentred model's."""
    held = split()
    torch.manual_seed(0)
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    evaluator = Evaluator(CPU)
    got = evaluator.sweep(model, held)
    assert model.centring.fixed is None
    with torch.no_grad():
        embedded = model.embed(held.windows(torch.arange(WORDS)), held.subject)
    assert held.run is not None
    model.centring.fixed = torch.stack([embedded[held.run == run].mean(0) for run in (0, 1)])
    explicit = evaluator.swept(model, held)
    model.centring.release()
    for head in Head:
        assert got[head].shape == (WORDS, VOCAB_SIZE)
        assert torch.allclose(got[head], explicit[head], atol=1e-4)
    torch.manual_seed(0)
    plain = evaluator.sweep(SentenceNet(nn.Identity(), replace(CONTEXT, run_centred=False)).eval(), held)
    assert not torch.allclose(got[Head.WORD], plain[Head.WORD], atol=1e-4)


def test_training_step_reads_moving_means() -> None:
    """A training batch's forward fills each present run's moving mean and backpropagates."""
    held = split()
    model = SentenceNet(nn.Identity(), CONTEXT).train()
    model.centring.reserve(held.runs())
    context, word = model(held.batch(torch.arange(len(held.index)), WORDS).inputs())
    (context.sum() + word.sum()).backward()
    assert bool(model.centring.seen.all())
    assert torch.isfinite(context).all()
