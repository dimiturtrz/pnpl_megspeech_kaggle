from pathlib import Path
from typing import cast

import pandas as pd
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import SpanInput
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.objective.auxiliary.contrastive.word import SameWord, SameWordTerm
from megspeech.objective.targets.loudness import CUT
from megspeech.training.auxiliary import Auxiliary, AuxiliaryConfig, Joint, Tables
from tests.unit.megspeech.model.auxiliary import ONSET, SLOT, net, x

CPU = torch.device("cpu")
WORDS = pd.DataFrame({"run": ["a"] * 3, "position": range(3), "sentence": [0, 0, 1]})
LABEL = torch.tensor([5, 5, 5])


def identity(model: WithAuxiliary) -> WithAuxiliary:
    return model


def test_builder_term_trains_a_tiny_model() -> None:
    """The auxiliary builder's same-word term, tapped on the context output, takes a finite step through a sentence
    model whose three words share a label across two sentences."""
    root = cast("DataRoot", Path())
    built = Auxiliary.of(AuxiliaryConfig(contrast=0.5, tap=Tap.CONTEXT), 0.0, Tables(root, CPU, WORDS, CUT))
    term = built.terms[0]
    assert isinstance(term, SameWordTerm)
    assert term.weight == 0.5
    model = Joint(built.terms, Tap.CONTEXT).trained(net(), identity)
    assert model.taps == (Tap.CONTEXT,)
    outputs = model(SpanInput(x(), ONSET, SLOT, torch.zeros(3, dtype=torch.long)))
    target = torch.where(SLOT < 3, SameWord.pack(LABEL, term.sentence)[SLOT.clamp(max=2)], -100)
    loss = term.loss(term.read(model.heads[0], outputs[2], target))
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
