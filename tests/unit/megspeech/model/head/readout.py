from dataclasses import replace
from typing import cast

import torch
from torch import nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import ContextConfig
from megspeech.model.head.hierarchy import Factorised, Flat
from megspeech.model.head.readout import ContextReadout
from megspeech.objective.anchored import AnchoredHead
from megspeech.objective.anchors import Source
from megspeech.objective.context import Objective
from megspeech.objective.scorer import DIM
from tests.unit.megspeech.model.sentence import CONTEXT, GENERATOR


def test_hierarchy() -> None:
    assert isinstance(ContextReadout.hierarchy(ContextConfig()), Flat)
    assert isinstance(ContextReadout.hierarchy(ContextConfig(hierarchical=True)), Factorised)


def test_build() -> None:
    anchors = torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR)
    torch.manual_seed(0)
    plain = cast("nn.Linear", ContextReadout.build(CONTEXT, None))
    torch.manual_seed(0)
    assert torch.equal(plain.weight, cast("nn.Linear", Objective.SOFTMAX.head(CONTEXT.width, None)).weight)
    anchored = replace(CONTEXT, anchor=Source.AUDIO)
    head = cast("AnchoredHead", ContextReadout.build(anchored, anchors))
    assert torch.equal(head.anchors, anchors)
    trained = cast("AnchoredHead", ContextReadout.build(replace(anchored, anchor_train=True), anchors))
    assert isinstance(trained.anchors, nn.Parameter)
