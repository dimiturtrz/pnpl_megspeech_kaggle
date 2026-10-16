import argparse

from core.tree import Tree
from megspeech.data.batching.split import EVAL_SPANS
from megspeech.training.adversary import AdversaryConfig
from megspeech.training.auxiliary import AuxiliaryConfig
from megspeech.training.optim import OptimConfig
from megspeech.training.teacher import TeacherConfig
from megspeech.training.transfer import TransferConfig

GROUPED = OptimConfig(
    seed=7,
    auxiliaries=AuxiliaryConfig(semantic=0.5),
    adversarial=AdversaryConfig(adversary=0.3),
    distillation=TeacherConfig(distil=0.2),
    transfer=TransferConfig(encoder_lr=0.1),
)


def test_flat_record_loads_into_its_groups() -> None:
    flat = {"seed": 7, "semantic": 0.5, "adversary": 0.3, "distil": 0.2, "encoder_lr": 0.1}
    assert Tree(OptimConfig).load(flat) == GROUPED


def test_flat_and_grouped_flags_agree() -> None:
    parser = argparse.ArgumentParser()
    Tree(OptimConfig).arguments(parser)
    flat = ["--seed", "7", "--semantic", "0.5", "--adversary", "0.3", "--distil", "0.2", "--encoder_lr", "0.1"]
    grouped = ["--seed", "7", "--auxiliaries.semantic", "0.5", "--adversarial.adversary", "0.3"]
    grouped += ["--distillation.distil", "0.2", "--transfer.encoder_lr", "0.1"]
    assert Tree(OptimConfig).parse(parser.parse_args(flat)) == GROUPED
    assert Tree(OptimConfig).parse(parser.parse_args(grouped)) == GROUPED


def test_eval_spans() -> None:
    assert OptimConfig().eval_spans() == EVAL_SPANS
    assert OptimConfig(batch=8).eval_spans() == EVAL_SPANS // 8
    assert OptimConfig(batch=1).eval_spans() >= 1
