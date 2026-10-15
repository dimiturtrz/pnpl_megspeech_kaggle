import argparse
import dataclasses
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.corpus import Corpus
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.metric.ruler import Interval
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Score
from megspeech.evaluation.scoring.telemetry import TelemetryConfig
from megspeech.model.config import ContextConfig
from megspeech.model.context.timing import TimingHead
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.adversary.adversarial import RunTerm
from megspeech.objective.auxiliary.aligned.semantic import SemanticTerm
from megspeech.objective.auxiliary.classified.phonetic import PhoneticTerm
from megspeech.objective.isda import Isda
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import ContextTerm, LabelTerm, WordTerm
from megspeech.objective.topk import TopKMix
from megspeech.training.adversary import Ramp
from megspeech.training.auxiliary import AuxiliaryConfig, Joint, NoAuxiliary
from megspeech.training.config import RunConfig, SelectionConfig, StopConfig
from megspeech.training.hook import Scheduled
from megspeech.training.loop.ema import Ema, Unaveraged
from megspeech.training.loop.trainer import Trainer, TrainState
from megspeech.training.optim import OptimConfig
from megspeech.training.regularise.anchor import Anchor
from megspeech.training.regularise.augment import Augment
from megspeech.training.regularise.pull import Pull
from megspeech.training.regularise.rightcut import RightCut, Whole
from megspeech.training.regularise.sharpness import Adaptive, Flat, Sharp
from megspeech.training.regularise.stretch import Stretch
from megspeech.training.sampling.curriculum import Annealed, Natural, Robust, Schedule
from megspeech.training.sampling.pairing import Pairing, Unpaired
from megspeech.training.sampling.sampler import SamplerConfig
from megspeech.training.timing.expert import TimingExpert
from megspeech.training.transfer import TransferConfig

SAMPLES = 50
CPU = torch.device("cpu")


def split(n: int = 12, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, CHANNELS, SAMPLES)).astype(np.float32)
    y = np.arange(n) % 3
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(n - DENSE) * 2.0]
    return Split.from_arrays(x, y, Spans.by_gap(onset), CPU)


def trainer(config: RunConfig) -> Trainer:
    return Trainer(config, Evaluator(CPU, min_novel=1))


def joint(train: Split) -> Joint:
    """Every token of `train` in the lexicon and the pronunciations but its first."""
    keys = train.token.unique()[1:]
    lexicon = Lexicon(keys, torch.randn(len(keys), DIM, generator=torch.Generator().manual_seed(0)))
    spelled = (torch.arange(len(keys)) % 3).unsqueeze(-1)
    return Joint((SemanticTerm(1.0, lexicon), PhoneticTerm(1.0, Pronunciation(keys, spelled))))


def test_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = Corpus(split(), {"a": split(seed=1)}, frozenset())
    monkeypatch.setattr(Corpus, "load", classmethod(lambda *_: expected))
    assert Trainer.corpus(RunConfig(), cast("DataRoot", tmp_path), CPU) is expected


def test_of(tmp_path: Path) -> None:
    corpus = Corpus(split(), {"a": split(seed=1)}, frozenset({(0, 1)}))
    made = Trainer.of(corpus, RunConfig(selection=SelectionConfig(7)), cast("DataRoot", tmp_path))
    assert made.evaluator.seen == {(0, 1)}
    assert made.evaluator.min_novel == 7
    assert made.evaluator.device == CPU
    assert made.prototypes is None
    assert isinstance(made.auxiliary, NoAuxiliary)
    assert type(made.pairing) is Unpaired
    paired = RunConfig(optim=OptimConfig(batch=4), sampler=SamplerConfig(pairs=0.5))
    masc = Corpus(split(n=24), {"a": split(seed=1)}, frozenset(), positions=heard(24))
    assert isinstance(Trainer.of(masc, paired, cast("DataRoot", tmp_path)).pairing, Pairing)


def test_model() -> None:
    model = trainer(RunConfig(optim=OptimConfig(compiled=True))).model(split())
    assert isinstance(model, SentenceNet)
    assert next(model.parameters()).device == CPU
    assert "_orig_mod" not in next(iter(model.state_dict()))


def test_model_levels() -> None:
    """`context.levels` lands: the subjects of the training split get fitted rows, saved with the model."""
    plain = trainer(RunConfig(optim=OptimConfig(compiled=False))).model(split())
    assert not any(key.startswith("levels.") for key in plain.state_dict())
    config = RunConfig(context=ContextConfig(levels=True), optim=OptimConfig(compiled=False))
    heard = split()
    model = trainer(config).model(dataclasses.replace(heard, subject=torch.arange(len(heard.y)) % 2))
    assert {"levels.gain", "levels.offset"} <= set(model.state_dict())
    assert not torch.equal(model.levels.gain[0], torch.ones_like(model.levels.gain[0]))


def test_model_stem_levels(tmp_path: Path) -> None:
    """Levels refit against the adopted stem's scaler, so each subject leaves it at unit interquartile range."""
    root = DataRoot(tmp_path)
    (root.models() / "pre").mkdir(parents=True)
    torch.save(
        trainer(RunConfig(optim=OptimConfig(compiled=False))).model(split(seed=3)).state_dict(),
        root.models() / "pre" / "model.pt",
    )
    config = RunConfig(
        context=ContextConfig(levels=True), optim=OptimConfig(compiled=False, transfer=TransferConfig(stem="pre"))
    )
    heard = split()
    heard = dataclasses.replace(heard, subject=torch.arange(len(heard.y)) % 2)
    model = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root).model(heard)
    own = heard.subject == 1
    _, spread = Scaler.quartiles(
        model.scale(model.levels(heard.windows(torch.nonzero(own).flatten()), heard.subject[own]))
    )
    assert torch.allclose(spread, torch.ones_like(spread), atol=1e-4)


def test_model_warm(tmp_path: Path) -> None:
    config = RunConfig(optim=OptimConfig(compiled=False, transfer=TransferConfig(warm="pre")))
    trained = trainer(config).model(split())
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(1.0)
    root = DataRoot(tmp_path)
    (root.models() / "pre").mkdir(parents=True)
    torch.save(trained.state_dict(), root.models() / "pre" / "model.pt")
    made = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root)
    warmed = made.model(split(seed=2))
    for key, value in trained.state_dict().items():
        assert torch.equal(warmed.state_dict()[key], value)


def test_model_stem(tmp_path: Path) -> None:
    config = RunConfig(optim=OptimConfig(compiled=False, transfer=TransferConfig(stem="pre")))
    trained = trainer(config).model(split())
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(1.0)
    root = DataRoot(tmp_path)
    (root.models() / "pre").mkdir(parents=True)
    torch.save(trained.state_dict(), root.models() / "pre" / "model.pt")
    made = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root)
    stemmed = made.model(split(seed=2))
    assert all(torch.equal(a, b) for a, b in zip(stemmed.encoder(), trained.encoder(), strict=True))
    assert not any(torch.equal(a, b) for a, b in zip(stemmed.head.parameters(), trained.head.parameters(), strict=True))


def test_compiled() -> None:
    model = nn.Linear(2, 2)
    assert trainer(RunConfig(optim=OptimConfig(compiled=True))).compiled(model) is model


def test_optimizer() -> None:
    made = trainer(RunConfig(optim=OptimConfig(lr=0.01)))
    optimizer = made.optimizer(made.model(split()), Anchor())
    assert optimizer.defaults["lr"] == 0.01
    assert not optimizer.defaults["fused"]


@pytest.mark.parametrize("share", [0.0, 0.1])
def test_optimizer_encoder_share(share: float) -> None:
    made = trainer(RunConfig(optim=OptimConfig(lr=0.01, transfer=TransferConfig(encoder_lr=share))))
    net = made.model(split())
    optimizer = made.optimizer(net, Anchor(), tuple(net.encoder()))
    encoder = {id(p) for p in net.encoder()}
    rates = {group["lr"] for group in optimizer.param_groups for p in group["params"] if id(p) in encoder}
    others = {group["lr"] for group in optimizer.param_groups for p in group["params"] if id(p) not in encoder}
    assert rates == {0.01 * share}
    assert others == {0.01}
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(list(net.parameters()))


def test_optimizer_subject_share() -> None:
    """A surgical adapt moves the subject layer at its own share of the encoder's rate, the rest of the encoder at
    the encoder's."""
    transfer = TransferConfig(encoder_lr=0.1, subject_lr=10.0)
    made = trainer(RunConfig(context=ContextConfig(subjects=2), optim=OptimConfig(lr=0.01, transfer=transfer)))
    net = made.model(split())
    subjects = tuple(net.subjects.parameters())
    optimizer = made.optimizer(net, Anchor(), tuple(net.encoder()), (), subjects)
    chosen = {id(p) for p in subjects}
    rates = {group["lr"] for group in optimizer.param_groups for p in group["params"] if id(p) in chosen}
    assert list(rates) == [pytest.approx(0.01)]


def test_terms() -> None:
    """The context head's objective, the word head's softmax, then the auxiliary terms, sharing the class weights."""
    train = split()
    auxiliary = joint(train)
    config = RunConfig(optim=OptimConfig(balance=0.5, label_smoothing=0.2))
    context, word, *rest = Trainer(config, Evaluator(CPU, min_novel=1), auxiliary=auxiliary).terms(train)
    assert isinstance(context, ContextTerm)
    assert context.objective is config.context.objective
    assert isinstance(word, WordTerm)
    assert word.smoothing == context.smoothing == 0.2
    assert word.classes is context.classes
    assert context.classes is not None
    assert tuple(rest) == auxiliary.terms
    plain = trainer(RunConfig()).terms(train)[1]
    assert isinstance(plain, WordTerm)
    assert plain.classes is None
    expert = trainer(RunConfig(optim=OptimConfig(heads=0.0))).terms(train)
    assert [term.weight for term in expert[:2]] == [0.0, 0.0]
    assert [term.weight for term in trainer(RunConfig()).terms(train)[:2]] == [1.0, 1.0]
    mix = TopKMix(share=0.3, temperature=2.0, margin=0.5)
    mixed = trainer(RunConfig(optim=OptimConfig(topk=0.3, topk_temperature=2.0, topk_margin=0.5))).terms(train)[:2]
    assert [cast("LabelTerm", term).topk for term in mixed] == [mix, mix]


def test_state() -> None:
    """The state trains over `total` steps with the run's anchor, schedule and average as hooks, as `fit` builds
    it."""
    optim = OptimConfig(lr=0.01, anchor=0.5, balance=0.5)
    made = trainer(RunConfig(optim=optim))
    net = made.model(split())
    state = made.state(net, split(), 10)
    assert state.model is net
    assert state.schedule.schedule.state_dict()["total_steps"] == 10
    assert state.hooks == (Anchor.of(0.5, 0.01, 10), Pull(), state.schedule, state.ema, Ramp([], 0.0, 1))
    assert state.readouts == net.readouts()
    assert len(state.terms) == len(state.readouts)
    assert state.rows == made.rows(split())
    assert isinstance(state.curriculum, Natural)
    tilted = trainer(RunConfig(optim=optim, sampler=SamplerConfig(robust=1.0))).state(net, split(), 10)
    assert isinstance(tilted.curriculum, Robust)
    assert type(state.cut) is Whole
    cut = trainer(RunConfig(optim=optim, augment=Augment(right_cut=0.3))).state(net, split(), 10)
    assert isinstance(cut.cut, RightCut)
    assert type(state.sharpness) is Flat
    sharp = trainer(RunConfig(optim=OptimConfig(lr=0.01, sharpness=0.05))).state(net, split(), 10)
    assert sharp.sharpness == Sharp(0.05)
    asam = RunConfig(optim=OptimConfig(lr=0.01, sharpness=0.5, adaptive=True))
    assert trainer(asam).state(net, split(), 10).sharpness == Adaptive(0.5)


def test_rows() -> None:
    train = split()
    natural = trainer(RunConfig(optim=OptimConfig(epochs=3)))
    assert natural.rows(train) == train.capacity(natural.config.optim.batch, train.weight[train.index[:, 0]])
    annealed = trainer(RunConfig(optim=OptimConfig(epochs=3), sampler=SamplerConfig(curriculum=0.9)))
    epochs = range(3)
    draws = (Annealed(0.9, Natural()).weights(Schedule(train, epoch, 3)) for epoch in epochs)
    assert annealed.rows(train) == max(train.capacity(annealed.config.optim.batch, draw) for draw in draws)


def test_step() -> None:
    made = trainer(RunConfig())
    model = made.model(split())
    optimizer = torch.optim.AdamW(model.parameters())
    before = [p.clone() for p in model.parameters()]
    ema = Ema.of(model, 0.0)
    schedule = Scheduled(torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    state = TrainState(
        model, optimizer, schedule, made.rows(split()), made.terms(split()), model.readouts(), ema, (schedule, ema)
    )
    value = made.step(state, split(), torch.tensor([0, 1]))
    assert value.shape == (len(made.terms(split())),)
    assert not value.requires_grad
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def stepped(made: Trainer) -> TrainState:
    model = made.model(split())
    optimizer = torch.optim.AdamW(model.parameters())
    ema = Ema.of(model, 0.0)
    schedule = Scheduled(torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    return TrainState(
        model, optimizer, schedule, made.rows(split()), made.terms(split()), model.readouts(), ema, (schedule, ema)
    )


def test_step_sharp() -> None:
    """A SAM step moves the weights too, its losses those at the unperturbed weights."""
    made = trainer(RunConfig())
    state = dataclasses.replace(stepped(made), sharpness=Sharp(0.05))
    before = [p.clone() for p in state.model.parameters()]
    value = made.step(state, split(), torch.tensor([0, 1]))
    assert torch.isfinite(value).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, state.model.parameters(), strict=True))


def test_losses() -> None:
    made = trainer(RunConfig())
    state = stepped(made)
    state.model.eval()
    batch = split().batch(torch.tensor([0, 1]), state.rows)
    losses, outputs = made.losses(state, batch, split(), batch.inputs())
    assert losses.shape == (len(state.terms),)
    assert len(outputs) == len(state.readouts)
    assert torch.allclose(made.value(state, batch, split(), batch.inputs()), losses.sum())


def test_value() -> None:
    made = trainer(RunConfig())
    state = stepped(made)
    batch = split().batch(torch.tensor([0, 1]), state.rows)
    assert made.value(state, batch, split(), batch.inputs()).requires_grad


def test_stretch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--augment.stretch 0.1` on the command line reaches the stretch every training step resamples its batch with."""
    tree = Tree(RunConfig)
    parser = argparse.ArgumentParser()
    tree.arguments(parser)
    config = tree.parse(parser.parse_args(["--augment.stretch", "0.1"]))
    made = trainer(config)
    assert made.stretch == Stretch(0.1, -config.data.cut().offset)
    assert trainer(RunConfig()).stretch == Stretch()
    seen: list[torch.Tensor] = []
    resampled = Stretch.resampled
    monkeypatch.setattr(
        Stretch, "resampled", lambda stretch, x, factor: seen.append(factor) or resampled(stretch, x, factor)
    )
    model = made.model(split())
    optimizer = torch.optim.AdamW(model.parameters())
    schedule = Scheduled(torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    ema = Ema.of(model, 0.0)
    state = TrainState(
        model, optimizer, schedule, made.rows(split()), made.terms(split()), model.readouts(), ema, (schedule, ema)
    )
    made.step(state, split(), torch.tensor([0, 1]))
    assert len(seen) == 1
    assert seen[0].shape == (state.rows, 1)
    assert ((seen[0] >= 0.9) & (seen[0] <= 1.1)).all()


def test_parameters() -> None:
    """Listed once, in the model's own order."""
    made = trainer(RunConfig())
    state = made.state(made.model(split()), split(), 10)
    assert state.parameters is state.parameters
    assert [id(p) for p in state.parameters] == [id(p) for p in state.model.parameters()]
    assert type(state.ema) is Unaveraged


def test_drained() -> None:
    """Every epoch counter, named as the epoch's log spells it; a plain state tallied nothing."""
    made = trainer(RunConfig())
    state = made.state(made.model(split()), split(), 10)
    assert state.drained() == {
        "grad norms": {},
        "bgmix mixed words": 0,
        "averaged words": 0,
        "right cut": {},
        "isda lambda": 0.0,
    }
    runs = RunTerm.of(1.0, pd.DataFrame({"run": ["a", "b"]}), CPU)
    runs.hits.add_(torch.tensor([1.0, 4.0]))
    tallied = dataclasses.replace(state, terms=(*state.terms, runs)).drained()
    assert tallied["RunTerm"] == {"accuracy": 0.25, "majority": 0.5, "classes": 2.0}


def test_cloned() -> None:
    weights = {"w": torch.ones(2)}
    copied = Trainer.cloned(weights)
    weights["w"].add_(1.0)
    assert torch.equal(copied["w"], torch.ones(2))


def test_epoch() -> None:
    """One loss per term, averaged over the `steps` it trains, and the model moved."""
    made = trainer(RunConfig(optim=OptimConfig(batch=4)))
    train = split()
    state = made.state(made.model(train), train, 10)
    before = [p.clone() for p in state.model.parameters()]
    terms = made.epoch(state, Schedule(train, 0, 1), 2)
    assert terms.shape == (len(state.terms),)
    assert torch.isfinite(terms).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, state.model.parameters(), strict=True))


def test_epoch_isda() -> None:
    """The word head's augmentation trains each epoch at its ramped strength over the features its linear read."""
    made = trainer(RunConfig(optim=OptimConfig(batch=4, isda=0.5, epochs=4)))
    train = split()
    net = made.model(train)
    state = made.state(net, train, 10)
    _, word = net.readouts()
    assert isinstance(state.isda, Isda)
    assert state.isda.linear is cast("nn.Sequential", word)[-1]
    assert cast("WordTerm", state.terms[1]).isda is state.isda
    terms = made.epoch(state, Schedule(train, 2, 4), 2)
    assert torch.isfinite(terms).all()
    assert state.drained()["isda lambda"] == pytest.approx(0.25)
    assert state.isda.count.sum() > 0


def test_fit() -> None:
    model, history = trainer(RunConfig(augment=Augment(0.5, 0.1), optim=OptimConfig(epochs=1, batch=4, clip=1.0))).fit(
        split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE)
    )
    assert len(history) == 1
    assert isinstance(model, SentenceNet)
    train = split()
    made = Trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4)), Evaluator(CPU, min_novel=1), auxiliary=joint(train))
    net, _ = made.fit(train, {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))
    assert isinstance(net, SentenceNet)
    assert made.heads
    assert all(key.startswith("heads.") for key in made.heads)
    anchored = trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, anchor=0.5)))
    assert isinstance(anchored.fit(split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))[0], SentenceNet)
    balanced = trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, balance=0.5)))
    assert isinstance(balanced.fit(split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))[0], SentenceNet)
    topk = trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, balance=0.5, topk=0.5)))
    assert isinstance(topk.fit(split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))[0], SentenceNet)
    with pytest.raises(FloatingPointError):
        trainer(RunConfig(optim=OptimConfig(epochs=1, batch=4, lr=math.inf))).fit(
            split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE)
        )


def test_fit_telemetry_equivalence() -> None:
    """Telemetry only reads: on or off, the same seed trains the same weights and scores the same epochs."""

    def steering_free(row: Epoch) -> tuple[int, float, object, object]:
        return row.epoch, row.loss, row.macro, row.interval

    def run(config: TelemetryConfig) -> tuple[SentenceNet, list[Epoch]]:
        torch.manual_seed(0)
        net, history = trainer(RunConfig(optim=OptimConfig(epochs=2, batch=4), telemetry=config)).fit(
            split(n=24), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE)
        )
        return cast("SentenceNet", net), history

    on, seen = run(TelemetryConfig())
    off, plain = run(TelemetryConfig(fit=False, neural=False, probs=False))
    assert all(row.fit is not None and row.shuffled is not None for row in seen)
    assert all(row.fit is None and row.shuffled is None for row in plain)
    assert [(r.epoch, r.loss, r.macro, r.interval) for r in seen] == [
        (r.epoch, r.loss, r.macro, r.interval) for r in plain
    ]
    assert all(torch.equal(a, b) for a, b in zip(on.state_dict().values(), off.state_dict().values(), strict=True))


def heard(n: int) -> pd.DataFrame:
    """`n` words: the first half subject 101's hearing of a MASC story, the second subject 102's of the same, in
    three-word sentences."""
    half = n // 2
    runs = ["sub-101_ses-0_task-MASC1_run-1"] * half + ["sub-102_ses-0_task-MASC1_run-1"] * (n - half)
    position = np.r_[np.arange(half), np.arange(n - half)]
    return pd.DataFrame({"run": runs, "position": position, "sentence": position // 3})


def test_fit_instance_off_is_bit_identical(tmp_path: Path) -> None:
    """The instance flags at 0 train the same weights from the same seed as a trainer that never heard of them."""
    root, optim = cast("DataRoot", tmp_path), OptimConfig(epochs=2, batch=4)

    def run(made: Trainer) -> tuple[list[float], dict[str, torch.Tensor]]:
        net, history = made.fit(split(n=24), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))
        return [row.loss for row in history], net.state_dict()

    corpus = Corpus(split(n=24), {"a": split(seed=1)}, frozenset(), positions=heard(24))
    off = RunConfig(
        optim=OptimConfig(epochs=2, batch=4, auxiliaries=AuxiliaryConfig(instance=0.0)),
        sampler=SamplerConfig(pairs=0.0),
    )
    explicit = Trainer.of(corpus, off, root)
    assert isinstance(explicit.auxiliary, NoAuxiliary)
    assert type(explicit.pairing) is Unpaired
    (losses, weights), (plain, reference) = run(explicit), run(trainer(RunConfig(optim=optim)))
    assert losses == plain
    assert all(torch.equal(weights[key], value) for key, value in reference.items())
    on = RunConfig(
        optim=OptimConfig(epochs=2, batch=4, auxiliaries=AuxiliaryConfig(instance=0.25)),
        sampler=SamplerConfig(pairs=0.5),
    )
    paired, _ = run(Trainer.of(corpus, on, root))
    assert all(math.isfinite(loss) for loss in paired)
    assert paired != plain
    apart = RunConfig(
        optim=OptimConfig(epochs=2, batch=4, auxiliaries=AuxiliaryConfig(instance=0.25, instance_apart=True)),
        sampler=SamplerConfig(pairs=0.75, hearers=3),
    )
    chorus, _ = run(Trainer.of(corpus, apart, root))
    assert all(math.isfinite(loss) for loss in chorus)
    assert chorus != paired


def scripted(monkeypatch: pytest.MonkeyPatch, bacc10: list[float]) -> None:
    def report(_: Evaluator, *args: object) -> Epoch:
        epoch, loss = cast("int", args[-2]), cast("float", args[-1])
        value = bacc10[epoch]
        return Epoch(epoch, loss, {}, Score(0.5, value, {0.5: value}, 1), Interval(value, value), 0.0, 0.0)

    monkeypatch.setattr(Evaluator, "report", report)


@pytest.mark.parametrize(("patience", "trained"), [(0, 6), (2, 4), (5, 6), (1, 3)])
def test_fit_patience(monkeypatch: pytest.MonkeyPatch, patience: int, trained: int) -> None:
    scripted(monkeypatch, [0.2, 0.5, 0.3, 0.3, 0.9, 0.9])
    config = RunConfig(optim=OptimConfig(epochs=6, batch=4), stop=StopConfig(patience))
    _, history = trainer(config).fit(split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE))
    assert len(history) == trained
    assert config.selected(history).epoch == (1 if trained < 5 else 4)


def test_fit_restores_the_selected_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The returned weights are the selected epoch's as validated, not an alias of the live ones the later epochs
    went on to train."""
    scripted(monkeypatch, [0.2, 0.9, 0.3])
    scored = Evaluator.report
    seen: list[dict[str, torch.Tensor]] = []

    def report(
        evaluator: Evaluator,
        model: nn.Module,
        domains: dict[str, Split],
        log_prior: torch.Tensor,
        epoch: int,
        loss: float,
    ) -> Epoch:
        seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return scored(evaluator, model, domains, log_prior, epoch, loss)

    monkeypatch.setattr(Evaluator, "report", report)
    net, _ = trainer(RunConfig(optim=OptimConfig(epochs=3, batch=4))).fit(
        split(), {"a": split(seed=1)}, torch.zeros(VOCAB_SIZE)
    )
    kept = net.state_dict()
    assert all(torch.equal(kept[k], v) for k, v in seen[1].items())
    assert not all(torch.equal(kept[k], v) for k, v in seen[2].items())


def test_auxiliary_state() -> None:
    model = nn.ModuleDict({"heads": nn.ModuleList([nn.Linear(2, 2)]), "words": nn.Linear(2, 2)})
    assert set(Trainer.auxiliary_state(model)) == {"heads.0.weight", "heads.0.bias"}


def test_model_timing(tmp_path: Path) -> None:
    """A run over a timing expert holds the expert's head frozen at its weights."""
    root = DataRoot(tmp_path)
    expert = TimingHead(VOCAB_SIZE)
    TimingExpert.save(expert, root.models() / "t")
    config = RunConfig(context=ContextConfig(rhythm=True), optim=OptimConfig(compiled=False, timing="t"))
    made = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root)
    model = made.model(split())
    assert all(torch.equal(model.timing.state_dict()[k], v) for k, v in expert.state_dict().items())
    assert not any(parameter.requires_grad for parameter in model.timing.parameters())


@pytest.mark.parametrize("shares", [(0.0, 0.5), ()])
def test_optimizer_fused_shares(shares: tuple[float, ...]) -> None:
    context = ContextConfig(fused=("a", "b"))
    made = trainer(RunConfig(context=context, optim=OptimConfig(lr=0.01, fused_lr=shares)))
    net = made.model(split())
    params = list(net.parameters())
    first, second = params[:2], params[2:4]
    optimizer = made.optimizer(net, Anchor(), (), (first, second))
    rate = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    expected = shares or (1.0, 1.0)
    assert {rate[id(p)] for p in first} == {0.01 * expected[0]}
    assert {rate[id(p)] for p in second} == {0.01 * expected[1]}


def test_model_fused(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    saved = trainer(RunConfig(optim=OptimConfig(compiled=False))).model(split())
    (root.models() / "a").mkdir(parents=True)
    torch.save(saved.state_dict(), root.models() / "a" / "model.pt")
    config = RunConfig(context=ContextConfig(fused=("a",)), optim=OptimConfig(compiled=False, fused_lr=(0.0,)))
    made = Trainer.of(Corpus(split(), {"a": split(seed=1)}, frozenset()), config, root)
    net = made.model(split(seed=2))
    assert net.fusion.names() == ("primary", "a")
    assert all(torch.equal(a, b) for a, b in zip(net.fusion.encoders()[0], saved.encoder(), strict=True))
    optimizer = made.optimizer(net, Anchor(), tuple(net.encoder()), net.fusion.encoders())
    frozen = {id(p) for p in net.fusion.encoders()[0]}
    assert {g["lr"] for g in optimizer.param_groups for p in g["params"] if id(p) in frozen} == {0.0}


def test_run_config_fused_rates() -> None:
    with pytest.raises(ValueError, match="fused_lr"):
        RunConfig(context=ContextConfig(fused=("a", "b")), optim=OptimConfig(fused_lr=(0.0,)))


def test_step_frozen_encoder() -> None:
    """At an `encoder_lr` of 0 a step leaves the encoder's weights and batch statistics bit-identical, even under the
    anchor's subject pull and a `train()` call, while the heads train."""
    made = trainer(RunConfig(optim=OptimConfig(lr=0.01, anchor=0.5, transfer=TransferConfig(encoder_lr=0.0))))
    net = made.model(split())
    state = made.state(net, split(), 10)
    state.model.train()
    stage = {k: v.clone() for module in net.stage() for k, v in module.state_dict(keep_vars=False).items()}
    buffers = [b for module in net.stage() for b in module.buffers() if b.is_floating_point()]
    head = [p.clone() for p in net.head.parameters()]
    made.step(state, split(), torch.tensor([0, 1]))
    after = {k: v for module in net.stage() for k, v in module.state_dict().items()}
    assert buffers
    assert all(torch.equal(stage[k], after[k]) for k in stage)
    assert not any(module.training for module in net.stage())
    assert all(p.grad is None for p in net.encoder())
    assert any(not torch.equal(a, b) for a, b in zip(head, net.head.parameters(), strict=True))
