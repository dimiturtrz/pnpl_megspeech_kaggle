from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.config import Cut
from megspeech.data.labels.context_tags import CLASSES, ContextTags
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.model.config import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.heads import WordHead
from megspeech.objective.auxiliary.acoustic import Acoustic, AcousticTerm
from megspeech.objective.auxiliary.adversarial import AdversaryTarget, RunTerm, TimingTerm
from megspeech.objective.auxiliary.audio import AudioTerm, Listening
from megspeech.objective.auxiliary.contrastive.instance import Apart, InstanceTerm, Together
from megspeech.objective.auxiliary.contrastive.word import SameWordTerm
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.auxiliary.neighbour import NextWordTerm, PreviousWordTerm
from megspeech.objective.auxiliary.phonetic import PhoneticTerm
from megspeech.objective.auxiliary.semantic import SemanticTerm
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.auxiliary.syntactic import SyntacticTerm
from megspeech.objective.auxiliary.tagging import TagTerm
from megspeech.objective.auxiliary.window import WindowTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.loudness import CUT, FRAMES, LENGTH, Loudness
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.objective.targets.t5 import Lexicon
from megspeech.training.adversary import AdversaryConfig
from megspeech.training.auxiliary import Auxiliary, AuxiliaryConfig, Joint, NoAuxiliary, Tables

CPU = torch.device("cpu")
GENERATOR = torch.Generator().manual_seed(0)
WORDS = pd.DataFrame({"run": ["a", "b"], "position": [1, 0]})


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def processed(self) -> Path:
        return self.base


def net() -> SentenceNet:
    return SentenceNet(
        nn.Identity(), ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
    )


def semantic() -> SemanticTerm:
    return SemanticTerm(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def phonetic() -> PhoneticTerm:
    return PhoneticTerm(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([[3, 1], [7, -1]])))


def identity(model: WithAuxiliary) -> WithAuxiliary:
    return model


def test_no_auxiliary_trained() -> None:
    model = net()
    assert NoAuxiliary().trained(model, identity) is model
    assert NoAuxiliary().terms == ()


def test_joint_trained() -> None:
    model = net()
    trained = Joint((semantic(), phonetic())).trained(model, identity)
    assert isinstance(trained, WithAuxiliary)
    assert trained.net is model
    assert len(trained.heads) == 2
    tapped = Joint((semantic(),), Tap.CONTEXT).trained(model, identity)
    assert tapped.tap is Tap.CONTEXT
    assert tapped.heads[0](torch.zeros(1, model.width)).shape == (1, DIM)


def test_joint_neighbour_taps_the_word() -> None:
    model = net()
    trained = Joint((NextWordTerm(weight=1.0), semantic()), Tap.CONTEXT).trained(model, identity)
    assert trained.taps == (Tap.WORD, Tap.CONTEXT)
    assert trained.heads[0](torch.zeros(1, model.words.width)).shape == (1, VOCAB_SIZE)


def test_of(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Auxiliary.of(AuxiliaryConfig(), AdversaryConfig(), Tables(root, CPU, WORDS, CUT)), NoAuxiliary)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(semantic=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(phonetic=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    semantic().lexicon.save(Lexicon.path(root))
    phonetic().pronunciation.save(Pronunciation.path(root))
    made = Auxiliary.of(AuxiliaryConfig(semantic=0.5), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert isinstance(made, Joint)
    assert [type(t) for t in made.terms] == [SemanticTerm]
    assert made.tap is Tap.WORD
    assert (
        cast(
            "Joint",
            Auxiliary.of(
                AuxiliaryConfig(semantic=0.5, tap=Tap.CONTEXT), AdversaryConfig(), Tables(root, CPU, WORDS, CUT)
            ),
        ).tap
        is Tap.CONTEXT
    )
    both = Auxiliary.of(AuxiliaryConfig(semantic=0.5, phonetic=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [type(t) for t in both.terms] == [SemanticTerm, PhoneticTerm]
    assert both.terms[1].weight == 1.0
    worded = Auxiliary.of(AuxiliaryConfig(semantic=0.5, lexical=2.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [type(t) for t in worded.terms] == [SemanticTerm, LexicalTerm]
    neighbours = Auxiliary.of(
        AuxiliaryConfig(next_word=0.3, prev_word=0.2), AdversaryConfig(), Tables(root, CPU, WORDS, CUT)
    )
    assert [(type(t), t.weight) for t in neighbours.terms] == [(NextWordTerm, 0.3), (PreviousWordTerm, 0.2)]
    windowed = Auxiliary.of(AuxiliaryConfig(semantic=0.5, window=0.3), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [type(t) for t in windowed.terms] == [SemanticTerm, WindowTerm]
    assert windowed.terms[1].weight == 0.3
    spelled = Auxiliary.of(AuxiliaryConfig(phonetic=1.0, positions=2), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert cast("PhoneticTerm", spelled.terms[0]).positions == 2
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(syntactic=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    PartsOfSpeech(torch.tensor([10, 20]), torch.tensor([7, 15])).save(PartsOfSpeech.path(root))
    tagged = Auxiliary.of(
        AuxiliaryConfig(phonetic=1.0, syntactic=0.3), AdversaryConfig(), Tables(root, CPU, WORDS, CUT)
    )
    assert [type(t) for t in tagged.terms] == [PhoneticTerm, SyntacticTerm]
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(tagging=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    ContextTags(np.array(["a", "b"]), np.array([0.5, 1.5]), np.array([3, 5])).save(ContextTags.path(root))
    tags = Auxiliary.of(
        AuxiliaryConfig(tagging=0.5), AdversaryConfig(), Tables(root, CPU, WORDS.assign(onset=[0.5, 9.0]), CUT)
    )
    assert cast("TagTerm", tags.terms[0]).tags.tolist() == [3, -1]
    timed = Auxiliary.of(
        AuxiliaryConfig(),
        AdversaryConfig(adversary=1.0),
        Tables(root, CPU, WORDS.assign(sentence=0, onset=[0.0, 0.2]), CUT),
    )
    assert [type(t) for t in timed.terms] == [TimingTerm]
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(speech=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    Targets(np.ones((2, DIM), np.float16)).save(Targets.directory(root) / "a.npy")
    heard = Auxiliary.of(AuxiliaryConfig(speech=1.0, semantic=0.5), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [type(t) for t in heard.terms] == [SpeechTerm, SemanticTerm]
    assert cast("SpeechTerm", heard.terms[0]).speech.rows.tolist() == [0, -1]
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(acoustic=1.0), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    Targets(np.ones((2, LENGTH), np.float16)).save(Loudness.directory(root, CUT) / "a.npy")
    loud = Auxiliary.of(AuxiliaryConfig(acoustic=0.4), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [(type(t), t.weight) for t in loud.terms] == [(AcousticTerm, 0.4)]
    assert cast("AcousticTerm", loud.terms[0]).speech.rows.tolist() == [0, -1]


def test_of_run_adversary(tmp_path: Path) -> None:
    """The run target builds the run adversary at weight 1, one class per training run, whatever its strength."""
    against_runs = AdversaryConfig(adversary=0.05, adversary_target=AdversaryTarget.RUN)
    runs = Auxiliary.of(AuxiliaryConfig(), against_runs, Tables(cast("DataRoot", Root(tmp_path)), CPU, WORDS, CUT))
    assert [(type(t), t.weight) for t in runs.terms] == [(RunTerm, 1.0)]
    assert cast("RunTerm", runs.terms[0]).classes == 2


def test_of_uncovered(tmp_path: Path) -> None:
    """Targets present for no training word, as a mis-pathed directory leaves, are refused, not trained at 0."""
    root = cast("DataRoot", Root(tmp_path))
    Targets(np.ones((2, DIM), np.float16)).save(Targets.directory(root) / "elsewhere.npy")
    Targets(np.ones((2, LENGTH), np.float16)).save(Loudness.directory(root, CUT) / "elsewhere.npy")
    for weights in (AuxiliaryConfig(speech=1.0), AuxiliaryConfig(acoustic=1.0), AuxiliaryConfig(audio=1.0)):
        with pytest.raises(ValueError, match="train on nothing"):
            Auxiliary.of(weights, AdversaryConfig(), Tables(root, CPU, WORDS, CUT))


def test_of_audio(tmp_path: Path) -> None:
    """The label-free audio contrast reads the same envelopes as the acoustic term."""
    root = cast("DataRoot", Root(tmp_path))
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(AuxiliaryConfig(audio=0.7), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    Targets(np.ones((2, LENGTH), np.float16)).save(Loudness.directory(root, CUT) / "a.npy")
    heard = Auxiliary.of(AuxiliaryConfig(audio=0.7), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    assert [(type(t), t.weight) for t in heard.terms] == [(AudioTerm, 0.7)]
    assert cast("AudioTerm", heard.terms[0]).speech.rows.tolist() == [0, -1]
    both = Auxiliary.of(
        AuxiliaryConfig(acoustic=0.3, audio=0.7), AdversaryConfig(), Tables(root, CPU, WORDS, CUT)
    ).terms
    assert cast("AcousticTerm", both[0]).speech is cast("AudioTerm", both[1]).speech
    wide = Cut(75, 0)
    Targets(np.ones((2, 75), np.float16)).save(Loudness.directory(root, wide) / "a.npy")
    widened = Auxiliary.of(
        AuxiliaryConfig(acoustic=0.3, audio=0.7), AdversaryConfig(), Tables(root, CPU, WORDS, wide)
    ).terms
    assert [cast("Acoustic | Listening", t.head(4)).frames for t in widened] == [75, 75]


def test_of_instance(tmp_path: Path) -> None:
    """The instance term needs MASC words to find any cross-subject positive."""
    root = cast("DataRoot", Root(tmp_path))
    with pytest.raises(ValueError, match="MASC instance"):
        Auxiliary.of(AuxiliaryConfig(instance=0.25), AdversaryConfig(), Tables(root, CPU, WORDS, CUT))
    runs = ["sub-101_ses-0_task-MASC1_run-1", "sub-102_ses-0_task-MASC1_run-1"]
    masc = pd.DataFrame({"run": runs, "position": 0, "sentence": 0})
    paired = Auxiliary.of(AuxiliaryConfig(instance=0.25), AdversaryConfig(), Tables(root, CPU, masc, CUT))
    assert [(type(t), t.weight) for t in paired.terms] == [(InstanceTerm, 0.25)]
    assert cast("Joint", paired).tapped(paired.terms[0]) is Tap.WORD
    assert type(cast("InstanceTerm", paired.terms[0]).neighbours) is Together
    apart = Auxiliary.of(
        AuxiliaryConfig(instance=0.25, instance_apart=True), AdversaryConfig(), Tables(root, CPU, masc, CUT)
    )
    assert isinstance(cast("InstanceTerm", apart.terms[0]).neighbours, Apart)


def test_of_contrast(tmp_path: Path) -> None:
    """The same-word term reads the run's tap, unlike the instance term's local one."""
    root = cast("DataRoot", Root(tmp_path))
    words = pd.DataFrame({"run": ["a", "a"], "position": [0, 1], "sentence": [0, 1]})
    context = Auxiliary.of(
        AuxiliaryConfig(contrast=0.4, tap=Tap.CONTEXT), AdversaryConfig(), Tables(root, CPU, words, CUT)
    )
    assert [(type(t), t.weight) for t in context.terms] == [(SameWordTerm, 0.4)]
    assert cast("Joint", context).tapped(context.terms[0]) is Tap.CONTEXT


def test_joint_trained_tagged() -> None:
    tagged = SentenceNet(
        nn.Identity(),
        ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2), head=WordHead.TAGGED)),
    )
    trained = Joint((TagTerm(1.0, torch.tensor([3, -1])),)).trained(tagged, identity)
    assert trained.heads[0](torch.zeros(1, tagged.words.width)).shape == (1, len(CLASSES))
    assert {id(p) for p in trained.heads.parameters()} <= {id(p) for p in tagged.parameters()}
    with pytest.raises(ValueError, match="tagged"):
        Joint((TagTerm(1.0, torch.tensor([3, -1])),)).trained(net(), identity)


def test_joint_acoustic_taps_the_time_course() -> None:
    model = net()
    loud = AcousticTerm(1.0, Speech(torch.tensor([0, -1]), torch.zeros(1, LENGTH)))
    trained = Joint((loud, semantic()), Tap.CONTEXT).trained(model, identity)
    assert trained.taps == (Tap.TIME, Tap.CONTEXT)
    assert isinstance(trained.heads[0], Acoustic)
    heard = AudioTerm(1.0, Speech(torch.tensor([0, -1]), torch.zeros(1, LENGTH)))
    listening = Joint((heard,), Tap.WORD).trained(net(), identity)
    assert listening.taps == (Tap.TIME,)
    assert isinstance(listening.heads[0], Listening)


def test_listens() -> None:
    assert AuxiliaryConfig(acoustic=0.3).listens
    assert AuxiliaryConfig(audio=1.0).listens
    assert not AuxiliaryConfig(semantic=1.0).listens


def test_rate() -> None:
    assert AuxiliaryConfig(audio=1.0).rate == FRAMES
    assert AuxiliaryConfig(semantic=1.0).rate is None


def test_time_tap_refused() -> None:
    assert AuxiliaryConfig(tap=Tap.CONTEXT).tap is Tap.CONTEXT
    with pytest.raises(ValueError, match="acoustic head"):
        AuxiliaryConfig(tap=Tap.TIME)


def test_tapped() -> None:
    joint = Joint((), Tap.CONTEXT)
    loud = AcousticTerm(1.0, Speech(torch.tensor([0]), torch.zeros(1, LENGTH)))
    assert joint.tapped(loud) is Tap.TIME
    assert joint.tapped(NextWordTerm(weight=1.0)) is Tap.WORD
    assert joint.tapped(semantic()) is Tap.CONTEXT
