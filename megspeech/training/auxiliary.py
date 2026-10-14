"""What trains beside the 50-way heads: nothing, or terms read off every heard word's embedding — its t5
embedding aligned, its word classified, its leading phonemes classified."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from typing import ClassVar, override

import pandas as pd
import torch
from torch import nn

from core.paths import DataRoot
from core.rules import Rule
from megspeech.data.config import Cut
from megspeech.data.labels.context_tags import ContextTags
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.acoustic import AcousticTerm
from megspeech.objective.auxiliary.adversarial import TimingTerm
from megspeech.objective.auxiliary.audio import AudioTerm
from megspeech.objective.auxiliary.contrastive.instance import InstanceTerm
from megspeech.objective.auxiliary.contrastive.word import SameWordTerm
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.auxiliary.neighbour import NextWordTerm, PreviousWordTerm
from megspeech.objective.auxiliary.phonetic import PhoneticTerm
from megspeech.objective.auxiliary.semantic import SemanticTerm
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.auxiliary.syntactic import SyntacticTerm
from megspeech.objective.auxiliary.tagging import TagTerm
from megspeech.objective.auxiliary.window import WindowTerm
from megspeech.objective.targets.loudness import FRAMES, Loudness
from megspeech.objective.targets.speech import Speech, Targets
from megspeech.objective.targets.t5 import Lexicon
from megspeech.objective.term import AuxiliaryTerm


@dataclass(frozen=True)
class AuxiliaryConfig:
    """The weights and knobs of the terms trained beside the 50-way heads; the `optim.auxiliaries` group, its leaves
    also under their flat `--optim.<name>` flags."""

    FLAT: ClassVar[bool] = True

    semantic: float = 0.0
    """Weight of the semantic auxiliary (`megspeech.objective.auxiliary.semantic`): every heard word's window, the
    ~60% outside the 50 included, SigLIP-aligned to the word's t5-large layer-12 embedding, d'Ascoli et al.'s
    objective. It supervises ~2.5x the windows the heads see; 0 trains the heads alone and never reads the lexicon."""
    window: float = 0.0
    """Weight of the window auxiliary (`megspeech.objective.auxiliary.window`): every heard word's window also predicts
    the t5-large embeddings of the words at offsets -1 to +3 of its span, each offset weighted by the share of its
    0.2-0.5 s response the 1 s window holds, since at ~4 words/s it mixes the responses to ~4 words
    (megspeech-7gf.108); 0 never reads the lexicon."""
    lexical: float = 0.0
    """Weight of the lexical auxiliary: every heard word's window, in or out of the 50, classified into its word among
    the ~25k distinct words of the lexicon (`megspeech.objective.auxiliary.lexical`). Word identity, where the
    semantic auxiliary aligns to a frozen embedding; 0 never reads the lexicon."""
    phonetic: float = 0.0
    """Weight of the phonetic auxiliary: every heard word's window, in or out of the 50, classified into the first of
    its 39 ARPAbet phonemes (`megspeech.data.labels.phonemes`). Speech MEG decodes phonetic features (Gwilliams et al.
    2022), and the onset phoneme is what the window's first ~100 ms hear; 0 never reads the table."""
    syntactic: float = 0.0
    """Weight of the syntactic auxiliary: every heard word's window, in or out of the 50, classified into its 17-way
    universal part of speech as spaCy tags it in its sentence (`megspeech.data.labels.syntax`): dense placeholder
    classes for the windows the 50 leave unlabelled, where the ~25k-way lexical target is sparse; 0 never reads it."""
    acoustic: float = 0.0
    """Weight of the acoustic auxiliary (`megspeech.objective.auxiliary.acoustic`): every training word whose chapter
    audio is aligned, its window's per-timestep encoder features read through a learned channel mix and lag into
    the 50 Hz speech envelope heard over the window, contrasted chunk by chunk against the batch's true chunks
    (Défossez et al. 2023); a dense target the 50 labels' and the word-level targets' sparse supervision lacks. 0
    creates no head and reads no envelope (megspeech-7gf.125.1)."""
    audio: float = 0.0
    """Weight of the audio contrastive term (`megspeech.objective.auxiliary.audio`): the same windows and envelopes as
    `acoustic`, each embedded by its own tower and aligned by CLIP's symmetric InfoNCE across the batch's spans,
    subjects and languages. With `heads` 0 it is the label-free all-language encoder pretrain (megspeech-7gf.109.3);
    0 creates no head and reads no envelope."""
    speech: float = 0.0
    """Weight of the speech auxiliary (`megspeech.objective.auxiliary.speech`): every training word whose chapter audio
    is aligned, SigLIP-aligned to the wav2vec2 latent of its 0.5 s of heard speech (Défossez et al. 2023), one target
    per word token, not per word type as the semantic auxiliary's; 0 never reads the targets."""
    speech_source: str = ""
    """The targets the speech auxiliary aligns to, a directory of `<run>.npy` under processed/speech: empty is the
    wav2vec2 latents, `hubert-large-ll60k` / `data2vec-audio-large` those teachers' (a
    `megspeech.objective.targets.speech.Reading.stem`, 7gf.141.6), `gpt2-large` the in-context text states of
    `megspeech.objective.targets.context` (megspeech-7gf.101)."""
    speech_centred: bool = False
    """Centre each run's speech targets on its own mean (`Targets.centred`), removing the direction every target
    shares: raw, two random targets' cosine is ~0.6, so the L2-normalised SigLIP target is mostly that one direction."""
    speech_softmax: bool = False
    """Contrast the speech targets by softmax over the batch's kinds, not independent sigmoids: the sigmoid contrast
    cannot learn a word-identity oracle in 40 epochs where the 50-way heads learn it in 4."""
    vocab_weight: float = 1.0
    """Each row of a word in the 50 weighs this against 1 for the others in every auxiliary term's mean, normalised by
    the weights (megspeech-7gf.126); 1 weighs every row alike."""
    tagging: float = 0.0
    """Weight of the tag auxiliary (`megspeech.objective.auxiliary.tagging`): every training word's embedding classified
    into its in-context part of speech (`megspeech.data.labels.context_tags`), 11 universal tags and `OTHER`. It trains
    the tag head a `tagged` word head (`ConvConfig.head`) mixes its per-tag readouts by; 0 never reads the table."""
    next_word: float = 0.0
    """Weight of the next-word auxiliary (`megspeech.objective.auxiliary.neighbour`): each word slot's local encoder
    embedding, before the span transformer whatever the `tap`, predicts the next word's 50-way label in its span
    (ignored where that word is outside the 50, absent or in another span); 0 creates no head."""
    prev_word: float = 0.0
    """Weight of the previous-word auxiliary, the next-word one's mirror; 0 creates no head."""
    instance: float = 0.0
    """Weight of the cross-subject instance term
    (`megspeech.objective.auxiliary.contrastive.instance`): InfoNCE over each word's
    local encoder embedding, its positives the same MASC token heard by another subject, its negatives every other
    word of the batch; `SamplerConfig.pairs` co-batches such pairs. 0 creates no head (megspeech-7gf.147.2)."""
    instance_apart: bool = False
    """Leave every other word of a word's own sentence, whoever heard it, out of the instance term's negatives
    (`megspeech.objective.auxiliary.contrastive.instance.Apart`): their windows overlap its sound (megspeech-ct5)."""
    contrast: float = 0.0
    """Weight of the same-word contrastive term
    (`megspeech.objective.auxiliary.contrastive.word`): supervised contrastive
    loss (Khosla et al. 2020) over the `tap` representation of every word in the 50, its positives the batch's other
    words of the same label from a different sentence, its negatives the other labels; pulls same-word embeddings
    together so a kNN readout finds them. Reads what `tap` names, `context` being what the kNN readout embeds; 0
    creates no head (megspeech-7gf.86)."""
    tap: Tap = Tap.WORD
    """What every auxiliary head reads: the encoder's word embedding, or the span transformer's output at the word
    (d'Ascoli et al. train that output of every heard word toward t5; the context head alone sees only the 50's)."""
    positions: int = 1
    """Leading phonemes the phonetic auxiliary classifies, up to `megspeech.data.labels.phonemes.POSITIONS`: a 0.5 s
    window hears ~4 of ~0.1 s, and 1 is the onset phoneme alone."""

    def __post_init__(self) -> None:
        Rule.enforce(self, AUXILIARY_RULES)

    @property
    def listens(self) -> bool:
        """Whether a term reads the window's time course against its envelope."""
        return bool(self.acoustic or self.audio)

    @property
    def rate(self) -> int | None:
        """The cache rate in Hz the envelope terms cut their targets in; None when no term listens."""
        return FRAMES if self.listens else None


AUXILIARY_RULES: tuple[Rule[AuxiliaryConfig], ...] = (
    Rule(
        lambda a: a.tap is Tap.TIME,
        lambda _: "optim.tap time reads the window's time course, which only the acoustic head asks for",
    ),
)


@dataclass(frozen=True)
class Auxiliary(ABC):
    """The module a run's optimiser steps, and the terms its auxiliary heads add to the heads' loss."""

    terms: tuple[AuxiliaryTerm, ...]

    @abstractmethod
    def trained(
        self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]
    ) -> SentenceNet | WithAuxiliary: ...

    @staticmethod
    def of(weights: AuxiliaryConfig, adversary: float, tables: "Tables") -> "Auxiliary":
        """Nothing when every weight is 0, so no table is read; otherwise the terms of the nonzero weights, the timing
        adversary's among them at its strength `adversary`, their tables read as `tables` says. A table several terms
        read is loaded once and shared, so the device holds one copy."""
        root, device, words = tables.root, tables.device, tables.words
        lexicon = cache(lambda: Lexicon.load(Lexicon.path(root)).to(device))
        loudness = cache(lambda: Speech.covering(Loudness.directory(root, tables.cut), words).to(device))
        builders: tuple[tuple[float, Callable[[], AuxiliaryTerm]], ...] = (
            (
                weights.speech,
                lambda: SpeechTerm(
                    weights.speech,
                    Speech.covering(
                        Targets.directory(root, weights.speech_source), words, centred=weights.speech_centred
                    ).to(device),
                    softmax=weights.speech_softmax,
                ),
            ),
            (weights.semantic, lambda: SemanticTerm(weights.semantic, lexicon())),
            (weights.lexical, lambda: LexicalTerm(weights.lexical, lexicon())),
            (
                weights.phonetic,
                lambda: PhoneticTerm(
                    weights.phonetic, Pronunciation.load(Pronunciation.path(root)).to(device), weights.positions
                ),
            ),
            (
                weights.syntactic,
                lambda: SyntacticTerm(weights.syntactic, PartsOfSpeech.load(PartsOfSpeech.path(root)).to(device)),
            ),
            (
                weights.tagging,
                lambda: TagTerm(weights.tagging, ContextTags.load(ContextTags.path(root)).tensor(words, device)),
            ),
            (weights.next_word, lambda: NextWordTerm(weight=weights.next_word)),
            (weights.prev_word, lambda: PreviousWordTerm(weight=weights.prev_word)),
            (adversary, lambda: TimingTerm.of(1.0, words, device)),
            (weights.instance, lambda: InstanceTerm.of(weights.instance, words, device, apart=weights.instance_apart)),
            (weights.contrast, lambda: SameWordTerm.of(weights.contrast, words, device)),
            (weights.window, lambda: WindowTerm(weights.window, lexicon())),
            (weights.acoustic, lambda: AcousticTerm(weights.acoustic, loudness())),
            (weights.audio, lambda: AudioTerm(weights.audio, loudness())),
        )
        terms = tuple(replace(build(), vocab_weight=weights.vocab_weight) for weight, build in builders if weight)
        return Joint(terms, weights.tap) if terms else NoAuxiliary()


@dataclass(frozen=True)
class Tables:
    """Where a run's auxiliary terms read their tables: the data `root`, the `device` they live on, the training
    `words` (`Corpus.positions`) the speech terms' targets are those of, and the `cut` of the run's windows
    (`DataConfig.cut`) the envelope terms' targets are cut as."""

    root: DataRoot
    device: torch.device
    words: pd.DataFrame
    cut: Cut


@dataclass(frozen=True)
class NoAuxiliary(Auxiliary):
    """The sentence model trained alone."""

    terms: tuple[AuxiliaryTerm, ...] = ()

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]) -> SentenceNet:
        return net


@dataclass(frozen=True)
class Joint(Auxiliary):
    """The sentence model wrapped with one head per term (`WithAuxiliary`), each reading its `tap`."""

    tap: Tap = Tap.WORD

    @override
    def trained(self, net: SentenceNet, compiled: Callable[[WithAuxiliary], WithAuxiliary]) -> WithAuxiliary:
        taps = tuple(self.tapped(term) for term in self.terms)
        heads = nn.ModuleList(
            term.readout(net.hierarchy.tagging(net.words.head), tap.width(net))
            for term, tap in zip(self.terms, taps, strict=True)
        )
        heads.to(next(net.parameters()).device)
        return compiled(WithAuxiliary(net, heads, self.tap, taps))

    def tapped(self, term: AuxiliaryTerm) -> Tap:
        """What `term`'s head reads: the window's time course if it is timed, the word embedding if local, else the
        run's `tap`."""
        if term.timed:
            return Tap.TIME
        return Tap.WORD if term.local else self.tap
