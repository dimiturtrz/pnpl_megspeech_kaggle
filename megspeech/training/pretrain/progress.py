"""A frozen-encoder progress probe for pretraining: the self-supervised loss moves with its EMA target, so epochs are
not comparable by it. Each reading embeds subject 0's labelled words with an encoder as it stands, classifies the
carved-val words by their nearest train words (weighted k-NN on cosine similarity, as DINO evaluates), and scores
that with the selection ruler: BAcc@10, 0.2 at chance. Only the carved val is read, never a local test."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Run
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.model.sentence import SentenceNet
from megspeech.training.pretrain.windows import Sessions

SEED = 0
BATCH = 256


@dataclass(frozen=True)
class Probing:
    every: int = 2
    """Epochs between probes, student and teacher; 0 disables. One is also read before any step, as the random-init
    floor, and one after the last epoch."""
    words: int = 4000
    """Labelled train words, and as many val words, the probe draws."""
    k: int = 20
    """Neighbours voting (DINO's evaluation: 20)."""

    def __post_init__(self) -> None:
        if self.every < 0 or self.words < 1 or self.k < 1:
            raise ValueError(f"probing {self} needs every >= 0 and positive words and k")


@dataclass(frozen=True)
class Progress:
    """Train and carved-val words, and the labelled `train_rows` / `val_rows` of each that the probe reads."""

    train: Split
    val: Split
    train_rows: Int[Tensor, " a"]
    val_rows: Int[Tensor, " b"]
    device: torch.device
    k: int = 20
    temperature: float = 0.07
    """The softmax temperature of a neighbour's vote over its similarity (DINO's evaluation: 0.07)."""
    ruler: Ruler = field(default_factory=Ruler)

    @classmethod
    def of(cls, train: Split, val: Split, probing: Probing, device: torch.device) -> "Progress":
        """A probe over at most `probing.words` randomly drawn labelled words of each split."""
        generator = torch.Generator().manual_seed(SEED)
        rows = (cls.draw(train, probing.words, generator), cls.draw(val, probing.words, generator))
        return cls(train, val, *rows, device, probing.k)

    @staticmethod
    def draw(split: Split, words: int, generator: torch.Generator) -> Int[Tensor, " n"]:
        """At most `words` of `split`'s labelled words, in split order."""
        labelled = split.labelled().cpu()
        return labelled[torch.randperm(len(labelled), generator=generator)[:words]].sort().values

    @classmethod
    def load(
        cls, data: DataConfig, root: DataRoot, train: DeviceRecording, probing: Probing, device: torch.device
    ) -> "Progress":
        """The probe over `train`'s words and those of the carved-val sessions, windows cut as supervised training
        cuts them."""
        names = [run.name for run in Sessions.carving(data).carved(Run.everywhere(root))]
        held = DeviceRecording.load(
            data.run_cache(root),
            train.meg.device,
            names,
            geometry=data.geometry(root, train.meg.device),
        )
        return cls.of(cls.cut(train, data), cls.cut(held, data), probing, device)

    @staticmethod
    def cut(source: DeviceRecording, data: DataConfig) -> Split:
        """Every word of `source` that `data.grouping` carries, as a split."""
        table = source.words.assign(position=source.words.groupby("run").cumcount())
        words = data.carried(table)
        return Corpus.split(source, words, data, np.ones(len(words)))

    @staticmethod
    @contextmanager
    def evaluating(net: nn.Module) -> Iterator[None]:
        """`net` in eval mode for the block, its mode restored after."""
        was = net.training
        net.eval()
        yield
        net.train(was)

    @torch.no_grad()
    def embed(self, net: SentenceNet, split: Split, rows: Int[Tensor, " n"]) -> Float[Tensor, "n e"]:
        """The unit-normalised word embedding `net` gives each of `split`'s `rows`."""
        out = []
        for part in rows.split(BATCH):
            windows = split.windows(part.to(split.y.device)).to(self.device)
            subject = torch.full((len(part),), DEEP, dtype=torch.long, device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                out.append(net.embed(windows, subject).float())
        return functional.normalize(torch.cat(out), dim=-1)

    @staticmethod
    def vote(
        similarity: Float[Tensor, "m n"], labels: Int[Tensor, " n"], k: int, temperature: float
    ) -> Float[Tensor, "m v"]:
        """Each row's scores over the vocabulary: its `k` most similar columns each voting their label with weight
        `exp(similarity / temperature)`, each label's votes divided by its count among the columns. Raw votes follow
        the prior, so the same frequent words win every row's top 10 and the balanced ruler reads chance."""
        nearest = similarity.topk(min(k, similarity.shape[1]), dim=1)
        weight = (nearest.values / temperature).exp()
        votes = torch.zeros(len(similarity), VOCAB_SIZE, device=similarity.device)
        counts = torch.bincount(labels, minlength=VOCAB_SIZE).clamp(min=1)
        return votes.scatter_add_(1, labels[nearest.indices], weight) / counts

    @torch.no_grad()
    def score(self, net: SentenceNet) -> float:
        """Balanced accuracy at 10 candidates of the carved-val words classified by `net`'s embeddings."""
        with self.evaluating(net):
            train = self.embed(net, self.train, self.train_rows)
            val = self.embed(net, self.val, self.val_rows)
        votes = self.vote(val @ train.T, self.train.y[self.train_rows].to(self.device), self.k, self.temperature)
        labels = self.val.y[self.val_rows].to(self.device)
        return float(self.ruler.balanced(self.ruler.hits(votes, labels), labels))

    def read(self, student: SentenceNet, teacher: SentenceNet) -> dict[str, float]:
        """Both encoders' scores, as `probe/student` and `probe/teacher`."""
        return {"probe/student": self.score(student), "probe/teacher": self.score(teacher)}
