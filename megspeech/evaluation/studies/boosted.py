"""The peer protocol's booster: a small net that learns only what a frozen base gets wrong from per-word side
information about each word and its sentence neighbours (megspeech-6ow).

Out = base log-probabilities + booster(features); the base is never trained here, and the booster's last layer starts
at zero, so epoch 0 is the base itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD

GAP_CEILING = 3.0
"""Seconds past which an onset gap carries no more: sentences' words are within ~1 s."""
LOG_FLOOR = -12.0
"""Floor of a lexical log-probability: the 50 columns of a 25k-way softmax reach far below any informative value."""
STD_FLOOR = 1e-3
"""Smallest feature spread a standardised feature is divided by."""
SIDES = 2
"""Neighbours a word has: the one before and the one after."""


class Side(StrEnum):
    """Per-word side information beyond the base's own logits."""

    TAGS = "tags"
    LEXICAL = "lexical"
    TIMING = "timing"


@dataclass(frozen=True)
class Spanned:
    """A split's spans as flat arrays: `index[g]` the words of span `g` (`PAD` past its end), `onset` relative to the
    span's first word."""

    index: Int[np.ndarray, "g w"]
    onset: Float[np.ndarray, "g w"]

    def pairs(self) -> tuple[Int[np.ndarray, " k"], Int[np.ndarray, " k"], Float[np.ndarray, " k"]]:
        """Every adjacent word pair of one span: the earlier word, the later word and the seconds between them."""
        both = (self.index[:, :-1] != PAD) & (self.index[:, 1:] != PAD)
        return self.index[:, :-1][both], self.index[:, 1:][both], (self.onset[:, 1:] - self.onset[:, :-1])[both]

    def near(self, count: int) -> Int[np.ndarray, "n 2"]:
        """Each of `count` words' previous and next word in its span, `PAD` where it has none."""
        before, after, _ = self.pairs()
        out = np.full((count, SIDES), PAD, dtype=np.int64)
        out[after, 0] = before
        out[before, 1] = after
        return out

    def gaps(self, count: int) -> Float[np.ndarray, "n 2"]:
        """Seconds from each of `count` words to its previous and to its next onset, 0 where it has none."""
        before, after, gap = self.pairs()
        out = np.zeros((count, SIDES))
        out[after, 0] = gap
        out[before, 1] = gap
        return out


@dataclass(frozen=True)
class Design:
    """What a booster trains on: each word's input, its base's log-probabilities and its label."""

    x: Float[Tensor, "n d"]
    base: Float[Tensor, "n v"]
    y: Int[Tensor, " n"]

    def where(self, keep: Bool[Tensor, " n"]) -> "Design":
        return Design(self.x[keep], self.base[keep], self.y[keep])

    def labelled(self) -> "Design":
        """The words with a vocabulary label: the rows a loss counts."""
        return self.where(self.y != IGNORE)

    def to(self, device: torch.device) -> "Design":
        return Design(self.x.to(device), self.base.to(device), self.y.to(device))


@dataclass(frozen=True)
class Rows:
    """Every word of some splits, as a frozen base and its probes read it: the base's log-probabilities, the tag
    head's probabilities, the lexical head's log-probabilities of the 50, the onset gaps, the label, each word's
    neighbours (rows of this same table, -1 for none) and the group (fold or domain) it belongs to."""

    base: Float[Tensor, "n v"]
    tags: Float[Tensor, "n t"]
    lexical: Float[Tensor, "n v"]
    gaps: Float[Tensor, "n 2"]
    y: Int[Tensor, " n"]
    near: Int[Tensor, "n 2"]
    group: Int[Tensor, " n"]

    @classmethod
    def stacked(cls, parts: Sequence["Rows"]) -> "Rows":
        """`parts` one after another, each part's neighbours moved to their rows in the whole."""
        starts = np.cumsum([0, *(len(part.y) for part in parts)])[:-1]
        moved = [
            torch.where(part.near >= 0, part.near + int(start), part.near)
            for part, start in zip(parts, starts, strict=True)
        ]
        names = ("base", "tags", "lexical", "gaps", "y", "group")
        return cls(
            **{name: torch.cat([getattr(part, name) for part in parts]) for name in names}, near=torch.cat(moved)
        )

    def side(self, kinds: Sequence[Side]) -> Float[Tensor, "n f"]:
        """Each word's own side information of `kinds`, side by side."""
        columns = {
            Side.TAGS: self.tags,
            Side.LEXICAL: self.lexical.clamp(min=LOG_FLOOR),
            Side.TIMING: self.gaps.clamp(max=GAP_CEILING),
        }
        return torch.cat([columns[kind] for kind in kinds], dim=1) if kinds else self.base[:, :0]

    def design(self, kinds: Sequence[Side], *, neighbours: bool) -> Design:
        """The booster's input per word: the base's log-probabilities, the word's side information of `kinds` and, with
        `neighbours`, its previous and next word's (zeros where it has none) with whether each exists."""
        own = self.side(kinds)
        parts = [self.base, own]
        if neighbours and kinds:
            padded = torch.cat([own, own.new_zeros(1, own.shape[1])])
            at = torch.where(self.near >= 0, self.near, len(own))
            parts += [padded[at].flatten(1), (self.near >= 0).float()]
        return Design(torch.cat(parts, dim=1), self.base, self.y)


class Booster(nn.Module):
    """An MLP residual on a frozen base's log-probabilities, its inputs standardised by the training rows' own."""

    mean: Tensor
    scale: Tensor

    def __init__(self, design: Design, hidden: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("mean", design.x.mean(0))
        self.register_buffer("scale", design.x.std(0).clamp(min=STD_FLOOR))
        self.body = nn.Sequential(nn.Linear(design.x.shape[1], hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, design.base.shape[1])
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @override
    def forward(self, x: Float[Tensor, "n d"], base: Float[Tensor, "n v"]) -> Float[Tensor, "n v"]:
        return base + self.out(self.body((x - self.mean) / self.scale))


@dataclass(frozen=True)
class Boosting:
    """How a booster is trained: AdamW at `lr` and `decay` over minibatches of `batch` rows, from `seed`."""

    seed: int
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    decay: float = 0.05
    batch: int = 512

    def fit(self, train: Design, epochs: int, watch: Design | None = None) -> tuple[Booster, list[float]]:
        """The booster trained `epochs` epochs on the labelled rows of `train`, and the cross-entropy of `watch`'s
        labelled rows after each epoch (empty without `watch`)."""
        torch.manual_seed(self.seed)
        rows = train.labelled()
        device = rows.x.device
        model = Booster(rows, self.hidden, self.dropout).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.decay)
        losses: list[float] = []
        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(rows.y), device=device)
            for chosen in order.split(self.batch):
                loss = functional.cross_entropy(model(rows.x[chosen], rows.base[chosen]), rows.y[chosen])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            if watch is not None:
                losses.append(self.loss(model, watch.labelled()))
        return model, losses

    @staticmethod
    def loss(model: Booster, design: Design) -> float:
        """Mean cross-entropy of `model` on `design`."""
        model.eval()
        with torch.no_grad():
            return float(functional.cross_entropy(model(design.x, design.base), design.y))

    @staticmethod
    def predict(model: Booster, design: Design) -> Float[Tensor, "n v"]:
        """`model`'s log-probabilities for every row of `design`, labelled or not."""
        model.eval()
        with torch.no_grad():
            return model(design.x, design.base).log_softmax(-1)


@dataclass(frozen=True)
class Arm:
    """One booster configuration: which side information it reads and whether it reads the neighbours'."""

    name: str
    kinds: tuple[Side, ...]
    neighbours: bool


ARMS = (
    Arm("base-only", (), neighbours=False),
    Arm("full", (Side.TAGS, Side.LEXICAL, Side.TIMING), neighbours=True),
    Arm("no-neighbours", (Side.TAGS, Side.LEXICAL, Side.TIMING), neighbours=False),
    Arm("tags-only", (Side.TAGS,), neighbours=True),
    Arm("lexical-only", (Side.LEXICAL,), neighbours=True),
    Arm("timing-only", (Side.TIMING,), neighbours=True),
)
