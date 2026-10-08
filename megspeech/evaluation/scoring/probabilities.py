"""A kept model's per-word predictions, saved beside it so a study never reloads the model to read them again.

Each section (`val`: the selecting validation read in training; `official-val` and `local-test`: the report's
partitions) holds, per domain, every labelled word's id, label, raw context-head log-probabilities (and the word
head's), and the probabilities after the report's own adjustment (logit-adjusted at the selected tau, balanced over
the domain's labelled words, softmaxed). `novel` marks the unseen-span words that selection and the studies score.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head

PROBS = "probs.pt"
META = "#meta"
SECTIONS = {Partition.VALIDATION: "official-val", Partition.TEST: "local-test"}
"""The sections of `probs.pt` each partition's predictions are saved under."""


@dataclass(frozen=True)
class Words:
    """One domain's labelled words, in word order."""

    ids: Int[Tensor, " n"]
    labels: Int[Tensor, " n"]
    novel: Bool[Tensor, " n"]
    logits: Float[Tensor, "n v"]
    word_logits: Float[Tensor, "n v"]
    probabilities: Float[Tensor, "n v"]

    @classmethod
    def of(
        cls,
        evaluator: Evaluator,
        logits: dict[Head, Float[Tensor, "n v"]],
        split: Split,
        log_prior: Float[Tensor, " v"],
        tau: float,
    ) -> "Words":
        ids = split.labelled()
        context = logits[Head.CONTEXT][ids]
        adjusted = evaluator.adjusted(context, log_prior, tau)
        return cls(
            ids.cpu(),
            split.y[ids].cpu(),
            torch.isin(ids, split.novel(evaluator.seen)).cpu(),
            context.cpu(),
            logits[Head.WORD][ids].cpu(),
            torch.softmax(adjusted, dim=-1).cpu(),
        )

    def record(self) -> dict[str, Tensor]:
        return {
            "ids": self.ids,
            "labels": self.labels,
            "novel": self.novel,
            "logits": self.logits,
            "word_logits": self.word_logits,
            "probabilities": self.probabilities,
        }


@dataclass(frozen=True)
class Probabilities:
    """The domains of one section, the tau their probabilities were adjusted at, the prior, and the unseen-span words
    a domain needs to count."""

    domains: dict[str, Words]
    tau: float
    log_prior: Float[Tensor, " v"]
    min_novel: int

    @classmethod
    def of(
        cls,
        evaluator: Evaluator,
        model: nn.Module,
        domains: dict[str, Split],
        log_prior: Float[Tensor, " v"],
        tau: float,
    ) -> "Probabilities":
        """`model`'s predictions on every labelled word of `domains`."""
        logits = {name: evaluator.logits(model, split) for name, split in domains.items()}
        return cls.from_logits(evaluator, logits, domains, log_prior, tau)

    @classmethod
    def from_logits(
        cls,
        evaluator: Evaluator,
        logits: dict[str, dict[Head, Float[Tensor, "n v"]]],
        domains: dict[str, Split],
        log_prior: Float[Tensor, " v"],
        tau: float,
    ) -> "Probabilities":
        """The predictions on every labelled word of `domains` from their already computed logits."""
        words = {name: Words.of(evaluator, logits[name], split, log_prior, tau) for name, split in domains.items()}
        return cls(words, tau, log_prior.cpu(), evaluator.min_novel)

    @staticmethod
    def stored(path: Path) -> dict[str, dict[str, Tensor]]:
        """Every tensor record saved at `path` (`<section>#meta`, `<section>/<domain>`), none when there is no file."""
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {}

    def save(self, path: Path, section: str) -> Path:
        """`section` written into `path`, the file's other sections kept."""
        meta = {"tau": torch.tensor(self.tau), "log_prior": self.log_prior, "min_novel": torch.tensor(self.min_novel)}
        record = {f"{section}/{name}": words.record() for name, words in self.domains.items()}
        kept = {key: value for key, value in self.stored(path).items() if not key.startswith(f"{section}/")}
        torch.save({**kept, f"{section}{META}": meta, **record}, path)
        return path

    @classmethod
    def load(cls, path: Path, section: str) -> "Probabilities":
        stored = cls.stored(path)
        meta = stored[f"{section}{META}"]
        domains = {
            key.removeprefix(f"{section}/"): Words(**value)
            for key, value in stored.items()
            if key.startswith(f"{section}/")
        }
        return cls(domains, float(meta["tau"]), meta["log_prior"], int(meta["min_novel"]))
