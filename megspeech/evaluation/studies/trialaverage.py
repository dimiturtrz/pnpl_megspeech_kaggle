"""The trial-averaging ceiling: how well a frozen word encoder reads a word off MEG when `k` same-word held-out trials
are averaged before it is read. A fast rise with `k` means the word is in the signal and single-trial noise is what
limits it; a low plateau means the word identity is weakly in the MEG, whatever the model.

Each trial's embedding is the stage-1 per-word encoder output (before the sentence transformer). Per held-out domain
and `k`, every class with at least `k` trials contributes one average of `k` random trials per draw (seeded, `draws`
repeats), read two ways: cosine to the class centroids of the training sessions' embeddings, and the model's own word
head, logit-adjusted by `TAU`. A score is the top-10 hit rate over eligible classes; `fixed` restricts every `k` to the
classes holding at least the largest `k`, so the curve is not moved by classes dropping out.

    uv run python -m megspeech.evaluation.studies.trialaverage sent40-s0 --partition test
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import DEEP, Partition, Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.ensemble import Ensemble
from megspeech.model.sentence import SentenceNet

LOG = logging.getLogger(__name__)

TRIALS = (1, 2, 4, 8, 16, 32)
DRAWS = 50
TAU = 1.0
BATCH = 256
"""Windows embedded per forward pass; small, as the GPU is shared with training."""
CHUNK = 6
"""Training runs loaded on the device at once while the centroids accumulate."""
Readout = Callable[[Float[Tensor, "c e"]], Float[Tensor, "c v"]]


@dataclass(frozen=True)
class Centroids:
    """Per-class sums and counts of embeddings, so training sessions stream through in chunks."""

    sums: Float[Tensor, "v e"]
    counts: Int[Tensor, " v"]

    @classmethod
    def empty(cls, width: int) -> "Centroids":
        return cls(torch.zeros(VOCAB_SIZE, width), torch.zeros(VOCAB_SIZE, dtype=torch.long))

    def add(self, embeddings: Float[Tensor, "n e"], labels: Int[Tensor, " n"]) -> "Centroids":
        """These centroids with the labelled `embeddings` (on the CPU) counted in."""
        sums = self.sums.index_add(0, labels, embeddings)
        return Centroids(sums, self.counts + torch.bincount(labels, minlength=VOCAB_SIZE))

    def scores(self, embeddings: Float[Tensor, "c e"]) -> Float[Tensor, "c v"]:
        """Cosine of each embedding to every class mean; a class with no training trial is never ranked."""
        means = nn.functional.normalize(self.sums / self.counts.clamp(min=1)[:, None], dim=-1)
        cosine = nn.functional.normalize(embeddings, dim=-1) @ means.T
        return cosine.masked_fill(self.counts == 0, float("-inf"))


@dataclass(frozen=True)
class Averaging:
    """Held-out trials: their `embeddings` and word `labels`, drawn from `seed`."""

    embeddings: Float[Tensor, "n e"]
    labels: Int[Tensor, " n"]
    seed: int = 0
    draws: int = DRAWS

    @property
    def counts(self) -> Int[Tensor, " v"]:
        return torch.bincount(self.labels, minlength=VOCAB_SIZE)

    def draw(self, k: int, generator: torch.Generator) -> tuple[Float[Tensor, "c e"], Int[Tensor, " c"]]:
        """One average of `k` random trials of every class holding at least `k`, and the classes."""
        classes = torch.nonzero(self.counts >= k).flatten()
        if not len(classes):
            return self.embeddings[:0], classes
        picks = [torch.nonzero(self.labels == c).flatten() for c in classes]
        chosen = [p[torch.randperm(len(p), generator=generator)[:k]] for p in picks]
        return torch.stack([self.embeddings[c].mean(dim=0) for c in chosen]), classes

    def rates(self, k: int, readout: Readout) -> Float[Tensor, " v"]:
        """Each class's top-10 hit rate over the draws of `k`-trial averages; NaN for a class with fewer than `k`."""
        generator = torch.Generator().manual_seed(self.seed)
        rates = torch.full((VOCAB_SIZE,), float("nan"))
        hits = torch.zeros(VOCAB_SIZE)
        for _ in range(self.draws):
            averaged, classes = self.draw(k, generator)
            hits[classes] += Ruler().hits(readout(averaged), classes).float()
        present = self.counts >= k
        rates[present] = hits[present] / self.draws
        return rates

    def curve(self, readout: Readout, trials: tuple[int, ...] = TRIALS) -> Float[Tensor, "k v"]:
        """`rates` for every `k` of `trials`."""
        return torch.stack([self.rates(k, readout) for k in trials])

    @staticmethod
    def balanced(curve: Float[Tensor, "k v"], keep: Bool[Tensor, " v"] | None = None) -> Float[Tensor, " k"]:
        """Mean over each row's eligible classes (those not NaN, and in `keep` when given)."""
        masked = curve if keep is None else curve.masked_fill(~keep, float("nan"))
        return masked.nanmean(dim=1)


@dataclass(frozen=True)
class Encoder:
    """A trained sentence model's per-word encoder, frozen, reading windows alone."""

    loaded: LoadedModel

    @property
    def network(self) -> SentenceNet:
        return cast("SentenceNet", cast("Ensemble", self.loaded.model).members[0])

    def embed(self, split: Split) -> Float[Tensor, "n e"]:
        """The stage-1 embedding of every word window of `split`, on the CPU."""
        net = self.network.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for words in torch.arange(len(split.y), device=split.y.device).split(BATCH):
                valid = torch.ones(len(words), dtype=torch.bool, device=words.device)
                out.append(net.embed(split.windows(words), split.subject[words], valid).float().cpu())
        return torch.cat(out)

    def head(self, embeddings: Float[Tensor, "c e"], tau: float = TAU) -> Float[Tensor, "c v"]:
        """The model's word head on `embeddings`, logit-adjusted by `tau` (`TAU`: the balanced metric's)."""
        device = self.loaded.log_prior.device
        with torch.no_grad():
            logits = self.network.word_logits(embeddings.to(device)).float().log_softmax(-1)
        return (logits - tau * self.loaded.log_prior).cpu()

    def centroids(self, runs: list[Run]) -> Centroids:
        """Class centroids of the embeddings of `runs`' labelled words, `CHUNK` runs on the device at a time."""
        found = None
        for start in range(0, len(runs), CHUNK):
            for _, split in self.loaded.splits(runs[start : start + CHUNK]).values():
                labelled = split.labelled()
                embeddings = self.embed(split)[labelled.cpu()]
                found = found or Centroids.empty(embeddings.shape[1])
                found = found.add(embeddings, split.y[labelled].cpu())
        if found is None:
            raise ValueError("no training runs to take centroids from")
        return found


@dataclass(frozen=True)
class Ceiling:
    """The trial-averaging curves of a loaded model's held-out domains."""

    encoder: Encoder
    centroids: Centroids
    seed: int = 0
    draws: int = DRAWS

    def domain(self, split: Split) -> tuple[Averaging, dict[str, Float[Tensor, "k v"]]]:
        """`split`'s labelled words averaged, and the per-class curve of each readout."""
        labelled = split.labelled()
        averaging = Averaging(self.encoder.embed(split)[labelled.cpu()], split.y[labelled].cpu(), self.seed, self.draws)
        readouts = {
            "centroid": self.centroids.scores,
            "head": self.encoder.head,
            "head-tau0": partial(self.encoder.head, tau=0.0),
        }
        return averaging, {name: averaging.curve(read) for name, read in readouts.items()}

    @staticmethod
    def lines(name: str, averaging: Averaging, curves: dict[str, Float[Tensor, "k v"]]) -> list[str]:
        """One line per readout: top-10 balanced accuracy at each `k`, over eligible classes and over the fixed set."""
        counts = averaging.counts
        fixed = counts >= TRIALS[-1]
        out = [f"{name} n={len(averaging.labels)} classes={int((counts > 0).sum())} fixed_classes={int(fixed.sum())}"]
        for readout, curve in curves.items():
            eligible, held = Averaging.balanced(curve).tolist(), Averaging.balanced(curve, fixed).tolist()
            seen = [int((counts >= k).sum()) for k in TRIALS]
            scored = zip(TRIALS, eligible, seen, strict=True)
            out.append(f"{name} {readout} eligible " + " ".join(f"k{k}={v:.4f}(c={c})" for k, v, c in scored))
            out.append(
                f"{name} {readout} fixed " + " ".join(f"k{k}={v:.4f}" for k, v in zip(TRIALS, held, strict=True))
            )
        return out

    def report(self, domains: dict[str, Split]) -> list[str]:
        """Every domain's lines."""
        lines: list[str] = []
        for name, split in domains.items():
            lines += self.lines(name, *self.domain(split))
        return lines


class Study:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="a trained model under models/ (the official ruler)")
        parser.add_argument("--partition", type=Partition, default=Partition.TEST)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--draws", type=int, default=DRAWS)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, device = DataRoot.from_config(), torch.device("cuda")
        loaded = Loader.of(root, [args.name], Scoring(partition=args.partition), device)
        encoder = Encoder(loaded)
        trained = [r for r in loaded.data.trained(Run.everywhere(root)) if r.subject == DEEP]
        centroids = encoder.centroids(trained)
        LOG.info("centroids from %d runs, %d words", len(trained), int(centroids.counts.sum()))
        domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
        for line in Ceiling(encoder, centroids, args.seed, args.draws).report(domains):
            LOG.info("%s %s", args.partition, line)


if __name__ == "__main__":
    Study.main()
