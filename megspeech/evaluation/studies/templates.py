"""Raw-MEG per-word templates: are they reliable, how well do they decode a held-out word, what do they encode.

Every labelled training window (default Deep `DataConfig`, official ruler), scaled per channel by the training windows'
own mean and std, is averaged per word, pooled and per dataset (`DataConfig.stratum`). Three readings, no model:

1. Reliability: each word's training trials dealt into random halves (seeded, `REPS` repeats); the correlation of
   the two half-templates (sensors x time flattened) of the same word against different words, and the per-word
   Spearman-Brown step-up of the same-word correlation.
2. Decoding: official validation and test words scored against the training templates, pooled or from their own
   dataset, by Pearson correlation and by the LDA discriminant `T.S^-1.x - T.S^-1.T/2` under a noise covariance
   taken from the training residuals `x - T[label]`, Kronecker-separable (sensors x time, each Ledoit-Wolf shrunk, the
   shrinkage computed exactly from streamed moments) or sensors only; top-10 balanced accuracy.
3. Content: Spearman (permuted words, `PERMUTATIONS`) of the template distance matrix (1 - correlation) with the
   words' mean log duration (onset gap to the next word) difference, their first phonemes' Hamming distance and their
   t5 prototypes' cosine distance.

    uv run python -m megspeech.evaluation.studies.templates
"""

import argparse
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.timing import Readings
from megspeech.objective.targets.t5 import Prototypes

LOG = logging.getLogger(__name__)

REPS = 20
PERMUTATIONS = 1000
BATCH = 256
"""Windows scored or accumulated per step; small, as the GPU is shared with training."""
CHUNK = 6
"""Runs loaded on the device at once."""
FLOOR = 1e-6
ABSENT = -1e30
"""The score of a word with no training template: finite, so the ruler counts the row, and below every real one."""
POOLED = "pooled"
SPLIT_HALF = 2


@dataclass
class Scale:
    """Per-channel centring and scaling, from the training windows' own moments."""

    mean: Float[Tensor, " c"]
    std: Float[Tensor, " c"]

    def apply(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        return (x - self.mean[:, None]) / self.std[:, None]

    def means(self, sums: Float[Tensor, "... c t"], counts: Float[Tensor, "..."]) -> Float[Tensor, "... c t"]:
        """Scaled means of `sums` over `counts` windows (zero where a cell holds none)."""
        mean = sums / counts.clamp(min=1)[..., None, None]
        return torch.where(counts[..., None, None] > 0, (mean - self.mean[:, None]) / self.std[:, None], 0.0)


@dataclass(frozen=True)
class Templates:
    """Scaled mean window per (group, word) and the trials behind each; a group is a dataset, or all of them."""

    means: Float[Tensor, "g v c t"]
    counts: Int[Tensor, "g v"]

    def pooled(self) -> "Templates":
        """One group: every dataset's mean weighted by its trials."""
        weight = self.counts.float()[..., None, None]
        total = self.counts.sum(dim=0)
        mean = (self.means * weight).sum(dim=0) / total.clamp(min=1).float()[:, None, None]
        return Templates(mean[None], total[None])

    def distance(self, group: int = 0) -> Float[Tensor, "v v"]:
        """1 - the correlation of every pair of words' flattened templates in `group`."""
        flat = Readout.standardised(self.means[group].flatten(1))
        return 1 - flat @ flat.T


@dataclass
class Sums:
    """Streamed per-(dataset, word) window sums, with the same sums over `reps` random halves of each word's trials,
    and the per-channel moments of every window."""

    sums: Float[Tensor, "g v c t"]
    counts: Float[Tensor, "g v"]
    halves: Float[Tensor, "r g h c t"]
    half_counts: Float[Tensor, "r g h"]
    moments: Float[Tensor, "m c"]
    seen: float
    generator: torch.Generator

    @classmethod
    def empty(cls, groups: int, reps: int, shape: tuple[int, int], device: torch.device, seed: int = 0) -> Self:
        channels, samples = shape
        return cls(
            torch.zeros(groups, VOCAB_SIZE, channels, samples, device=device),
            torch.zeros(groups, VOCAB_SIZE, device=device),
            torch.zeros(reps, groups, SPLIT_HALF * VOCAB_SIZE, channels, samples, device=device),
            torch.zeros(reps, groups, SPLIT_HALF * VOCAB_SIZE, device=device),
            torch.zeros(2, channels, device=device, dtype=torch.float64),
            0.0,
            torch.Generator(device=device).manual_seed(seed),
        )

    def add(self, group: int, x: Float[Tensor, "b c t"], y: Int[Tensor, " b"]) -> None:
        """Count the labelled windows `x` of dataset `group` in."""
        self.sums[group].index_add_(0, y, x)
        self.counts[group] += torch.bincount(y, minlength=VOCAB_SIZE)
        dealt = torch.randint(0, SPLIT_HALF, (len(y), len(self.halves)), generator=self.generator, device=y.device)
        for rep in range(len(self.halves)):
            cell = dealt[:, rep] * VOCAB_SIZE + y
            self.halves[rep, group].index_add_(0, cell, x)
            self.half_counts[rep, group] += torch.bincount(cell, minlength=SPLIT_HALF * VOCAB_SIZE)
        self.moments[0] += x.sum(dim=(0, 2)).double()
        self.moments[1] += x.pow(2).sum(dim=(0, 2)).double()
        self.seen += len(y) * x.shape[2]

    def scale(self) -> Scale:
        mean = self.moments[0] / self.seen
        std = (self.moments[1] / self.seen - mean**2).clamp(min=FLOOR).sqrt()
        return Scale(mean.float(), std.float())

    def templates(self, scale: Scale) -> Templates:
        return Templates(scale.means(self.sums, self.counts), self.counts.long())

    def split(
        self, scale: Scale, group: int | None
    ) -> tuple[Float[Tensor, "r v c t"], Float[Tensor, "r v c t"], Bool[Tensor, "r v"]]:
        """The two half-templates of every word and repeat in `group` (every dataset when None), and where both
        halves hold a trial."""
        halves = self.halves.sum(dim=1) if group is None else self.halves[:, group]
        counts = self.half_counts.sum(dim=1) if group is None else self.half_counts[:, group]
        means = scale.means(halves, counts)
        return means[:, :VOCAB_SIZE], means[:, VOCAB_SIZE:], (counts[:, :VOCAB_SIZE] > 0) & (counts[:, VOCAB_SIZE:] > 0)


@dataclass(frozen=True)
class Reliability:
    """Half-template correlations: each word's with its own other half, and its mean with the other words'."""

    same: Float[Tensor, "r v"]
    diff: Float[Tensor, "r v"]

    @classmethod
    def of(cls, a: Float[Tensor, "r v c t"], b: Float[Tensor, "r v c t"], valid: Bool[Tensor, "r v"]) -> "Reliability":
        """NaN for a word a half of which holds no trial."""
        first, second = Readout.standardised(a.flatten(2)), Readout.standardised(b.flatten(2))
        corr = first @ second.transpose(1, 2)
        pair = valid[:, :, None] & valid[:, None, :] & ~torch.eye(VOCAB_SIZE, dtype=torch.bool, device=a.device)
        others = pair.sum(dim=2)
        diff = (corr * pair).sum(dim=2) / others.clamp(min=1)
        nan = torch.tensor(float("nan"), device=a.device)
        return cls(
            torch.where(valid, corr.diagonal(dim1=1, dim2=2), nan),
            torch.where(valid & (others > 0), diff, nan),
        )

    def spearman_brown(self) -> Float[Tensor, " v"]:
        """Each word's full-template reliability `2r / (1 + r)` from its mean half correlation."""
        half = self.same.nanmean(dim=0)
        return 2 * half / (1 + half)

    def summary(self, counts: Int[Tensor, " v"]) -> str:
        """Mean same / different correlation, the Spearman-Brown reliabilities and the trials per word."""
        brown = self.spearman_brown()
        held = counts[counts > 0].float()
        return (
            f"words={int((~brown.isnan()).sum())} trials/word min={int(held.min())} median={held.median():.0f} "
            f"max={int(held.max())} same={self.same.nanmean():.4f} diff={self.diff.nanmean():.4f} "
            f"SB mean={brown.nanmean():.4f} median={brown.nanmedian():.4f} min={brown[~brown.isnan()].min():.4f} "
            f"max={brown[~brown.isnan()].max():.4f}"
        )


@dataclass(frozen=True)
class Whitener:
    """The inverse square roots of a sensors x time Kronecker noise covariance: `apply` whitens windows by both."""

    spatial: Float[Tensor, "c c"]
    temporal: Float[Tensor, "t t"]

    def apply(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        return torch.einsum("ij,bjt,tk->bik", self.spatial, x, self.temporal)

    def spatial_only(self) -> "Whitener":
        return replace(self, temporal=torch.eye(len(self.temporal), device=self.temporal.device))


@dataclass
class Noise:
    """Streamed second moments of residual windows, read as sensors-by-sensors (samples are time points) and
    time-by-time (samples are sensors), plus each one's Ledoit-Wolf fourth-moment sum."""

    spatial: Float[Tensor, "c c"]
    temporal: Float[Tensor, "t t"]
    quartic: Float[Tensor, " 2"]
    seen: Float[Tensor, " 2"]

    @classmethod
    def empty(cls, shape: tuple[int, int], device: torch.device) -> Self:
        channels, samples = shape
        kwargs = {"device": device, "dtype": torch.float64}
        return cls(
            torch.zeros(channels, channels, **kwargs),
            torch.zeros(samples, samples, **kwargs),
            torch.zeros(2, **kwargs),
            torch.zeros(2, **kwargs),
        )

    def add(self, residual: Float[Tensor, "b c t"]) -> None:
        for at, rows in enumerate((residual.permute(0, 2, 1).flatten(0, 1), residual.flatten(0, 1))):
            rows = rows.double()
            moment = rows.T @ rows
            if at:
                self.temporal += moment
            else:
                self.spatial += moment
            self.quartic[at] += rows.pow(2).sum(dim=1).pow(2).sum()
            self.seen[at] += len(rows)

    @staticmethod
    def shrunk(total: Float[Tensor, "p p"], quartic: Tensor, seen: Tensor) -> Float[Tensor, "p p"]:
        """The sample covariance `total / seen` pulled towards its mean variance by the Ledoit-Wolf weight,
        `b^2 / d^2` with `d^2 = |S - mI|^2` and `b^2 = (sum |x x' - S|^2) / n^2 = (quartic - n |S|^2) / n^2`."""
        cov = total / seen
        width = len(cov)
        level = cov.trace() / width
        spread = cov.pow(2).sum() - width * level**2
        noise = ((quartic - seen * cov.pow(2).sum()) / seen**2).clamp(min=0)
        weight = torch.minimum(noise, spread) / spread.clamp(min=FLOOR)
        return (1 - weight) * cov + weight * level * torch.eye(width, device=cov.device, dtype=cov.dtype)

    @staticmethod
    def inverse_root(cov: Float[Tensor, "p p"]) -> Float[Tensor, "p p"]:
        values, vectors = torch.linalg.eigh(cov)
        return ((vectors * values.clamp(min=FLOOR).rsqrt()) @ vectors.T).float()

    def whitener(self) -> Whitener:
        spatial = self.shrunk(self.spatial, self.quartic[0], self.seen[0])
        temporal = self.shrunk(self.temporal, self.quartic[1], self.seen[1])
        return Whitener(self.inverse_root(spatial), self.inverse_root(temporal))


@dataclass(frozen=True)
class Readout:
    """Templates and windows through one transform, scored by a dot product less a bias: Pearson correlation
    (standardise, bias 0) or the LDA discriminant (whiten, bias half the whitened template's squared norm)."""

    means: Float[Tensor, "g v d"]
    counts: Int[Tensor, "g v"]
    bias: Float[Tensor, "g v"]
    transform: Callable[[Float[Tensor, "b c t"]], Float[Tensor, "b d"]]

    @staticmethod
    def standardised(flat: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return nn.functional.normalize(flat - flat.mean(dim=-1, keepdim=True), dim=-1)

    @classmethod
    def correlation(cls, templates: Templates) -> "Readout":
        def transform(x: Float[Tensor, "b c t"]) -> Float[Tensor, "b d"]:
            return cls.standardised(x.flatten(1))

        return cls.of(templates, transform, discriminant=False)

    @classmethod
    def discriminant(cls, templates: Templates, whitener: Whitener) -> "Readout":
        def transform(x: Float[Tensor, "b c t"]) -> Float[Tensor, "b d"]:
            return whitener.apply(x).flatten(1)

        return cls.of(templates, transform, discriminant=True)

    @classmethod
    def of(cls, templates: Templates, transform: Callable[[Tensor], Tensor], *, discriminant: bool) -> "Readout":
        groups, words = templates.counts.shape
        means = transform(templates.means.flatten(0, 1)).view(groups, words, -1)
        bias = 0.5 * means.pow(2).sum(dim=-1) if discriminant else torch.zeros_like(templates.counts).float()
        return cls(means, templates.counts, bias, transform)

    def scores(self, x: Float[Tensor, "b c t"], group: int) -> Float[Tensor, "b v"]:
        """Every word's score per window of `x` from `group`'s templates; `ABSENT` where it has none."""
        scored = self.transform(x) @ self.means[group].T - self.bias[group]
        return scored.masked_fill(self.counts[group] == 0, ABSENT)


@dataclass
class Durations:
    """Streamed per-word sums of log duration (onset gap to the next word of the sentence)."""

    total: Float[Tensor, " v"] = field(default_factory=lambda: torch.zeros(VOCAB_SIZE, dtype=torch.float64))
    counts: Float[Tensor, " v"] = field(default_factory=lambda: torch.zeros(VOCAB_SIZE, dtype=torch.float64))

    def add(self, labels: Int[Tensor, " n"], log_gap: Float[Tensor, " n"]) -> None:
        """Count in the words whose gap is known (a sentence's last word has none)."""
        known = ~log_gap.isnan()
        self.total.index_add_(0, labels[known], log_gap[known].double())
        self.counts += torch.bincount(labels[known], minlength=VOCAB_SIZE)

    def mean(self) -> Float[Tensor, " v"]:
        return (self.total / self.counts.clamp(min=1)).float()


@dataclass(frozen=True)
class Rsa:
    """A word-by-word distance matrix compared with another by Spearman correlation of their upper triangles."""

    permutations: int = PERMUTATIONS
    seed: int = 0

    @staticmethod
    def gap(value: Float[Tensor, " v"]) -> Float[Tensor, "v v"]:
        """Absolute difference of a per-word scalar."""
        return (value[:, None] - value[None, :]).abs()

    @staticmethod
    def phonetic(pronunciation: Pronunciation) -> Float[Tensor, "v v"]:
        """Positions of the first phonemes at which two vocabulary words differ."""
        spelled = pronunciation.phonemes(torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))))
        return (spelled[:, None, :] != spelled[None, :, :]).sum(dim=-1).float()

    @staticmethod
    def semantic(vectors: Float[Tensor, "v d"]) -> Float[Tensor, "v v"]:
        """Cosine distance of word vectors."""
        unit = nn.functional.normalize(vectors, dim=-1)
        return 1 - unit @ unit.T

    @staticmethod
    def ranks(values: Float[Tensor, " n"]) -> Float[Tensor, " n"]:
        return torch.from_numpy(pd.Series(values.numpy()).rank().to_numpy(dtype=np.float32))

    def correlate(
        self, model: Float[Tensor, "v v"], target: Float[Tensor, "v v"], words: Bool[Tensor, " v"]
    ) -> tuple[float, float, int]:
        """Spearman r of the two matrices over pairs of `words`, its two-sided permutation p over word relabellings
        of the model matrix, and the pairs."""
        kept = torch.nonzero(words).flatten().cpu()
        rows, cols = torch.triu_indices(len(kept), len(kept), offset=1)
        sub = model.cpu()[kept][:, kept]
        order = self.ranks(sub[rows, cols])
        rank = torch.zeros(len(kept), len(kept))
        rank[rows, cols] = order
        rank = rank + rank.T
        goal = self.ranks(target.cpu()[kept][:, kept][rows, cols])
        goal = nn.functional.normalize(goal - goal.mean(), dim=0)

        def pearson(vals: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
            return nn.functional.normalize(vals - vals.mean(dim=-1, keepdim=True), dim=-1) @ goal

        found = pearson(order)
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.rand(self.permutations, len(kept), generator=generator).argsort(dim=1)
        null = pearson(rank[perm[:, rows], perm[:, cols]])
        extreme = int((null.abs() >= found.abs()).sum())
        return float(found), (1 + extreme) / (1 + self.permutations), len(rows)


@dataclass(frozen=True)
class Experiment:
    """The study over the default Deep `DataConfig`'s training runs and official validation / test domains."""

    root: DataRoot
    device: torch.device
    reps: int = REPS
    seed: int = 0

    def reader(self, partition: Partition = Partition.VALIDATION) -> Reader:
        return Reader(self.root, DataConfig(), self.device, partition=partition)

    @staticmethod
    def chunks(reader: Reader, runs: list[Run]) -> Iterator[tuple[str, pd.DataFrame, Split]]:
        """Each run's dataset, words and split, `CHUNK` runs on the device at a time."""
        for start in range(0, len(runs), CHUNK):
            for run, (words, split) in reader.splits(runs[start : start + CHUNK]).items():
                yield DataConfig.stratum(run.key[0]), words, split

    @staticmethod
    def trials(split: Split) -> Iterator[tuple[Float[Tensor, "b c t"], Int[Tensor, " b"]]]:
        """The labelled words of `split` as batches of windows and labels."""
        for part in split.labelled().split(BATCH):
            yield split.windows(part).float(), split.y[part]

    def first(self, reader: Reader, runs: list[Run], datasets: list[str]) -> tuple[Sums, Durations]:
        """Class sums, split-half sums and channel moments of the training runs, and each word's durations."""
        sums, durations = None, Durations()
        for dataset, words, split in self.chunks(reader, runs):
            labelled = split.labelled()
            durations.add(split.y[labelled].cpu(), torch.from_numpy(Readings.targets(words)[0])[labelled.cpu()].float())
            for x, y in self.trials(split):
                sums = sums or Sums.empty(len(datasets), self.reps, (x.shape[1], x.shape[2]), self.device, self.seed)
                sums.add(datasets.index(dataset), x, y)
        if sums is None:
            raise ValueError("no training runs")
        return sums, durations

    def second(
        self, reader: Reader, runs: list[Run], datasets: list[str], scale: Scale, templates: Templates
    ) -> tuple[Noise, Noise]:
        """Noise of the residuals about the pooled and the dataset templates."""
        pooled = templates.pooled()
        found: tuple[Noise, Noise] | None = None
        for dataset, _, split in self.chunks(reader, runs):
            for x, y in self.trials(split):
                x = scale.apply(x)
                found = found or (
                    Noise.empty((x.shape[1], x.shape[2]), self.device),
                    Noise.empty((x.shape[1], x.shape[2]), self.device),
                )
                found[0].add(x - pooled.means[0][y])
                found[1].add(x - templates.means[datasets.index(dataset)][y])
        if found is None:
            raise ValueError("no training runs")
        return found

    @staticmethod
    def readouts(templates: Templates, noises: tuple[Noise, Noise]) -> dict[str, tuple[Readout, bool]]:
        """Every reading by name, and whether it takes each domain's own dataset's templates."""
        pooled = templates.pooled()
        out: dict[str, tuple[Readout, bool]] = {
            "corr/pooled": (Readout.correlation(pooled), False),
            "corr/dataset": (Readout.correlation(templates), True),
        }
        for name, source, noise in (("pooled", pooled, noises[0]), ("dataset", templates, noises[1])):
            white = noise.whitener()
            out[f"lda-spatial/{name}"] = (Readout.discriminant(source, white.spatial_only()), name == "dataset")
            out[f"lda-kron/{name}"] = (Readout.discriminant(source, white), name == "dataset")
        return out

    def decode(
        self,
        partition: Partition,
        datasets: list[str],
        scale: Scale,
        readouts: dict[str, tuple[Readout, bool]],
    ) -> list[str]:
        """Top-10 balanced accuracy of every reading on each domain of `partition` and on all of them together."""
        reader = self.reader(partition)
        domains, _ = reader.domains(reader.ruled(Run.discover(self.root.raw())))
        labels: dict[str, Tensor] = {}
        scored: dict[str, dict[str, Tensor]] = {}
        for domain, split in domains.items():
            group = datasets.index(DataConfig.stratum(domain))
            labels[domain] = split.y[split.labelled()].cpu()
            scored[domain] = {
                name: self.scores(split, scale, read, group if own else 0) for name, (read, own) in readouts.items()
            }
        lines = []
        for name in readouts:
            row = {d: self.balanced(scored[d][name], labels[d]) for d in domains}
            pooled = self.balanced(torch.cat([scored[d][name] for d in domains]), torch.cat(list(labels.values())))
            parts = " ".join(f"{d}={v:.4f}(n={len(labels[d])})" for d, v in row.items())
            lines.append(f"{partition} decode {name} pooled={pooled:.4f}(n={sum(map(len, labels.values()))}) {parts}")
        return lines

    def scores(self, split: Split, scale: Scale, readout: Readout, group: int) -> Float[Tensor, "n v"]:
        return torch.cat([readout.scores(scale.apply(x), group).cpu() for x, _ in self.trials(split)])

    @staticmethod
    def balanced(scores: Float[Tensor, "n v"], labels: Int[Tensor, " n"]) -> float:
        return Ruler().balanced(Ruler().hits(scores, labels), labels)

    def reliability(self, sums: Sums, scale: Scale, datasets: list[str]) -> list[str]:
        """Split-half reliability lines pooled and per dataset."""
        lines = []
        for group, name in [(None, POOLED), *enumerate(datasets)]:
            chosen = sums.counts.sum(dim=0) if group is None else sums.counts[group]
            if not (chosen > 1).any():
                continue
            summary = Reliability.of(*sums.split(scale, group)).summary(chosen.long())
            lines.append(f"reliability {name if group is None else datasets[group]} {summary}")
        return lines

    def content(self, templates: Templates, durations: Durations, datasets: list[str]) -> list[str]:
        """RSA of the template distances, pooled and per dataset, against duration, phonemes and t5 prototypes."""
        rsa = Rsa(PERMUTATIONS, self.seed)
        targets = {
            "duration": Rsa.gap(durations.mean()),
            "phonemes": Rsa.phonetic(Pronunciation.load(Pronunciation.path(self.root))),
            "t5": Rsa.semantic(Prototypes.load(Prototypes.path(self.root)).vectors),
        }
        seen = durations.counts > 0
        lines = []
        for group, name in [(None, POOLED), *enumerate(datasets)]:
            source = templates.pooled() if group is None else templates
            at = 0 if group is None else group
            words = (source.counts[at] > 0) & seen.to(source.counts.device)
            model = source.distance(at)
            for target, matrix in targets.items():
                r, p, pairs = rsa.correlate(model, matrix, words.cpu())
                lines.append(
                    f"rsa {name if group is None else datasets[group]} vs {target} r={r:.4f} p={p:.4f} pairs={pairs}"
                )
        return lines

    def run(self) -> Iterator[str]:
        """Every result line, as it is known."""
        reader = self.reader()
        runs = [r for r in reader.data.trained(Run.everywhere(self.root))]
        datasets = sorted({DataConfig.stratum(r.key[0]) for r in runs})
        sums, durations = self.first(reader, runs, datasets)
        scale = sums.scale()
        templates = sums.templates(scale)
        yield f"train runs={len(runs)} datasets={datasets} labelled words={int(sums.counts.sum())}"
        yield from self.reliability(sums, scale, datasets)
        yield from self.content(templates, durations, datasets)
        readouts = self.readouts(templates, self.second(reader, runs, datasets, scale, templates))
        for partition in (Partition.VALIDATION, Partition.TEST):
            yield from self.decode(partition, datasets, scale, readouts)


class Study:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--reps", type=int, default=REPS)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        for line in Experiment(DataRoot.from_config(), torch.device("cuda"), args.reps, args.seed).run():
            LOG.info("%s", line)


if __name__ == "__main__":
    Study.main()
