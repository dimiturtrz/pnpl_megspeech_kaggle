"""Where a trained model reads its answer: integrated gradients of each scored word's label log-probability on the MEG
windows of its span (Sundararajan et al. 2017, captum), folded to a share per sensor, per time lag after the word's
onset, and per class of the ten most frequent; and feature ablation of what is not MEG (the onsets' intervals, every
onset) against the MEG itself, which sizes what timing alone supplies.

An attribution is gradient times input against a zero window in raw MEG space, summed in magnitude: a share says where
the model's evidence sits, not that the sensor causes the word. Ablation drops are in nats of the label's
log-probability, the model's own, before any logit adjustment.

    uv run python -m megspeech.evaluation.studies.attribution <run>
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from captum.attr import FeatureAblation, IntegratedGradients
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import SpanInput, Split
from megspeech.data.constants import RATE
from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.masc import Neuromag
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.context.timing import KAGGLE_GAP

LOG = logging.getLogger(__name__)

SUMMARY = "attribution.json"
TOPOMAP = "attribution.png"
MEG = 0
ONSET = 1


@dataclass(frozen=True)
class AttributionConfig:
    steps: int = 16
    """Integration steps from the zero window to the real one."""
    words: int = 400
    """Scored words attributed, drawn at random across the domains."""
    seed: int = 0
    classes: int = 10
    """Most frequent classes whose attributions are kept apart."""
    chunk: int = 4
    """Words of one span attributed in one captum call."""


@dataclass(frozen=True)
class Objective:
    """One span's context head, read as each of `positions`' label log-probability: what captum differentiates and
    ablates. `x` and `onset` carry a leading dimension of `n` copies of the span (captum's steps and ablated copies),
    cycling through the targets."""

    model: nn.Module
    slot: Int[Tensor, "w"]
    subject: Int[Tensor, "r"]
    positions: Int[Tensor, "b"]
    labels: Int[Tensor, "b"]

    def __call__(self, x: Float[Tensor, "n r c t"], onset: Float[Tensor, "n w"]) -> Float[Tensor, " n"]:
        n, rows = x.shape[:2]
        which = torch.arange(n, device=x.device)
        slot = torch.where(self.slot < rows, self.slot + which[:, None] * rows, n * rows)
        context, _ = self.model(SpanInput(x.flatten(0, 1), onset, slot, self.subject.repeat(n)))
        target = which % len(self.labels)
        return context.log_softmax(-1)[which, self.positions[target], self.labels[target]]


@dataclass(frozen=True)
class Reads:
    """The sampled words of one span: the model's inputs for it and where each target sits."""

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "w"]
    objective: Objective
    own: Int[Tensor, "b"]
    """Each target's own window, a row of `x`."""

    @classmethod
    def of(cls, model: nn.Module, split: Split, span: int, words: Int[Tensor, "b"]) -> "Reads":
        batch = split.batch(torch.tensor([span], device=split.y.device), split.index.shape[1])
        positions = (batch.index[0, None, :] == words[:, None]).float().argmax(dim=1)
        objective = Objective(model, batch.slot[0], batch.subject, positions, split.y[words])
        return cls(batch.x, batch.onset[0], objective, batch.slot[0][positions])

    def copies(self, count: int) -> tuple[Float[Tensor, "n r c t"], Float[Tensor, "n w"]]:
        return self.x.expand(count, *self.x.shape).contiguous(), self.onset.expand(count, -1).contiguous()


@dataclass(frozen=True)
class Saliency:
    """Each attributed word's magnitude of attribution over its own window, with its class."""

    magnitude: Float[Tensor, "n c t"]
    labels: Int[Tensor, " n"]

    @staticmethod
    def share(values: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        return values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)

    def sensors(self) -> Float[Tensor, " c"]:
        """Each sensor's share of the evidence, summed over words and lags."""
        return self.share(self.magnitude.sum(dim=(0, 2)))

    def lags(self) -> Float[Tensor, " t"]:
        """Each sample after the onset's share of the evidence."""
        return self.share(self.magnitude.sum(dim=(0, 1)))

    def frequent(self, count: int) -> list[int]:
        """The `count` most frequent labels, most first."""
        labels, counts = self.labels.unique(return_counts=True)
        return labels[counts.argsort(descending=True)][:count].tolist()

    def by_class(self, label: int) -> "Saliency":
        return Saliency(self.magnitude[self.labels == label], self.labels[self.labels == label])

    def summary(self, names: list[str], top: int = 5) -> dict[str, object]:
        sensors, lags = self.sensors(), self.lags()
        strongest = sensors.argsort(descending=True)[:top].tolist()
        return {
            "words": len(self.labels),
            "sensor_share": sensors.tolist(),
            "lag_share": lags.tolist(),
            "peak_lag_s": float(lags.argmax()) / RATE,
            "top_sensors": [names[i] for i in strongest],
        }


@dataclass(frozen=True)
class Drops:
    """How far the label's log-probability falls, in nats, mean over words, when an input is removed."""

    full: float
    meg: float
    interval: float
    timing: float

    @property
    def timing_to_meg(self) -> float:
        """Intervals' drop as a share of the MEG's: the part of what MEG earns that timing already holds."""
        return self.interval / self.meg if self.meg else float("nan")

    def summary(self) -> dict[str, float]:
        return {
            "label_logprob": self.full,
            "meg_drop": self.meg,
            "interval_drop": self.interval,
            "timing_drop": self.timing,
            "interval_over_meg": self.timing_to_meg,
        }


@dataclass(frozen=True)
class Attribution:
    """`model` read on the validation `domains`, its words attributed to the MEG windows and ablated by input."""

    model: nn.Module
    domains: dict[str, Split]
    seen: frozenset[tuple[int, ...]]
    config: AttributionConfig = AttributionConfig()

    def sampled(self) -> dict[str, Int[Tensor, " k"]]:
        """The scored words, `config.words` of them at random across the domains in proportion to their size."""
        novel = {name: split.novel(self.seen) for name, split in self.domains.items()}
        total = sum(len(words) for words in novel.values())
        generator = torch.Generator().manual_seed(self.config.seed)
        keep = min(1.0, self.config.words / max(total, 1))
        chosen = {}
        for name, words in novel.items():
            order = torch.randperm(len(words), generator=generator).to(words.device)
            chosen[name] = words[order[: max(1, round(len(words) * keep))]] if len(words) else words
        return chosen

    @staticmethod
    def spans(split: Split, words: Int[Tensor, " k"]) -> dict[int, Int[Tensor, " b"]]:
        """The sampled `words` of each span that reads them, by the span's first read."""
        found = (split.index[:, :, None] == words[None, None, :]).any(dim=1).float()
        first = found.argmax(dim=0)
        return {int(span): words[first == span] for span in first.unique()}

    def reads(self) -> list[Reads]:
        self.model.eval()
        reads: list[Reads] = []
        for name, words in self.sampled().items():
            split = self.domains[name]
            for span, group in self.spans(split, words).items():
                reads += [Reads.of(self.model, split, span, part) for part in group.split(self.config.chunk)]
        return reads

    def integrated(self, read: Reads) -> Float[Tensor, "b c t"]:
        """Each target's magnitude of attribution over its own window."""
        count = len(read.own)
        x, _ = read.copies(count)
        method = IntegratedGradients(lambda window: read.objective(window, read.onset.expand(len(window), -1)))
        attributed = method.attribute(x, baselines=torch.zeros_like(x), n_steps=self.config.steps)
        return attributed[torch.arange(count), read.own].abs().detach()

    def ablated(self, read: Reads, baselines: tuple[Tensor, Tensor]) -> Float[Tensor, " b"]:
        """What each target's log-probability loses when the inputs become `baselines`, the MEG and the onsets ablated
        as two groups."""
        x, onset = read.copies(len(read.own))
        masks = (torch.zeros_like(x, dtype=torch.long), torch.ones_like(onset, dtype=torch.long))
        drop = FeatureAblation(read.objective).attribute((x, onset), baselines, feature_mask=masks)
        return drop[MEG].flatten(1)[:, 0] + drop[ONSET].flatten(1)[:, 0]

    def dropped(self, read: Reads) -> tuple[Float[Tensor, " b"], ...]:
        """Per target: the label's log-probability, then its fall without MEG, without the onsets' intervals (each
        onset evenly spaced from the span's first), without any onset (all zero)."""
        x, onset = read.copies(len(read.own))
        even = onset[:, :1] + torch.arange(onset.shape[1], device=onset.device) * KAGGLE_GAP
        full = read.objective(x, onset).detach()
        meg = self.ablated(read, (torch.zeros_like(x), onset))
        interval = self.ablated(read, (x, even))
        timing = self.ablated(read, (x, torch.zeros_like(onset)))
        return full, meg, interval, timing

    def run(self) -> tuple[Saliency, Drops]:
        reads = self.reads()
        magnitude = torch.cat([self.integrated(read) for read in reads])
        labels = torch.cat([read.objective.labels for read in reads])
        parts = list(zip(*(self.dropped(read) for read in reads), strict=True))
        full, meg, interval, timing = (float(torch.cat(part).mean()) for part in parts)
        return Saliency(magnitude, labels), Drops(full, meg, interval, timing)

    def report(self, names: list[str]) -> dict[str, object]:
        """The study's summary: the overall and per-class saliency, the ablation drops, and the config."""
        saliency, drops = self.run()
        classes = {
            PRIMARY.words[label]: saliency.by_class(label).summary(names)
            for label in saliency.frequent(self.config.classes)
        }
        return {
            "config": vars(self.config) | {"lag_s_per_sample": 1 / RATE},
            "sensors": names,
            "overall": saliency.summary(names),
            "classes": classes,
            "ablation": drops.summary(),
        }


@dataclass(frozen=True)
class Topomap:
    """A sensor share drawn over the helmet's flattened positions."""

    positions: Float[np.ndarray, "c 3"]
    share: Float[np.ndarray, " c"]

    def save(self, path: Path) -> Path:
        figure = Figure(figsize=(5, 5))
        axes = figure.subplots()
        shown = axes.scatter(self.positions[:, 0], self.positions[:, 1], c=self.share, cmap="viridis", s=40)
        axes.set_aspect("equal")
        axes.set_title("share of attribution per sensor")
        figure.colorbar(shown, ax=axes)
        figure.savefig(path, dpi=150)
        return path


class Study:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="a trained run under models/")
        parser.add_argument("--words", type=int, default=AttributionConfig.words)
        parser.add_argument("--steps", type=int, default=AttributionConfig.steps)
        parser.add_argument("--no-png", action="store_true")
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, device = DataRoot.from_config(), Device.CUDA.as_torch()
        loaded: LoadedModel = Loader.of(Group(root, Scoring(partition=Partition.VALIDATION), device), [args.name])
        domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
        config = AttributionConfig(steps=args.steps, words=args.words)
        helmet = Neuromag.of(NEUROMAG.home(root))
        summary = Attribution(loaded.model, domains, loaded.evaluator.seen, config).report(helmet.names)
        out = root.models() / args.name
        (out / SUMMARY).write_text(json.dumps(summary, indent=1))
        LOG.info("ablation %s", summary["ablation"])
        if not args.no_png:
            overall = cast("dict[str, list[float]]", summary["overall"])
            Topomap(helmet.positions, np.array(overall["sensor_share"])).save(out / TOPOMAP)


if __name__ == "__main__":
    Study.main()
