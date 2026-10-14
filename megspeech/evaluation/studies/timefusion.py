"""Whether a timing expert trained alone adds to a MEG model's context head: each scored word's context logits plus a
weight times the expert's log-probabilities from its span's onsets, on a grid of weights, every domain the model holds
out (megspeech-r9m). The expert reads only onsets, which the Kaggle test gives, so a fused submission needs no rules
decision beyond the model's own.

    uv run python -m megspeech.evaluation.studies.timefusion ema-cd25-xsp10sem05-ac05x-s1-ft-s1 --expert zz-tseq-1.0
"""

import argparse
import json
import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import METRICS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import Head
from megspeech.model.context.timing_only import FILE, TimingOnly
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
"""Weights on the expert's log-probability; 0 is the context head alone."""


@dataclass(frozen=True)
class Expert:
    """A trained timing expert, read as every word's log-probability over the vocabulary."""

    model: TimingOnly

    @classmethod
    def load(cls, root: DataRoot, name: str, device: torch.device) -> "Expert":  # devtools-ignore: test-mirror
        layers = json.loads((root.models() / name / METRICS).read_text())["expert"].get("layers", 0)
        model = TimingOnly(VOCAB_SIZE, layers)
        model.timing.load_state_dict(torch.load(root.models() / name / FILE, weights_only=True))
        return cls(model.to(device).eval())

    def words(self, split: Split) -> Float[Tensor, "n v"]:
        """Each of `split`'s words' log-probabilities from its span's onsets; zero for words no span holds."""
        mask = split.index != PAD
        with torch.no_grad():
            logp = self.model.timing(split.onset, mask).float().log_softmax(-1)
        out = torch.zeros(len(split.y), logp.shape[-1], device=logp.device)
        out[split.index[mask]] = logp[mask]
        return out


@dataclass(frozen=True)
class TimeFusion:
    """A loaded model's held-out domains read with and without the timing expert."""

    loaded: LoadedModel
    expert: Expert

    def domain(self, split: Split) -> dict[float, float]:  # devtools-ignore: test-mirror
        """BAcc@10 on `split`'s unseen-span words at every fusion weight."""
        evaluator = self.loaded.evaluator
        context = evaluator.logits(self.loaded.model, split)[Head.CONTEXT]
        timing = self.expert.words(split)
        novel = split.novel(evaluator.seen)
        alone = evaluator.evaluate(timing[novel], split.y[novel], self.loaded.log_prior)
        LOG.info("expert alone %.4f over %d words", alone.bacc10, alone.n)
        return {
            weight: evaluator.evaluate((context + weight * timing)[novel], split.y[novel], self.loaded.log_prior).bacc10
            for weight in WEIGHTS
        }

    def table(self, held: dict[str, list[Run]]) -> dict[str, dict[float, float]]:  # devtools-ignore: test-mirror
        """Every counted domain's scores by fusion weight."""
        domains, _ = self.loaded.domains(held)
        evaluator = self.loaded.evaluator
        counted = {name: s for name, s in domains.items() if len(s.novel(evaluator.seen)) >= evaluator.min_novel}
        return {name: self.domain(split) for name, split in counted.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--expert", required=True, help="a timing expert under models/ (megspeech.training.cli.timing)")
    parser.add_argument("--device", type=Device, default=Device.CUDA)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), args.device.as_torch()
    loaded = Loader.of(Group(root, Scoring(fallback=Fallback(Unseen.DEEP)), device), args.names)
    runs = Run.discover(root.raw())
    fusion = TimeFusion(loaded, Expert.load(root, args.expert, device))
    for part, held in (("val", loaded.data.held(runs)), ("local test", loaded.data.tested(runs))):
        table = fusion.table(held)
        for name, scores in table.items():
            LOG.info("%s %s %s", part, name, {weight: round(score, 4) for weight, score in scores.items()})
        for weight in WEIGHTS:
            LOG.info("%s weight %s macro %.4f", part, weight, sum(s[weight] for s in table.values()) / len(table))


if __name__ == "__main__":
    main()
