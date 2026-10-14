"""The neighbour heads read at scoring: a word's context-head score plus what its span neighbours' next- and
previous-word heads say about it.

Word j's fused score is `log_softmax(context[j]) + alpha * (log_softmax(next[j-1]) + log_softmax(previous[j+1]))`, each
neighbour term present only where that neighbour is a word of the span; the heads are those a run trained with
`--optim.next_word` and `--optim.prev_word`, read off each word's local embedding. Scored as `ioi` scores: the tau the
`--partition` report selects, the same domains and macro. `--alpha 0` is the context head alone.

    uv run python -m megspeech.evaluation.studies.fusion.neighbour ema-cd25-nbr03-s1 --grouping kaggle \\
        --partition test --alpha 0 0.25 0.5 1
"""

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.config import METRICS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.readout.ioi import Scored
from megspeech.model.auxiliary import Tap
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.neighbour import NextWordTerm, PreviousWordTerm
from megspeech.objective.auxiliary.order import HeadOrder

LOG = logging.getLogger(__name__)

HEADS = "heads.pt"
ALPHAS = (0.0, 0.25, 0.5, 1.0)
DEEP = Scoring(fallback=Fallback(Unseen.DEEP))


@dataclass(frozen=True)
class Member:
    """One trained model and its two neighbour heads."""

    net: SentenceNet
    after: nn.Module
    before: nn.Module

    @classmethod
    def load(cls, net: SentenceNet, directory: Path, device: torch.device) -> "Member":
        """`net` with the two neighbour heads of `directory`'s saved list, each over the word embedding."""
        weights = Tree.flat(json.loads((directory / METRICS).read_text())["config"]["optim"])
        if not (weights.get("next_word") and weights.get("prev_word")):
            raise ValueError(
                f"{directory.name} has no neighbour heads: train with --optim.next_word and --optim.prev_word"
            )
        saved = torch.load(directory / HEADS, map_location=device, weights_only=True)
        terms = {
            "next_word": NextWordTerm(weight=weights["next_word"]),
            "prev_word": PreviousWordTerm(weight=weights["prev_word"]),
        }
        heads = []
        for name, term in terms.items():
            prefix = f"heads.{HeadOrder.position(weights, name)}."
            head = term.head(Tap.WORD.width(net)).to(device)
            head.load_state_dict({key.removeprefix(prefix): v for key, v in saved.items() if key.startswith(prefix)})
            heads.append(head.eval())
        return cls(net.eval(), *heads)

    def beside(self, slots: Float[Tensor, "g w e"], present: Bool[Tensor, "g w"]) -> Float[Tensor, "g w v"]:
        """At each position, the log-probabilities its previous word's next-head and its next word's previous-head give
        it, zero for a neighbour the span lacks."""
        mask = present[..., None]
        after = self.after(slots).float().log_softmax(-1) * mask
        before = self.before(slots).float().log_softmax(-1) * mask
        return functional.pad(after[:, :-1], (0, 0, 1, 0)) + functional.pad(before[:, 1:], (0, 0, 0, 1))


@dataclass(frozen=True)
class Neighbours:
    """The members of the scored ensemble with their neighbour heads, averaged as the ensemble averages its heads."""

    members: tuple[Member, ...]

    @classmethod
    def of(cls, model: nn.Module, directories: list[Path], device: torch.device) -> "Neighbours":
        if not isinstance(model, Ensemble):
            raise TypeError(f"{type(model).__name__} wraps the ensemble: score without --lowpass")
        nets = [net for net in model.members if isinstance(net, SentenceNet)]
        if len(nets) != len(model.members):
            raise TypeError("every ensemble member must be a SentenceNet to carry neighbour heads")
        return cls(tuple(Member.load(net, d, device) for net, d in zip(nets, directories, strict=True)))

    def scores(self, split: Split) -> Float[Tensor, "n v"]:
        """Every word of `split`'s neighbour log-probabilities, in word order."""
        device = split.index.device
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        out = torch.zeros(words + 1, VOCAB_SIZE, device=device)
        with torch.no_grad(), torch.autocast(Device.CUDA, dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                at = torch.where(batch.index != PAD, batch.index, words)
                reads = [
                    member.beside(member.net.heard(batch.inputs())[1], batch.index != PAD) for member in self.members
                ]
                out[at] = torch.stack(reads).mean(0)
        return out[:words]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("names", nargs="+")
    tree = Tree(Scoring, DEEP)
    tree.arguments(parser)
    args = parser.parse_args()
    scoring = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, scoring, Device.CUDA.as_torch()), args.names)
    neighbours = Neighbours.of(loaded.model, Scoring.members(root.models(), args.names), loaded.evaluator.device)
    runs = Run.discover(root.raw())
    tau = loaded.report(runs).tau
    for partition in (Partition.VALIDATION, Partition.TEST):
        held = replace(loaded, partition=partition)
        domains, _ = held.domains(held.ruled(runs))
        counted = held.evaluator.eligible(domains)
        context = {name: held.evaluator.logits(held.model, split)[Head.CONTEXT] for name, split in counted.items()}
        beside = {name: neighbours.scores(split) for name, split in counted.items()}
        for alpha in args.alpha:
            fused = {name: context[name] + alpha * beside[name] for name in counted}
            graded = held.evaluator.grade(fused, counted, held.calibration(tau))
            LOG.info("%s alpha=%s tau %s: %s", partition, alpha, tau, Scored(graded).score())


if __name__ == "__main__":
    main()
