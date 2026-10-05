"""Whether the loss is signal or model: an ensemble's predictive entropy split into the aleatoric part (what each member
still finds uncertain, which no agreement among members removes) and the epistemic part (the members' disagreement, the
mutual information between word and member), per scored word of a labelled partition.

Total = H[mean_m p_m], aleatoric = mean_m H[p_m], epistemic = total - aleatoric (Depeweg et al. 2018, "Decomposition of
Uncertainty in Bayesian Deep Learning"; the deep-ensemble predictive of Lakshminarayanan et al. 2017). Members differing
only by seed share data, architecture and recipe, so their disagreement UNDERESTIMATES the epistemic part: a bias every
seed shares is invisible to it, and the aleatoric share is an upper bound.

    uv run python -m megspeech.evaluation.studies.uncertainty ens-ema16 --partitions validation test
"""

import argparse
import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.metric.ruler import Ruler
from megspeech.evaluation.scoring.score import Head
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

TAU = 1.0
"""Logit adjustment a hit is read at: the balanced-metric Bayes rule (Menon et al. 2021)."""
POOLED = "pooled"
CAVEAT = "seed ensemble: members share data and recipe, so epistemic is underestimated and aleatoric an upper bound"


@dataclass(frozen=True)
class Decomposition:
    """Each word's predictive entropy in nats: `total`, its `aleatoric` share and the `epistemic` remainder."""

    total: Float[Tensor, " n"]
    aleatoric: Float[Tensor, " n"]
    epistemic: Float[Tensor, " n"]

    @staticmethod
    def entropy(probabilities: Float[Tensor, "... v"]) -> Float[Tensor, "..."]:
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

    @classmethod
    def of(cls, log_probabilities: Float[Tensor, "m n v"]) -> "Decomposition":
        """The decomposition of `m` members' log-probabilities over `n` words."""
        probabilities = log_probabilities.exp()
        total = cls.entropy(probabilities.mean(dim=0))
        aleatoric = cls.entropy(probabilities).mean(dim=0)
        return cls(total, aleatoric, total - aleatoric)


@dataclass(frozen=True)
class Members:
    """A loaded ensemble read member by member on the scored words of one split."""

    loaded: LoadedModel

    @property
    def networks(self) -> list[nn.Module]:
        return list(cast("Ensemble", self.loaded.model).members)

    def logits(self, split: Split) -> Float[Tensor, "m n v"]:
        """Every member's context-head log-probabilities for every word of `split`."""
        evaluator = self.loaded.evaluator
        return torch.stack([evaluator.logits(net, split)[Head.CONTEXT] for net in self.networks])

    def hits(self, logits: Float[Tensor, "m n v"], labels: Int[Tensor, " n"]) -> Bool[Tensor, "e n"]:
        """Top-k hits at `TAU` of each member, then of the ensemble (the mean of its members' log-probabilities)."""
        evaluator, prior = self.loaded.evaluator, self.loaded.log_prior
        rows = [*logits, logits.mean(dim=0)]
        return torch.stack([Ruler().hits(evaluator.adjusted(row, prior, TAU), labels) for row in rows])

    def frame(self, split: Split, domain: str) -> tuple[pd.DataFrame, Bool[Tensor, "e n"]]:
        """The scored (unseen-span) words of `split`: `domain`, `label`, `hit` of the ensemble and the three entropies;
        with every member's and then the ensemble's hits."""
        novel = split.novel(self.loaded.evaluator.seen)
        logits, labels = self.logits(split)[:, novel], split.y[novel]
        hits = self.hits(logits, labels)
        parts = Decomposition.of(logits)
        frame = pd.DataFrame(
            {
                "domain": domain,
                "label": labels.cpu().numpy(),
                "hit": hits[-1].cpu().numpy(),
                "total": parts.total.cpu().numpy(),
                "aleatoric": parts.aleatoric.cpu().numpy(),
                "epistemic": parts.epistemic.cpu().numpy(),
            }
        )
        return frame, hits.cpu()

    def partition(self, runs: list[Run]) -> tuple[pd.DataFrame, dict[str, Bool[Tensor, "e n"]]]:
        """Every counted domain of the scored partition of `runs`: the words' frame and each domain's hits."""
        domains, _ = self.loaded.domains(self.loaded.ruled(runs))
        evaluator = self.loaded.evaluator
        counted = {name: s for name, s in domains.items() if len(s.novel(evaluator.seen)) >= evaluator.min_novel}
        read = {name: self.frame(split, name) for name, split in counted.items()}
        return pd.concat([f for f, _ in read.values()], ignore_index=True), {name: h for name, (_, h) in read.items()}


@dataclass(frozen=True)
class Tally:
    """Scored words (`Members.partition`'s frame) and each domain's hits, read as table lines."""

    frame: pd.DataFrame
    hits: dict[str, Bool[Tensor, "e n"]]

    def entropies(self, rows: pd.DataFrame) -> str:
        """Count and mean entropies of `rows`."""
        means = rows[["total", "aleatoric", "epistemic"]].mean()
        share = means["epistemic"] / means["total"]
        return (
            f"n={len(rows)} total={means['total']:.3f} aleatoric={means['aleatoric']:.3f} "
            f"epistemic={means['epistemic']:.3f} epistemic/total={share:.3f}"
        )

    def accuracy(self, name: str) -> str:
        """BAcc@10 of the members (mean, min, max) and of the ensemble on domain `name`'s words."""
        labels = torch.tensor(self.frame.loc[self.frame["domain"] == name, "label"].to_numpy())
        scores = [Ruler().balanced(row, labels) for row in self.hits[name]]
        members = np.array(scores[:-1])
        return (
            f"members bacc10 mean={members.mean():.4f} min={members.min():.4f} max={members.max():.4f} "
            f"ensemble={scores[-1]:.4f} members={len(members)} n={len(labels)}"
        )

    def lines(self) -> list[str]:
        """Entropies of every domain and pooled, overall and for hit / missed words, then the accuracies."""
        pooled = self.frame.assign(domain=POOLED)
        lines = [CAVEAT]
        for name, rows in pd.concat([self.frame, pooled]).groupby("domain", sort=False):
            lines.append(f"{name} all: {self.entropies(rows)}")
            lines += [
                f"{name} {'correct' if hit else 'wrong'}: {self.entropies(part)}"
                for hit, part in rows.groupby("hit")
                if len(part)
            ]
        return lines + [f"{name} {self.accuracy(name)}" for name in self.hits]


class Study:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/")
        parser.add_argument("--partitions", nargs="+", default=[Partition.VALIDATION, Partition.TEST])
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, device = DataRoot.from_config(), torch.device("cuda")
        runs = Run.discover(root.raw())
        for partition in map(Partition, args.partitions):
            loaded = Loader.of(root, [args.name], Scoring(partition=partition), device)
            frame, hits = Members(loaded).partition(runs)
            for line in Tally(frame, hits).lines():
                LOG.info("%s %s", partition, line)


if __name__ == "__main__":
    Study.main()
