"""Average trained context models in log-probability space; score per-domain validation, predict the holdout.

uv run python -m megspeech.ensemble --out ensemble-v1v2 context-v1 context-v2
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import override

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.epochs import Epocher
from megspeech.model import SentenceNet
from megspeech.runs import Partition, Run
from megspeech.scaler import Scaler
from megspeech.train import Split, TrainConfig, Trainer
from megspeech.wordnet import CHANNELS

LOG = logging.getLogger(__name__)


class Ensemble(nn.Module):
    """Members' log-softmax averaged per head: a geometric mean of their predictive distributions."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def load(cls, paths: list[Path]) -> "Ensemble":
        """SentenceNet checkpoints; each state dict overwrites the placeholder scaler's statistics."""
        members: list[nn.Module] = []
        for path in paths:
            model = SentenceNet(Scaler(torch.zeros(1, CHANNELS, 1)))
            model.load_state_dict(torch.load(path, map_location="cpu"))
            members.append(model)
        return cls(members)

    @override
    def forward(
        self, x: Float[Tensor, "n c t"], onset: Float[Tensor, "g w"], mask: Bool[Tensor, "g w"]
    ) -> tuple[Float[Tensor, "g w v"], Float[Tensor, "g w v"]]:
        outputs = [member(x, onset, mask) for member in self.members]
        context = torch.stack([out[0].float().log_softmax(-1) for out in outputs]).mean(0)
        word = torch.stack([out[1].float().log_softmax(-1) for out in outputs]).mean(0)
        return context, word


def main() -> None:
    """Score the ensemble on each validation task and write its holdout submission under models/<out>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda")
    root = DataRoot.from_config()
    epocher = Epocher(root.processed() / "epochs")
    runs = Run.discover(root.raw())
    model = Ensemble.load([root.models() / name / "model.pt" for name in args.names]).to(device)
    labels = np.concatenate([epocher.labels(r) for r in runs if r.partition is Partition.TRAIN])
    log_prior = Split.prior(torch.from_numpy(labels), torch.ones(len(labels))).to(device)
    trainer = Trainer(TrainConfig(), device)
    report = trainer.report(model, Split.validation(epocher, runs, device), log_prior, 0, 0.0)
    LOG.info("%s", {task: round(r.context.bacc10, 4) for task, r in report.domains.items()})
    LOG.info("macro %.4f tau %s over %d words", report.macro.bacc10, report.macro.tau, report.macro.n)
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"members": args.names, "best": asdict(report)}, indent=1))
    trainer.submit(model, root, log_prior, report.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
