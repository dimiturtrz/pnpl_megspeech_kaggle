"""Average trained context models in log-probability space; score per-domain validation, predict the holdout.

uv run python -m megspeech.submission.ensemble --out ensemble-v1v2 context-v1 context-v2
"""

import argparse
import json
import logging
from dataclasses import asdict

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.epochs import Epocher
from megspeech.data.runs import Partition, Run
from megspeech.data.split import Split
from megspeech.evaluation.report import Evaluator
from megspeech.model.ensemble import Ensemble
from megspeech.submission.submission import Submitter

LOG = logging.getLogger(__name__)


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
    labelled = [epocher.labelled(r) for r in runs if r.partition is Partition.TRAIN]
    labels = np.concatenate([y for y, _ in labelled])
    log_prior = Split.prior(torch.from_numpy(labels), torch.ones(len(labels))).to(device)
    evaluator = Evaluator(device, Split.seen(labelled))
    report = evaluator.report(model, Split.validation(epocher, runs, device), log_prior, 0, 0.0)
    LOG.info("%s", {task: (round(r.context.bacc10, 4), round(r.novel.bacc10, 4)) for task, r in report.domains.items()})
    LOG.info("macro %.4f tau %s over %d words", report.macro.bacc10, report.macro.tau, report.macro.n)
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"members": args.names, "best": asdict(report)}, indent=1))
    Submitter(evaluator).submit(model, root, log_prior, report.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
