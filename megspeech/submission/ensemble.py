"""Average trained context models in log-probability space; score per-domain validation, predict the holdout.

uv run python -m megspeech.submission.ensemble --out ensemble-v1v2 context-v1 context-v2
"""

import argparse
import json
import logging
from dataclasses import asdict

import torch

from core.paths import DataRoot
from megspeech.data.corpus import METRICS, DataConfig
from megspeech.data.grouping import Grouping
from megspeech.data.runs import Run
from megspeech.evaluation.balance import BALANCES
from megspeech.evaluation.pooling import Pooling, Scoring
from megspeech.submission.submission import Submitter

LOG = logging.getLogger(__name__)


def main() -> None:
    """Score the ensemble on each validation task and write its holdout submission under models/<out>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--grouping", type=Grouping, choices=list(Grouping), default=DataConfig().grouping)
    parser.add_argument("--balance", choices=list(BALANCES), default="none")
    parser.add_argument("--shifts", type=int, nargs="+", default=[0], help="samples each window is also read moved by")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    scoring = Scoring(args.grouping, torch.device("cuda"), balance=BALANCES[args.balance], shifts=tuple(args.shifts))
    pooling = Pooling.loaded(root, [root.models() / name for name in args.names], scoring)
    model, data, evaluator, log_prior = pooling.model, pooling.data, pooling.evaluator, pooling.log_prior
    validation, _ = pooling.domains(data.held(Run.discover(root.raw())))
    report = evaluator.report(model, validation, log_prior, 0, 0.0)
    LOG.info("%s", {task: (round(r.context.bacc10, 4), round(r.novel.bacc10, 4)) for task, r in report.domains.items()})
    LOG.info("macro %.4f tau %s over %d words", report.macro.bacc10, report.macro.tau, report.macro.n)
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    record = {"members": args.names, "config": {"data": asdict(data)}, "best": asdict(report)}
    (out / METRICS).write_text(json.dumps(record, indent=1))
    Submitter(evaluator, data).submit(model, root, log_prior, report.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
