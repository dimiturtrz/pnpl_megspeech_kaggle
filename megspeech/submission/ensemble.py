"""Average trained context models in log-probability space; score per-domain validation, predict the holdout.

uv run python -m megspeech.submission.ensemble --out ensemble-v1v2 context-v1 context-v2
"""

import argparse
import json
import logging
from dataclasses import asdict

import torch

from core.paths import DataRoot
from megspeech.data.continuous import DIRECTORY, Continuous
from megspeech.data.corpus import METRICS, Corpus, DataConfig
from megspeech.data.grouping import Grouping
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
    parser.add_argument("--grouping", type=Grouping, choices=list(Grouping), default=DataConfig().grouping)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda")
    root = DataRoot.from_config()
    cache, runs = root.processed() / DIRECTORY, Run.discover(root.raw())
    model = Ensemble.load([root.models() / name / "model.pt" for name in args.names]).to(device)
    heard = Corpus.heard(cache, runs, args.grouping)
    labels = heard["target"].to_numpy()
    log_prior = Split.prior(torch.tensor(labels), torch.ones(len(labels))).to(device)
    evaluator = Evaluator(device, Corpus.sequences(heard, args.grouping))
    held = [r for r in runs if r.partition is Partition.VALIDATION]
    data = DataConfig(
        grouping=args.grouping, window=DataConfig.window_of([root.models() / name for name in args.names])
    )
    source = Continuous.load(cache, device, [r.name for r in held])
    validation = Corpus.domains(source, Corpus.words(source.words, args.grouping), held, data)
    report = evaluator.report(model, validation, log_prior, 0, 0.0)
    LOG.info("%s", {task: (round(r.context.bacc10, 4), round(r.novel.bacc10, 4)) for task, r in report.domains.items()})
    LOG.info("macro %.4f tau %s over %d words", report.macro.bacc10, report.macro.tau, report.macro.n)
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    record = {"members": args.names, "config": {"data": asdict(data)}, "best": asdict(report)}
    (out / METRICS).write_text(json.dumps(record, indent=1))
    Submitter(evaluator, data.samples).submit(model, root, log_prior, report.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
