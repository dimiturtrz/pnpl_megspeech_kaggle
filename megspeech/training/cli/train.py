"""Train the sentence-context word classifier on the official train split, select on val, predict the Kaggle test.

uv run python -m megspeech.training.cli.train --name context-v0
"""

import argparse
import json
import logging
from dataclasses import asdict

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.cache.kaggle import Track
from megspeech.data.corpus import Corpus
from megspeech.device import Device
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.submission.report import Report
from megspeech.submission.writer import KaggleWriter
from megspeech.training.config import METRICS, RunConfig
from megspeech.training.loop.trainer import Trainer

LOG = logging.getLogger(__name__)

HEADS = "heads.pt"


def main() -> None:
    """Train, write metrics + checkpoint + Kaggle test submission under models/pnpl_libribrain/<name>."""
    origin = argparse.ArgumentParser(add_help=False)
    origin.add_argument("--origin", help="a run whose recorded config the flags start from (default: the warm run)")
    warm = argparse.ArgumentParser(add_help=False, parents=[origin])
    warm.add_argument("--optim.warm", dest="warm", default="")
    known, _ = warm.parse_known_args()
    start = known.origin or known.warm
    saved = RunConfig.saved(DataRoot.from_config().models() / start) if start else None
    tree = Tree(RunConfig, saved)
    parser = argparse.ArgumentParser(description=__doc__, parents=[origin])
    parser.add_argument("--name", required=True)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    corpus = Corpus.load(config.data, root, Device.CUDA.as_torch())
    train, validation, log_prior = corpus.train, corpus.validation, corpus.train.log_prior()
    trainer = Trainer.of(corpus, config, root)
    model, history = trainer.fit(train, validation, log_prior)
    out = root.models() / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    if trainer.heads:
        torch.save(trainer.heads, out / HEADS)
    config.context.save(out)
    best = config.selected(history)
    summary = {
        "config": asdict(config),
        "train": len(train.y),
        "best": asdict(best),
        "stopped": len(history) < config.optim.epochs,
    }
    (out / METRICS).write_text(json.dumps({**summary, "history": [asdict(row) for row in history]}, indent=1))
    LOG.info("best %s", best)
    evaluator = trainer.evaluator
    evaluator.graded(model, validation, log_prior, best.tau).save(out / "hits.pt")
    if config.telemetry.probs:
        Probabilities.of(evaluator, model, validation, log_prior, best.tau).save(out / PROBS, "val")
    loaded = ExportedModel(
        root,
        config.data,
        evaluator.device,
        geometry=corpus.geometry,
        model=model,
        evaluator=evaluator,
        log_prior=log_prior,
    )
    if not config.data.adapt:
        KaggleWriter(Track.DEEP).write(loaded, best.tau, out / "submission.csv")
    Report(config.report, root, args.name, evaluator.device, corpus.geometry, config.telemetry.probs).run()


if __name__ == "__main__":
    main()
