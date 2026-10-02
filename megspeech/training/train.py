"""Train the sentence-context word classifier on the official train split, select on validation, predict the holdout.

uv run python -m megspeech.training.train --name context-v0
"""

import argparse
import json
import logging
from dataclasses import asdict

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.corpus import Corpus
from megspeech.submission.submission import Submitter
from megspeech.training.config import METRICS, RunConfig
from megspeech.training.trainer import Trainer

LOG = logging.getLogger(__name__)


def main() -> None:
    """Train, write metrics + checkpoint + holdout submission under models/pnpl_libribrain/<name>."""
    origin = argparse.ArgumentParser(add_help=False)
    origin.add_argument("--origin", help="a run whose recorded config the flags start from")
    known, _ = origin.parse_known_args()
    saved = RunConfig.saved(DataRoot.from_config().models() / known.origin) if known.origin else None
    tree = Tree(RunConfig, saved)
    parser = argparse.ArgumentParser(description=__doc__, parents=[origin])
    parser.add_argument("--name", required=True)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    corpus = Corpus.load(config.data, root, torch.device("cuda"))
    train, validation, log_prior = corpus.train, corpus.validation, corpus.train.log_prior()
    trainer = Trainer.of(corpus, config, root)
    model, history = trainer.fit(train, validation, log_prior)
    out = root.models() / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    config.context.save(out)
    best = config.selected(history)
    summary = {"config": asdict(config), "train": len(train.y), "best": asdict(best)}
    (out / METRICS).write_text(json.dumps({**summary, "history": [asdict(row) for row in history]}, indent=1))
    LOG.info("best %s", best)
    evaluator = trainer.evaluator
    evaluator.graded(model, validation, log_prior, best.macro.tau).save(out / "hits.pt")
    Submitter(evaluator).submit(model, root, log_prior, best.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
