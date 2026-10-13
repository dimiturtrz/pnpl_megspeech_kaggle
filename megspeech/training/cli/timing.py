"""Train a timing expert alone (no MEG) on the official train split, for `--optim.timing` to freeze beside a MEG model.

uv run python -m megspeech.training.cli.timing --origin sent40-ioi-ema9998-s0 --name timing-s0
"""

import argparse
import json
import logging
from dataclasses import asdict, replace

from core.paths import DataRoot
from core.tree import Tree
from megspeech.device import Device
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.training.config import METRICS, RunConfig
from megspeech.training.loop.trainer import Trainer
from megspeech.training.timing_corpus import TimingCorpus
from megspeech.training.timing_expert import TimingExpert

LOG = logging.getLogger(__name__)


def main() -> None:
    """Fit and save the timing head under models/<name>, with the data the origin run read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="a run whose recorded data config the expert reads")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", type=Device, default=Device.CUDA)
    parser.add_argument("--share", type=float, default=1.0, help="data.train_share: the timing learning curve")
    tree = Tree(TimingExpert)
    tree.arguments(parser)
    args = parser.parse_args()
    expert = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    saved = RunConfig.saved(root.models() / args.origin)
    config = replace(saved, data=replace(saved.data, train_share=args.share))
    corpus = Trainer.corpus(config, root, args.device.as_torch())
    evaluator = Evaluator(corpus.train.y.device, corpus.seen, config.selection.min_novel)
    pretrain = TimingCorpus.read(root, expert.pretrain) if expert.pretrain else None
    if pretrain is not None:
        LOG.info("pretrain corpus %s: %s", expert.pretrain, pretrain.describe())
    model, history = expert.fit(corpus.train, corpus.validation, evaluator, corpus.train.log_prior(), pretrain)
    out = root.models() / args.name
    TimingExpert.save(model.timing, out)
    best = max(history, key=lambda row: row.macro.bacc10)
    summary = {"config": asdict(config), "expert": asdict(expert), "best": asdict(best)}
    (out / METRICS).write_text(json.dumps(summary, indent=1))
    LOG.info("best epoch %d macro %.4f", best.epoch, best.macro.bacc10)


if __name__ == "__main__":
    main()
