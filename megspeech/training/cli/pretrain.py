"""Pretrain subject 0's word encoder on its unlabelled training MEG, to start a supervised run from (`--optim.stem`).

uv run python -m megspeech.training.cli.pretrain --name ssl-v0 --origin sent40-ioi-ema9998-s0
"""

import argparse
import logging

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.training.config import RunConfig
from megspeech.training.pretrain.pretrain import PretrainConfig, Pretrainer


def main() -> None:
    """Pretrain and write model.pt, predictor.pt, the context config and metrics under models/<name>."""
    origin = argparse.ArgumentParser(add_help=False)
    origin.add_argument("--origin", help="a run whose recorded data and context config the flags start from")
    known, _ = origin.parse_known_args()
    root = DataRoot.from_config()
    saved = RunConfig.saved(root.models() / known.origin) if known.origin else None
    start = PretrainConfig(data=saved.data, context=saved.context) if saved else None
    tree = Tree(PretrainConfig, start)
    parser = argparse.ArgumentParser(description=__doc__, parents=[origin])
    parser.add_argument("--name", required=True)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    trainer = Pretrainer.of(config, root, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net, predictor, history = trainer.run()
    trainer.save(root.models() / args.name, net, predictor, history)


if __name__ == "__main__":
    main()
