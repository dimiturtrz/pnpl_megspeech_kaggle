"""Score trained models averaged as one on a track's local ruler, and write its holdout submission at the adjustment
that ruler selects, under models/<out>.

    uv run python -m megspeech.submission sent40-s0 sent40-s1 --out ensemble [--track broad] [--shifts 0 2]
"""

import argparse
import json
import logging
from dataclasses import asdict

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.cache.holdout import Track
from megspeech.data.config import METRICS
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel, Scoring
from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.rulers.deep import DeepRuler
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.submission.writer import HoldoutWriter

LOG = logging.getLogger(__name__)

RULERS: dict[Track, TrackRuler] = {Track.DEEP: DeepRuler(), Track.BROAD: BroadRuler()}
SUBMISSIONS = {Track.DEEP: "submission.csv", Track.BROAD: "broad_submission.csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--out", required=True)
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = LoadedModel.of(root, args.names, tree.parse(args), torch.device("cuda"))
    selected = RULERS[args.track].select(loaded, Run.discover(root.raw()))
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    record = {"members": list(loaded.names), "config": {"data": asdict(loaded.data)}, "best": asdict(selected)}
    (out / METRICS).write_text(json.dumps(record, indent=1))
    LOG.info("wrote %s", HoldoutWriter(args.track).write(loaded, selected.tau, out / SUBMISSIONS[args.track]))


if __name__ == "__main__":
    main()
