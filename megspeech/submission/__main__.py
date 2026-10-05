"""Score trained models averaged as one on a track's local ruler, and write its Kaggle test submission at the adjustment
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
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.rulers.deep import DeepRuler
from megspeech.evaluation.rulers.localtest import LocalTest
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.submission.writer import KaggleWriter

LOG = logging.getLogger(__name__)

RULERS: dict[Track, TrackRuler] = {Track.DEEP: DeepRuler(), Track.BROAD: BroadRuler()}
SUBMISSIONS = {Track.DEEP: "submission.csv", Track.BROAD: "broad_submission.csv"}
HITS = {Track.DEEP: "hits.pt", Track.BROAD: "broad_hits.pt"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--out", required=True)
    parser.add_argument("--local", action="store_true", help="report val and official test at val tau; write nothing")
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(root, args.names, tree.parse(args), torch.device("cuda"))
    ruler, runs = RULERS[args.track], Run.discover(root.raw())
    if args.local:
        return LocalTest(loaded, runs).report("+".join(args.names))
    selected = ruler.select(loaded, runs)
    out = root.models() / args.out
    out.mkdir(parents=True, exist_ok=True)
    ruler.graded(loaded, runs, selected.tau).save(out / HITS[args.track])
    record = {"members": list(loaded.names), "config": {"data": asdict(loaded.data)}, "best": asdict(selected)}
    (out / METRICS).write_text(json.dumps(record, indent=1))
    LOG.info("wrote %s", KaggleWriter(args.track).write(loaded, selected.tau, out / SUBMISSIONS[args.track]))


if __name__ == "__main__":
    main()
