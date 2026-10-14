"""Score trained models averaged as one on a track's local ruler, and write its Kaggle test submission at the adjustment
that ruler selects, under models/<out>.

    uv run python -m megspeech.submission sent40-s0 sent40-s1 --out ensemble [--track broad] [--shifts 0 2]
    uv run python -m megspeech.submission --rebuild ensemble --out ensemble-again
"""

import argparse
import logging
from dataclasses import replace

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.cache.kaggle import Track
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.labelweighted import Shrinkage
from megspeech.evaluation.memorisation import Memorisation
from megspeech.evaluation.rulers.localtest import LocalTest
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import ADJUSTMENTS
from megspeech.submission.labelweighted import LabelWeighted
from megspeech.submission.report import SCORING, Recorded, Submission

LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="trained models under models/, averaged as one")
    parser.add_argument(
        "--rebuild",
        metavar="BLEND",
        help="a blend under models/ to rebuild from its record: members, scoring and track; their flags are ignored",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--local", action="store_true", help="report val and local test at val tau; write nothing")
    parser.add_argument(
        "--memorisation", action="store_true", help="with --local, tally test hits by what training heard"
    )
    parser.add_argument("--track", type=Track, choices=list(Track), default=Track.DEEP)
    parser.add_argument("--device", default=Device.CUDA, help="cpu keeps scoring off a GPU a training run owns")
    parser.add_argument(
        "--labelweights",
        nargs=2,
        type=float,
        metavar=("K", "BETA"),
        help="blend members per label (shrinkage K, sharpness BETA) fit on the whole Broad ruler; Broad track only",
    )
    parser.add_argument(
        "--tau", type=float, choices=ADJUSTMENTS, help="write at this adjustment: a refit blend cannot select one"
    )
    tree = Tree(Scoring, SCORING)
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    parsed = Recorded(tuple(args.names), tree.parse(args), args.track)
    recorded = Recorded.read(root.models() / args.rebuild) if args.rebuild else parsed
    if not recorded.members:
        parser.error("name the models to blend, or --rebuild a written blend")
    group = Group(root, recorded.scoring, torch.device(args.device))
    if args.labelweights:
        if recorded.track is not Track.BROAD:
            parser.error("--labelweights fits on the Broad ruler: pass --track broad")
        weighted = LabelWeighted.read(group, list(recorded.members), Shrinkage(*args.labelweights))
        LOG.info("wrote %s", weighted.write(root.raw(), root.models() / args.out))
        return None
    loaded = Loader.of(group, list(recorded.members))
    runs = Run.discover(root.raw())
    if args.local:
        test, label = LocalTest(loaded, runs), "+".join(recorded.members)
        test.report(label)
        if args.memorisation:
            Memorisation(replace(loaded, partition=Partition.TEST)).report(runs, test.tau(), label)
        return None
    LOG.info(
        "wrote %s", Submission(recorded.track, group.scoring, args.tau).write(loaded, runs, root.models() / args.out)
    )


if __name__ == "__main__":
    main()
