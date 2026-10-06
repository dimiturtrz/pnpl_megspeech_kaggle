"""Read the frozen-encoder k-NN probe on a saved supervised model: the positive control for the pretraining probe.

uv run python -m megspeech.training.cli.probe --model sent40-ioi-ema9998-s0
"""

import argparse
import logging

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.raw.runs import Run
from megspeech.model.sentence import SentenceNet
from megspeech.training.config import RunConfig
from megspeech.training.pretrain.progress import Probing, Progress
from megspeech.training.pretrain.windows import Sessions

LOG = logging.getLogger(__name__)


def main() -> None:
    """Log the probe's carved-val BAcc@10 for `--model`'s encoder, on the sessions the model trained on."""
    tree = Tree(Probing)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a saved run under the models root")
    tree.arguments(parser)
    args = parser.parse_args()
    probing = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = root.models() / args.model
    data = RunConfig.saved(directory).data
    sessions = Sessions.of(data, Run.everywhere(root))
    recording = DeviceRecording.load(
        data.run_cache(root),
        device,
        sessions.names,
        standardised=data.standardised,
        geometry=data.geometry(root, device),
    )
    net = SentenceNet.load(directory / "model.pt").to(device)
    score = Progress.load(data, root, recording, probing, device).score(net)
    LOG.info("probe %s BAcc@10 %.4f (chance 0.2, words %d, k %d)", args.model, score, probing.words, probing.k)


if __name__ == "__main__":
    main()
