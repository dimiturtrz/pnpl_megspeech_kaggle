"""Whether a model's acoustic head (`--optim.acoustic`) reads the speech envelope out of held-out MEG: the Pearson r
between its prediction over each validation word's window and the true envelope there, on the validation runs that
have an envelope file (`megspeech.objective.targets.loudness`). r > 0 on windows it never trained on is the gate;
the mean over windows scores the envelope's shape within a window, the pooled r its level across them.

    uv run python -m megspeech.evaluation.studies.acoustic acoustic-s0
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.raw.runs import Column, Run
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.evaluation.reader import Reader
from megspeech.model.auxiliary import Tap
from megspeech.model.checkpoint import Checkpoint
from megspeech.model.sentence import SentenceNet
from megspeech.objective.auxiliary.acoustic import Acoustic
from megspeech.objective.auxiliary.order import HeadOrder
from megspeech.objective.targets.loudness import Loudness

LOG = logging.getLogger(__name__)

HEADS = "heads.pt"


@dataclass(frozen=True)
class Envelopes:
    """Predicted and true envelope windows of the same words, the true ones NaN where a word has none."""

    predicted: Float[np.ndarray, "n f"]
    truth: Float[np.ndarray, "n f"]

    def present(self) -> "Envelopes":
        """The words with a true envelope."""
        keep = np.isfinite(self.truth).all(axis=1)
        return Envelopes(self.predicted[keep], self.truth[keep])

    def windows(self) -> float:
        """The mean over present words of the Pearson r between a window's prediction and its truth; NaN windows
        (constant ones) are left out."""
        present = self.present()
        centred = present.predicted - present.predicted.mean(axis=1, keepdims=True)
        truth = present.truth.astype(np.float64) - present.truth.astype(np.float64).mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (centred * truth).sum(axis=1) / np.sqrt((centred**2).sum(axis=1) * (truth**2).sum(axis=1))
        return float(np.nanmean(r))

    def pooled(self) -> float:
        """The Pearson r over every present frame at once."""
        present = self.present()
        return float(np.corrcoef(present.predicted.ravel(), present.truth.astype(np.float64).ravel())[0, 1])


@dataclass(frozen=True)
class Readout:
    """A trained model and its acoustic head."""

    net: SentenceNet
    head: Acoustic

    @classmethod
    def load(  # devtools-ignore: test-mirror
        cls, net: SentenceNet, directory: Path, device: torch.device, frames: int
    ) -> "Readout":
        """`net` with the acoustic head among `directory`'s saved heads, spanning the run's `frames`."""
        weights = Tree.flat(json.loads((directory / METRICS).read_text())["config"]["optim"])
        prefix = f"heads.{HeadOrder.position(weights, 'acoustic')}."
        saved = torch.load(directory / HEADS, map_location=device, weights_only=True)
        head = Acoustic(Tap.TIME.width(net), frames).to(device)
        head.load_state_dict({key.removeprefix(prefix): v for key, v in saved.items() if key.startswith(prefix)})
        return cls(net.eval(), head.eval())

    def __call__(self, split: Split) -> Float[np.ndarray, "n f"]:  # devtools-ignore: test-mirror
        """The predicted envelope of every word of `split`'s window, in word order."""
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        device = split.index.device
        out = torch.zeros(words + 1, self.head.frames, device=device)
        with torch.no_grad():
            for spans in torch.arange(len(split.index), device=device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                *_, times = self.net.heard_timed(batch.inputs())
                out[torch.where(batch.index != PAD, batch.index, words)] = self.head(times).float()
        return out[:words].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a model trained with --optim.acoustic under models/")
    parser.add_argument("--device", default=Device.CUDA)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device(args.device)
    trained_in = root.models() / args.name
    data = DataConfig.shared([trained_in], DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False))
    cut = data.cut()
    readout = Readout.load(Checkpoint.load(trained_in / "model.pt").to(device), trained_in, device, cut[0])
    predicted: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for runs in reader.ruled(Run.discover(root.raw())).values():
        for run, (words, split) in reader.splits(runs).items():
            path = Loudness.directory(root, cut) / f"{run.name}.npy"
            if not path.exists():
                continue
            predicted.append(readout(split))
            truth.append(np.load(path)[words[Column.POSITION].to_numpy()].astype(np.float32))
    if not predicted:
        raise FileNotFoundError("no validation run has an envelope file: nothing to score")
    envelopes = Envelopes(np.concatenate(predicted), np.concatenate(truth))
    LOG.info(
        "held-out envelope r: per-window mean %.4f, pooled %.4f over %d of %d words",
        envelopes.windows(),
        envelopes.pooled(),
        len(envelopes.present().truth),
        len(envelopes.truth),
    )


if __name__ == "__main__":
    main()
