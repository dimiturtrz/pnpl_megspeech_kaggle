"""Frozen Whisper encoder states of the audiobook speech each training word was heard in (megspeech-6ow.1): the audio
expert's per-word teacher, next to the wav2vec2 latents of `megspeech.objective.targets.speech`.

Each trained run's chapter is read by the encoder in `WINDOW`-second windows advancing by `HOP`, each frame kept from
the window where it is at least `MARGIN` seconds from an edge. A word's target is the mean state over the same 0.5 s
after its onset as the wav2vec2 target's, per layer in `LAYERS`, z-scored over the run's aligned words and written to
its own directory under processed/speech, so a ridge probe can pick the layer MEG predicts best.

uv run python -m megspeech.objective.targets.whisper
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformers import WhisperFeatureExtractor, WhisperModel

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import SAMPLING, STRIDE, Chapter, Heard, Targets
from megspeech.objective.targets.t5 import Encoder

MODEL = "openai/whisper-medium"
STEM = MODEL.split("/")[-1]
LAYERS = (8, 12, 16, 20, 24)
"""Encoder hidden states of whisper-medium's 24 layers, from the lower-middle to the last."""
WINDOW = 30 * SAMPLING
"""Samples one forward pass reads: Whisper's fixed 30 s."""
HOP = 20 * SAMPLING
"""Samples between window starts: 10 s of overlap, 5 s read on each side of every kept frame."""
FRAMES = WINDOW // STRIDE
"""Frames of one window's encoder output, 50 a second like the wav2vec2 latents'."""
MARGIN = 5 * SAMPLING // STRIDE
"""Frames of a window, from each edge, that a later or earlier window supplies instead."""
BATCH = 8
"""Windows read per forward pass."""
DEVICE = "cuda"
LOG = logging.getLogger(__name__)

Extractor = Callable[[list[np.ndarray]], Float[Tensor, "b m t"]]


@dataclass(frozen=True)
class Listener:
    """Whisper's encoder behind its log-mel `extractor`."""

    extractor: Extractor
    encoder: Encoder

    def latents(self, chapter: Chapter, device: torch.device) -> dict[int, Float[Tensor, "f d"]]:
        """Every frame's state at each of `LAYERS`, windows of `WINDOW` advancing by `HOP`, a frame kept from the first
        window that has it `MARGIN` frames inside."""
        total = chapter.frames
        starts = list(range(0, max(total * STRIDE - WINDOW + HOP, 1), HOP))
        peak = float(np.abs(chapter.audio).max()) or 1.0
        audio = chapter.audio / peak
        kept: dict[int, list[Tensor]] = {layer: [] for layer in LAYERS}
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for first in range(0, len(starts), BATCH):
                chosen = starts[first : first + BATCH]
                features = self.extractor([audio[s : s + WINDOW] for s in chosen]).to(device)
                states = self.encoder(input_features=features, output_hidden_states=True).hidden_states
                for row, start in enumerate(chosen):
                    begin = 0 if start == 0 else MARGIN
                    end = FRAMES if start == starts[-1] else FRAMES - MARGIN
                    for layer in LAYERS:
                        kept[layer].append(states[layer][row, begin:end].float())
        return {layer: torch.cat(parts)[:total] for layer, parts in kept.items()}

    @classmethod
    def load(cls, root: DataRoot, device: torch.device) -> "Listener":
        folder = root.foreign(STEM)
        extractor = WhisperFeatureExtractor.from_pretrained(folder)
        encoder = WhisperModel.from_pretrained(folder).encoder.to(device).eval()
        return cls(
            lambda windows: (
                extractor(windows, sampling_rate=SAMPLING, return_tensors="pt", device=device.type).input_features
            ),
            encoder,
        )

    def targets(self, entry: Heard, device: torch.device) -> dict[int, Targets]:
        """Each layer's targets of one run's heard words, those outside its aligned blocks absent."""
        chapter = Chapter.read(entry.chapters()[0])
        onset = entry.onsets()
        keep = Targets.aligned(chapter.contrast(onset))
        return {
            layer: Targets.pool(latents, onset).masked(keep).standardised()
            for layer, latents in self.latents(chapter, device).items()
        }

    @staticmethod
    def directory(root: DataRoot, layer: int) -> Path:
        """Where the targets of `layer` live."""
        return Targets.directory(root, f"{STEM}-L{layer}")

    def write(self, root: DataRoot, device: torch.device) -> Path:
        """Every trained run's targets under `directory`, the runs already written kept."""
        for entry in Heard.trained(Run.discover(root.raw())):
            paths = {layer: Listener.directory(root, layer) / f"{entry.run.name}.npy" for layer in LAYERS}
            if all(path.exists() for path in paths.values()):
                continue
            written = self.targets(entry, device)
            for layer, targets in written.items():
                targets.save(paths[layer])
            LOG.info("%s: %d words", entry.run.name, len(written[LAYERS[0]].vectors))
        return Listener.directory(root, LAYERS[0]).parent

    @staticmethod
    def main() -> None:
        """Encode every trained run's chapter on the GPU and write its words' targets under processed/speech."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root, device = DataRoot.from_config(), torch.device(DEVICE)
        LOG.info("wrote %s", Listener.load(root, device).write(root, device))


if __name__ == "__main__":
    Listener.main()
