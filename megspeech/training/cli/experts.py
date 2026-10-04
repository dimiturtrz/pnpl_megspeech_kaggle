"""Assemble a stacked run and the expert runs it read into one model whose experts fine-tune under its context model.

The stacked run's context model read the experts' frozen embeddings (`--context.encoder trained`); the assembled one
reads the MEG through the experts' own CNNs (`--context.encoder experts`), each on its own crop of the widest
expert's window, so at step 0 it is the stacked model and a warm start from it trains every part together
(megspeech-7gf.90).

uv run python -m megspeech.training.cli.experts --stack comp3-s0 --experts sent40-ioi-ema9998-s0 acou-w02-ctx-s0 \
    rec-d02-ctx-s0 --name experts3
uv run python -m megspeech.training.cli.train --name experts3-ft --optim.warm experts3 --optim.lr 3e-4
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS
from megspeech.model.config import Encoder
from megspeech.model.input.scaler import CLAMP, Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.training.config import METRICS, RunConfig

LOG = logging.getLogger(__name__)

MODEL = "model.pt"
SPREAD = torch.finfo(torch.float16).max / CLAMP
"""The sentence model's own scaler's spread under experts that scale for themselves: it divides by this, each
expert's scaler by its spread over this, so the two compose to the expert's own and the outer clamp never acts."""
REBUILT = ("scale.", "subjects.", "whiten.")
"""The stacked run's state over the embeddings it read, which the assembled model reads from the sensors instead."""


@dataclass(frozen=True)
class Run:
    config: RunConfig
    state: dict[str, Tensor]

    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls(RunConfig.saved(directory), torch.load(directory / MODEL, map_location="cpu", weights_only=True))

    def plain(self) -> bool:
        """Whether the run read the sensors as recorded, through its CNN alone: what an expert branch reproduces."""
        context, data = self.config.context, self.config.data
        plain = context.encoder is Encoder.CONV and not (context.subjects or context.whiten or context.latency)
        return plain and not (data.projected or data.features or data.baseline)

    def weights(self, prefixes: tuple[str, ...], *, kept: bool) -> dict[str, Tensor]:
        """The run's weights under any of `prefixes` when `kept`, else those under none of them."""
        return {key: value for key, value in self.state.items() if key.startswith(prefixes) == kept}


@dataclass(frozen=True)
class Assembly:
    """`stack`, a context model over the joined embeddings of `experts` in order, rebuilt over the experts' CNNs."""

    stack: Run
    experts: tuple[Run, ...]

    def config(self, *, rhythm: bool = False) -> RunConfig:
        """The stack's config reading the MEG in the widest expert's window, each expert on its own crop of it."""
        conv = self.experts[0].config.context.conv
        if any(expert.config.context.conv != conv or not expert.plain() for expert in self.experts):
            raise ValueError("experts must share one CNN shape and read the sensors as recorded through it alone")
        widest = max((expert.config.data for expert in self.experts), key=self.end)
        crops = [self.crop(widest, expert.config.data) for expert in self.experts]
        starts, lengths = (tuple(column) for column in zip(*crops, strict=True))
        context = replace(
            self.stack.config.context, encoder=Encoder.EXPERTS, conv=conv, starts=starts, lengths=lengths, rhythm=rhythm
        )
        return replace(self.stack.config, data=self.placement(self.stack.config.data, widest), context=context)

    @staticmethod
    def offset(data: DataConfig) -> int:
        """Samples from each word's onset to the start of its window as `data` reads it."""
        return data.to_samples(data.delay) - data.to_samples(data.lead)

    @classmethod
    def end(cls, data: DataConfig) -> int:
        return cls.offset(data) + data.samples + data.to_samples(data.lead)

    @classmethod
    def crop(cls, window: DataConfig, expert: DataConfig) -> tuple[int, int]:
        """Where in `window`'s reading `expert`'s window starts, and its samples."""
        start, length = cls.offset(expert) - cls.offset(window), expert.samples + expert.to_samples(expert.lead)
        if start < 0 or start + length > cls.end(window) - cls.offset(window) or expert.rate != window.rate:
            raise ValueError(f"an expert window of {length} samples at {start} lies outside the shared window")
        return start, length

    @staticmethod
    def placement(base: DataConfig, data: DataConfig) -> DataConfig:
        """`base` reading the MEG in the window `data` places and samples."""
        return replace(
            base,
            window=data.window,
            lead=data.lead,
            delay=data.delay,
            baseline=data.baseline,
            decimate=data.decimate,
            features="",
        )

    def state(self, config: RunConfig) -> dict[str, Tensor]:
        """The assembled model's weights: the stack's above the embeddings, its embedding scaler over the joined
        experts, each expert's CNN and scaler beneath, and the shape `config` builds for whatever none of them has."""
        template = SentenceNet(Scaler(torch.zeros(1, CHANNELS, 1)), context=config.context).state_dict()
        state = self.stack.weights(REBUILT, kept=False)
        state |= {f"words.features.{name}": self.stack.state[f"scale.{name}"] for name in ("median", "spread")}
        for index, expert in enumerate(self.experts):
            branch = f"words.experts.{index}."
            state |= {branch + key: value for key, value in expert.weights(("words.",), kept=True).items()}
            state |= {f"{branch}scale.{name}": expert.state[f"scale.{name}"] / SPREAD for name in ("median", "spread")}
        state |= {"scale.median": torch.zeros(CHANNELS, 1), "scale.spread": torch.full((CHANNELS, 1), SPREAD)}
        mismatched = [key for key, value in state.items() if key not in template or template[key].shape != value.shape]
        if mismatched:
            raise ValueError(f"assembled weights the model has no place for: {mismatched}")
        return template | state

    def save(self, directory: Path, *, rhythm: bool = False) -> None:
        config = self.config(rhythm=rhythm)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(config), directory / MODEL)
        config.context.save(directory)
        (directory / METRICS).write_text(json.dumps({"config": asdict(config)}, indent=1))


def main() -> None:
    """Write the assembled model under models/<name>, ready to warm-start a run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rhythm", action="store_true", help="a timing head beside the context head, fresh")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    models = DataRoot.from_config().models()
    assembly = Assembly(Run.load(models / args.stack), tuple(Run.load(models / name) for name in args.experts))
    assembly.save(models / args.name, rhythm=args.rhythm)
    LOG.info("%s -> %s", assembly.config(rhythm=args.rhythm).context, models / args.name)


if __name__ == "__main__":
    main()
