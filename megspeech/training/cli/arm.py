"""Run one recipe arm: a stem from `<base>-s<seed>`, then its fine-tune from `<base>-s<seed>-ft-s<seed>` started on that
stem, the same flags in both stages plus `--stem-flags` in the stem alone. A stem already saved is reused, so a run that
died after it resumes at the fine-tune. Each stage reports val and local test when it ends.

uv run python -m megspeech.training.cli.arm --base ema-cd25-xsp10sem05-ac05x --seed 1 --tag h512 \
    -- --context.conv.hidden 512
uv run python -m megspeech.training.cli.arm --base ema-cd25-xsp10sem05-ac05x --seed 1 --tag stem20 \
    --stem-flags="--optim.epochs 20"
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.paths import DataRoot

TRAIN = (sys.executable, "-m", "megspeech.training.cli.train")
WEIGHTS = "model.pt"


@dataclass(frozen=True)
class Arm:
    base: str
    seed: int
    tag: str
    flags: tuple[str, ...] = ()
    stem_flags: tuple[str, ...] = ()

    @property
    def stem(self) -> str:
        return f"{self.base}-{self.tag}-s{self.seed}"

    @property
    def fine(self) -> str:
        return f"{self.stem}-ft-s{self.seed}"

    def commands(self, models: Path) -> list[list[str]]:
        """The training commands still to run: the fine-tune, preceded by the stem unless `models` holds it."""
        shared = ["--optim.seed", str(self.seed), *self.flags]
        stem = [*TRAIN, "--origin", f"{self.base}-s{self.seed}", *shared, *self.stem_flags, "--name", self.stem]
        origin = f"{self.base}-s{self.seed}-ft-s{self.seed}"
        fine = [*TRAIN, "--origin", origin, "--optim.stem", self.stem, *shared, "--name", self.fine]
        return [fine] if (models / self.stem / WEIGHTS).is_file() else [stem, fine]

    def run(self, models: Path) -> None:
        for command in self.commands(models):
            subprocess.run(command, check=True)  # noqa: S603

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base", required=True, help="recipe whose -s<seed> stem and -ft-s<seed> runs are origins")
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--tag", required=True)
        parser.add_argument("--stem-flags", default="", help="training flags for the stem alone, one quoted string")
        parser.add_argument("flags", nargs=argparse.REMAINDER, help="training flags for both stages, after --")
        args = parser.parse_args()
        flags = tuple(flag for flag in args.flags if flag != "--")
        arm = Arm(args.base, args.seed, args.tag, flags, tuple(shlex.split(args.stem_flags)))
        arm.run(DataRoot.from_config().models())


if __name__ == "__main__":
    Arm.main()
