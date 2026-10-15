"""The viewer's two steps: `export` a stratified sample set with model outputs, `serve` it.

uv run python -m megspeech.evaluation.viz export --models <run> [<run> ...] [--n 1000] [--seed 0] [--device cuda]
uv run python -m megspeech.evaluation.viz serve [--port 8000]
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from megspeech.device import Device
from megspeech.evaluation.viz.export import ExportConfig, Exporter, Sampler
from megspeech.evaluation.viz.outputs import Reader
from megspeech.evaluation.viz.serve import ServeConfig, Server

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cli:
    """Parses one command line and runs the subcommand it names."""

    def parser(self) -> argparse.ArgumentParser:
        """The `export` / `serve` parser; each subcommand binds its handler as `run`."""
        parser = argparse.ArgumentParser(prog="python -m megspeech.evaluation.viz")
        commands = parser.add_subparsers(required=True)
        export = commands.add_parser("export", help="write a sample set + model outputs under <data>/viz/<name>/")
        export.add_argument("--models", nargs="+", required=True)
        export.add_argument("--n", type=int, default=1000)
        export.add_argument("--seed", type=int, default=0)
        export.add_argument("--name", default="default")
        export.add_argument("--device", default=Device.CPU, help="where models read and attribute")
        export.set_defaults(run=self.export)
        serve = commands.add_parser("serve", help="serve web/ at / and <data>/viz/<name>/ at /data/")
        serve.add_argument("--port", type=int, default=8000)
        serve.add_argument("--name", default="default")
        serve.set_defaults(run=self.serve)
        return parser

    def export(self, args: argparse.Namespace) -> None:
        """Write the export `args` describe and log where it landed."""
        config = ExportConfig(models=tuple(args.models), n=args.n, seed=args.seed, name=args.name)
        exporter = Exporter(sampler=Sampler(args.device), reader=Reader(args.device))
        log.info("exported %s", exporter.run(config))

    def serve(self, args: argparse.Namespace) -> None:
        """Serve the export `args.name` on `args.port` until interrupted."""
        try:
            Server(ServeConfig.for_export(args.name, args.port)).serve()
        except KeyboardInterrupt:
            log.info("stopped")

    def run(self, argv: Sequence[str]) -> None:
        """Parse `argv` and run its subcommand."""
        args = self.parser().parse_args(argv)
        args.run(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Cli().run(sys.argv[1:])


if __name__ == "__main__":
    main()
