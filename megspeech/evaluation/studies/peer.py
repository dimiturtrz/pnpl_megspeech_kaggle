"""The peer protocol (megspeech-6ow): a frozen base plus a booster that learns only the base's residual from per-word
side information about the word and its neighbours.

The base's out-of-fold features come from cross-fit fold models (`--folds`, each read on the sessions it never trained
on), the booster trains on those rows alone, and the official validation split scores `base + booster` with the
official ruler `megspeech.submission` reports. The booster's epoch count is picked on the last fold's rows.

    uv run python -m megspeech.evaluation.studies.peer --folds kf5t-f0 kf5t-f1 kf5t-f2 kf5t-f3 kf5t-f4 \\
        --base peerbase-s0
"""

import argparse
import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.vocabulary import PRIMARY, Vocabulary
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.boosted import ARMS, Arm, Boosting, Rows, Spanned
from megspeech.evaluation.studies.syntax import Embedded
from megspeech.objective.targets.t5 import Lexicon

LOG = logging.getLogger(__name__)

HEADS = "heads.pt"
OUT = "peer"
EPOCHS = 40
"""Most epochs a booster is trained for when its count is picked on the watched fold."""
SEEDS = (0, 1, 2)
LEXICON_CHUNK = 4096
"""Words whose lexical log-probabilities are normalised over the whole lexicon at once."""


@dataclass(frozen=True)
class Probe:
    """A tagged-head model with a lexical head, reading every word of a split as the booster's side information."""

    embedded: Embedded
    lexical: nn.Linear
    columns: torch.Tensor
    """The lexicon rows of the 50 words, in label order."""

    @classmethod
    def load(cls, directory: Path, root: DataRoot, device: torch.device) -> "Probe":  # devtools-ignore: test-mirror
        """The model, and the lexical head (the saved head whose outputs span the lexicon), under `directory`."""
        embedded = Embedded.load(directory / "model.pt", device)
        lexicon = Lexicon.load(Lexicon.path(root)).to(device)
        saved = torch.load(directory / HEADS, map_location=device)
        weight = next(v for k, v in saved.items() if k.endswith("weight") and v.shape[0] == len(lexicon.keys))
        bias = saved[next(k for k in saved if saved[k] is weight).replace("weight", "bias")]
        head = nn.Linear(weight.shape[1], weight.shape[0]).to(device)
        head.load_state_dict({"weight": weight, "bias": bias})
        columns = lexicon.rows(
            torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words).str.replace("’", "'"))).to(device)
        )
        if (columns < 0).any():
            raise ValueError("a word of the 50 is missing from the lexicon")
        return cls(embedded, head, columns)

    def lexicon(self, slots: Float[Tensor, "n e"]) -> Float[Tensor, "n v"]:  # devtools-ignore: test-mirror
        """The lexical head's log-probabilities of the 50 for each embedding in `slots`, normalised over the lexicon."""
        with torch.no_grad():
            chunks = [self.lexical(chunk).log_softmax(-1)[:, self.columns] for chunk in slots.split(LEXICON_CHUNK)]
        return torch.cat(chunks)

    def rows(self, split: Split, group: int) -> Rows:  # devtools-ignore: test-mirror
        """Every word of `split` as `Rows`, in word order, all of them `group`."""
        base, slots = self.embedded.read(split)
        words = len(split.y)
        spans = Spanned(split.index.cpu().numpy(), split.onset.cpu().numpy())
        return Rows(
            base.cpu(),
            self.embedded.tag_probabilities(slots).cpu(),
            self.lexicon(slots).cpu(),
            torch.from_numpy(spans.gaps(words)).float(),
            split.y.cpu(),
            torch.from_numpy(spans.near(words)),
            torch.full((words,), group),
        )


@dataclass(frozen=True)
class Dump:
    """The out-of-fold rows the booster trains on (group = fold) and the official validation rows it is scored on
    (group = index into `domains`)."""

    oof: Rows
    val: Rows
    domains: tuple[str, ...]

    @staticmethod
    def path(root: DataRoot, name: str) -> Path:
        return root.models() / OUT / f"{name}.pt"

    def save(self, path: Path) -> None:  # devtools-ignore: test-mirror
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = {
            kind: {f.name: getattr(rows, f.name) for f in fields(Rows)}
            for kind, rows in (("oof", self.oof), ("val", self.val))
        }
        torch.save({**parts, "domains": self.domains}, path)

    @classmethod
    def load(cls, path: Path) -> "Dump":  # devtools-ignore: test-mirror
        saved = torch.load(path)
        return cls(Rows(**saved["oof"]), Rows(**saved["val"]), tuple(saved["domains"]))

    @classmethod
    def build(  # devtools-ignore: test-mirror
        cls, root: DataRoot, folds: list[str], base: str, domains: dict[str, Split], device: torch.device
    ) -> "Dump":
        """Fold `k`'s model read on the sessions it held out, and `base` read on the official validation `domains`."""
        out_of_fold: list[Rows] = []
        for k, name in enumerate(folds):
            probe = Probe.load(root.models() / name, root, device)
            data = DataConfig.shared([root.models() / name], DataConfig.grouping)
            reader = Reader(root, data, device, geometry=data.geometry(root, device))
            held = data.split(Run.everywhere(root))[Partition.VALIDATION]
            out_of_fold += [probe.rows(split, k) for _, split in reader.splits(held).values()]
            LOG.info("fold %d (%s): %d held-out sessions", k, name, len(held))
        probe = Probe.load(root.models() / base, root, device)
        val = [probe.rows(split, g) for g, split in enumerate(domains.values())]
        return cls(Rows.stacked(out_of_fold), Rows.stacked(val), tuple(domains))


@dataclass(frozen=True)
class Study:
    """The arms of the booster, each fitted on the out-of-fold rows and scored on the official validation split."""

    oof: Rows
    val: Rows
    names: tuple[str, ...]
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]
    domains: dict[str, Split]
    device: torch.device

    def official(self, logp: Float[Tensor, "n v"]) -> float:  # devtools-ignore: test-mirror
        """The macro balanced accuracy@10 the official ruler reads, of `logp` for every validation word."""
        logits = {}
        for g, name in enumerate(self.names):
            mine = logp[self.val.group == g].to(self.device)
            logits[name] = {Head.CONTEXT: mine, Head.WORD: mine}
        return self.evaluator.summarised(logits, self.domains, self.log_prior, 0, 0.0).macro.bacc10

    def arm(self, arm: Arm) -> dict[str, object]:  # devtools-ignore: test-mirror
        """`arm`'s epoch count from the last fold's rows, then its official scores over `SEEDS` fitted on every fold."""
        train = self.oof.design(arm.kinds, neighbours=arm.neighbours).to(self.device)
        held = (self.oof.group == torch.max(self.oof.group)).to(self.device)
        watched = train.where(held).labelled()
        _, curve = Boosting(seed=0).fit(train.where(~held), EPOCHS, watched)
        epochs = int(np.argmin(curve)) + 1
        base_loss = functional.cross_entropy(watched.base, watched.y).item()
        val = self.val.design(arm.kinds, neighbours=arm.neighbours).to(self.device)
        scores = []
        for seed in SEEDS:
            model, _ = Boosting(seed=seed).fit(train, epochs)
            scores.append(self.official(model.predict(val).cpu()))
        LOG.info("%s epochs %d fold-CE base %.4f -> %.4f official %s", arm.name, epochs, base_loss, min(curve), scores)
        return {"arm": arm.name, "epochs": epochs, "base_ce": base_loss, "boosted_ce": min(curve), "official": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True, help="cross-fit fold models, fold k held out by member k")
    parser.add_argument("--base", required=True, help="the full-data base scored on the official validation split")
    parser.add_argument("--redump", action="store_true", help="read the rows from the models again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    loaded = Loader.of(Group(root, Scoring(), device), [args.base])
    domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
    path = Dump.path(root, args.base)
    if args.redump or not path.exists():
        Dump.build(root, args.folds, args.base, domains, device).save(path)
    dump = Dump.load(path)
    study = Study(dump.oof, dump.val, dump.domains, loaded.evaluator, loaded.log_prior, domains, device)
    base = study.official(study.val.base)
    LOG.info("base %s alone, official %.4f (%d words, macro over novel spans)", args.base, base, len(study.val.y))
    results = [{"arm": "base", "official": [base]}, *(study.arm(arm) for arm in ARMS)]
    (path.parent / f"{args.base}-results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
