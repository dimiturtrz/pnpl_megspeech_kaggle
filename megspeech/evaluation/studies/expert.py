"""The expert gates (megspeech-6ow.3): does a MEG encoder trained on one teacher target alone (`--optim.heads 0`) hold
anything the frozen base lacks?

An expert is three fold models and a full-data model, each trained on one teacher's per-word targets. Its feature is
the L2-normalised output of its speech head, which lives in the teacher's space, so the fold models' features are
comparable. Each OOF word is read by the expert fold that held its session out; official-validation words by the
full-data model. The booster of `megspeech.evaluation.studies.boosted` then reads the base's log-probabilities and the
experts' features of the word and its neighbours, and a ridge asks whether the features predict the base's residual
on rows no model trained on.

    uv run python -m megspeech.evaluation.studies.expert --folds kf5t-f0 ... kf5t-f4 --base peerbase-s0 \\
        --experts textword textsent audword audsent
"""

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.boosted import Booster, Boosting, Design, Rows
from megspeech.evaluation.studies.peer import EPOCHS, HEADS, SEEDS, Dump, Probe, Study
from megspeech.evaluation.studies.syntax import Embedded
from megspeech.model.auxiliary import Tap

LOG = logging.getLogger(__name__)

TAPS = {
    "textword": Tap.WORD,
    "oracle": Tap.WORD,
    "oraclew": Tap.WORD,
    "oraclel": Tap.WORD,
    "oraclex": Tap.CONTEXT,
    "oraclesw": Tap.WORD,
    "oraclesx": Tap.CONTEXT,
    "textwordw": Tap.WORD,
    "textwordc": Tap.WORD,
    "textsent": Tap.CONTEXT,
    "audword": Tap.WORD,
    "audwordw": Tap.WORD,
    "audsent": Tap.CONTEXT,
}
"""What each expert's speech head read in training."""
FOLDS = 3
COMPONENTS = 32
"""Directions of an expert's feature the booster reads: its leading principal components over the OOF rows."""
RIDGE = 0.01
"""Ridge of the residual probe per fitted row, on standardised features."""
RIDGE_FLOOR = 1e-6


@dataclass(frozen=True)
class Expert:
    """One teacher's expert: its models are `xp-<name>-<k>` for k in `FOLDS` and `xp-<name>-full`."""

    name: str

    @property
    def tap(self) -> Tap:
        return TAPS[self.name]

    def model(self, fold: int | None) -> str:
        """The run's name: fold `fold`'s, or the full-data one."""
        return f"xp-{self.name}-{'full' if fold is None else fold}"

    def held(self, root: DataRoot, folds: Sequence[int] = range(FOLDS)) -> list[set[str]]:
        """The names of the runs each of `folds`' models held out."""
        out = []
        for k in folds:
            data = DataConfig.shared([root.models() / self.model(k)], DataConfig.grouping)
            out.append({run.name for run in data.split(Run.everywhere(root))[Partition.VALIDATION]})
        return out


@dataclass(frozen=True)
class Head:
    """An expert model's speech head over its tap: the unit-norm teacher-space vector of every word."""

    embedded: Embedded
    tap: Tap
    weight: Tensor
    bias: Tensor

    @classmethod
    def load(cls, directory: Path, tap: Tap, device: torch.device) -> "Head":  # devtools-ignore: test-mirror
        saved = torch.load(directory / HEADS, map_location=device)
        projection = saved["heads.0.project.weight"], saved["heads.0.project.bias"]
        return cls(Embedded.load(directory / "model.pt", device), tap, *projection)

    def vectors(self, split: Split) -> Float[Tensor, "n d"]:
        """Every word of `split`'s head output, L2-normalised, in word order."""
        taken = torch.from_numpy(self.embedded(split)[self.tap]).to(self.weight.device)
        return functional.normalize(functional.linear(taken, self.weight, self.bias), dim=-1).cpu()


@dataclass(frozen=True)
class Features:
    """An expert's vectors for the OOF rows and the official-validation rows, in the rows' own order."""

    oof: Float[Tensor, "n d"]
    val: Float[Tensor, "m d"]

    @classmethod
    def build(  # devtools-ignore: test-mirror
        cls, expert: Expert, root: DataRoot, folds: list[str], domains: dict[str, Split], device: torch.device
    ) -> "Features":
        """Each OOF word read by the expert fold that held its run out, in `Dump.build`'s row order."""
        held = expert.held(root)
        heads = [Head.load(root.models() / expert.model(k), expert.tap, device) for k in range(FOLDS)]
        parts: list[Tensor] = []
        for name in folds:
            data = DataConfig.shared([root.models() / name], DataConfig.grouping)
            reader = Reader(root, data, device, geometry=data.geometry(root, device))
            runs = data.split(Run.everywhere(root))[Partition.VALIDATION]
            for run, (_, split) in reader.splits(runs).items():
                parts.append(heads[next(k for k, names in enumerate(held) if run.name in names)].vectors(split))
        return cls(torch.cat(parts), cls.full(expert, root, domains.values(), device))

    @staticmethod
    def full(expert: Expert, root: DataRoot, splits: Iterable[Split], device: torch.device) -> Float[Tensor, "m d"]:
        """The full-data expert's vectors of every word of `splits`, in order."""
        head = Head.load(root.models() / expert.model(None), expert.tap, device)
        return torch.cat([head.vectors(split) for split in splits])

    def reduced(self, count: int = COMPONENTS) -> "Features":
        """The leading `count` principal components of the OOF rows, both sets projected on them."""
        mean = self.oof.mean(0)
        _, _, basis = torch.linalg.svd(self.oof - mean, full_matrices=False)
        return Features((self.oof - mean) @ basis[:count].T, (self.val - mean) @ basis[:count].T)


@dataclass(frozen=True)
class Gate:
    """The arms of the booster over the experts' features, and the residual probe, on one dump."""

    study: Study
    features: dict[str, Features]

    def design(self, rows: Rows, names: tuple[str, ...], *, val: bool) -> Design:
        """The booster's input: the base's log-probabilities, and the experts' features of the word and its
        neighbours (zeros where it has none) with whether each exists."""
        if not names:
            return Design(rows.base, rows.base, rows.y)
        own = torch.cat([getattr(self.features[n], "val" if val else "oof") for n in names], dim=1)
        padded = torch.cat([own, own.new_zeros(1, own.shape[1])])
        at = torch.where(rows.near >= 0, rows.near, len(own))
        parts = [rows.base, own, padded[at].flatten(1), (rows.near >= 0).float()]
        return Design(torch.cat(parts, dim=1), rows.base, rows.y)

    def trained(self, names: tuple[str, ...]) -> tuple[int, list[Booster]]:
        """`names`' epoch count from the last fold's rows, and a booster per `SEEDS` fitted on every fold for it."""
        study, device = self.study, self.study.device
        train = self.design(study.oof, names, val=False).to(device)
        held = (study.oof.group == torch.max(study.oof.group)).to(device)
        watched = train.where(held).labelled()
        _, curve = Boosting(seed=0).fit(train.where(~held), EPOCHS, watched)
        epochs = int(np.argmin(curve)) + 1
        return epochs, [Boosting(seed=seed).fit(train, epochs)[0] for seed in SEEDS]

    def averaged(self, models: Sequence[Booster], names: tuple[str, ...]) -> Float[Tensor, "n v"]:
        """The `models`' probabilities of every validation row, averaged."""
        val = self.design(self.study.val, names, val=True).to(self.study.device)
        return torch.stack([Boosting.predict(model, val).exp().cpu() for model in models]).mean(0)

    def submissions(
        self, subsets: dict[str, tuple[str, ...]], test: "Gate"
    ) -> dict[str, tuple[float, Float[Tensor, "k v"]]]:
        """Each subset's seed-averaged official validation score, and its probabilities of `test`'s rows."""
        out = {}
        for label, names in subsets.items():
            _, models = self.trained(names)
            out[label] = (self.study.official(self.averaged(models, names).log()), test.averaged(models, names))
            LOG.info("%s holdout: official %.4f", label, out[label][0])
        return out

    def arm(self, label: str, names: tuple[str, ...]) -> dict[str, object]:
        """`names`' arm: its epoch count from the last fold's rows, then its official scores over `SEEDS`."""
        study, device = self.study, self.study.device
        epochs, models = self.trained(names)
        val = self.design(study.val, names, val=True).to(device)
        scores = [study.official(Boosting.predict(model, val).cpu()) for model in models]
        LOG.info("%s epochs %d official %s", label, epochs, scores)
        return {"arm": label, "epochs": epochs, "official": scores, "mean": float(np.mean(scores))}

    def arms(self) -> list[dict[str, object]]:
        """Base-only, each expert, all, and each drop-one."""
        names = tuple(self.features)
        subsets = {"control": ()} | {f"+{n}": (n,) for n in names} | {"all": names}
        if len(names) > 1:
            subsets |= {f"all-{n}": tuple(m for m in names if m != n) for n in names}
        return [self.arm(label, chosen) for label, chosen in subsets.items()]

    def residual(self, name: str) -> dict[str, float]:
        """Held-out R2 gain of a ridge on base log-probabilities plus `name`'s features over the base alone, for the
        base's error and the true word's log-probability margin, each fold predicted by a ridge fitted on the others."""
        rows = self.study.oof
        keep = rows.y >= 0
        base = rows.base[keep]
        truth = rows.y[keep]
        wrong = (base.argmax(-1) != truth).float()
        margin = base.gather(1, truth[:, None])[:, 0] - base.logsumexp(-1)
        group = rows.group[keep]
        own = self.features[name].oof[keep]
        out = {}
        for label, target in (("error", wrong), ("margin", margin)):
            out[label] = self.held_out_r2(torch.cat([base, own], 1), target, group) - self.held_out_r2(
                base, target, group
            )
        return out

    @staticmethod
    def held_out_r2(x: Float[Tensor, "n d"], y: Float[Tensor, " n"], group: Int[Tensor, " n"]) -> float:
        """Mean over groups of the R2 of a ridge fitted on the other groups, standardised on them."""
        scores = []
        for g in group.unique():
            fit, test = group != g, group == g
            mean, scale = x[fit].mean(0), x[fit].std(0).clamp(min=RIDGE_FLOOR)
            a = (x[fit] - mean) / scale
            centre = y[fit].mean()
            gram = a.T @ a + RIDGE * len(a) * torch.eye(a.shape[1])
            beta = torch.linalg.solve(gram.double(), (a.T @ (y[fit] - centre)).double()).float()
            predicted = ((x[test] - mean) / scale) @ beta + centre
            scores.append(1 - float(((y[test] - predicted) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()))
        return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True, help="the peer's cross-fit models, as for `peer`")
    parser.add_argument("--base", required=True)
    parser.add_argument("--experts", nargs="+", required=True, choices=sorted(TAPS))
    parser.add_argument("--refeature", action="store_true")
    parser.add_argument(
        "--holdout",
        type=Path,
        help="write the Deep Kaggle test's submissions to <stem>-<arm>.csv for each of `--arms`, not the gate",
    )
    parser.add_argument(
        "--arms", nargs="+", default=["control"], help="comma-joined expert names per arm; `control` is none"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    loaded = Loader.of(root, [args.base], Scoring(), device)
    domains, _ = loaded.domains(loaded.ruled(Run.discover(root.raw())))
    dump = Dump.load(Dump.path(root, args.base))
    features = {}
    for name in args.experts:
        path = Dump.path(root, f"{args.base}-{name}")
        if args.refeature or not path.exists():
            built = Features.build(Expert(name), root, args.folds, domains, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"oof": built.oof, "val": built.val}, path)
        saved = torch.load(path)
        features[name] = Features(saved["oof"], saved["val"]).reduced()
    study = Study(dump.oof, dump.val, dump.domains, loaded.evaluator, loaded.log_prior, domains, device)
    gate = Gate(study, features)
    if args.holdout:
        subsets = {arm: () if arm == "control" else tuple(arm.split(",")) for arm in args.arms}
        split = loaded.kaggle(Track.DEEP)
        rows = Probe.load(root.models() / args.base, root, device).rows(split, 0)
        tested = {
            name: Features(
                torch.load(Dump.path(root, f"{args.base}-{name}"))["oof"],
                Features.full(Expert(name), root, [split], device),
            ).reduced()
            for name in features
        }
        for label, (score, probabilities) in gate.submissions(subsets, Gate(replace(study, val=rows), tested)).items():
            path = args.holdout.with_name(f"{args.holdout.stem}-{label.replace(',', '-')}.csv")
            LOG.info("%s: official %.4f -> %s", label, score, Track.DEEP.write(root.raw(), probabilities.numpy(), path))
        return
    results = {"arms": gate.arms(), "residual": {name: gate.residual(name) for name in features}}
    LOG.info("%s", json.dumps(results["residual"]))
    (Dump.path(root, f"{args.base}-expert").with_suffix(".json")).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
