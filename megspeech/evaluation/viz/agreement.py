"""How far an export's per-word sensor saliency is a property of the recipe rather than of one model: each map's
residual (the model's mean map removed) correlated across models for the same word, against words of other
sentences, and against the target span's input amplitude (IG is input x gradient, so a loud sensor earns share)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

TEMPORAL = ("Left-temporal", "Right-temporal")


@dataclass(frozen=True)
class Saliencies:
    """One export's maps, read back from its files."""

    names: list[str]
    maps: Float[np.ndarray, "m n c"]
    """Each model's per-word share per sensor, rows summing to 1."""
    correct: Bool[np.ndarray, "m n"]
    sentence: Int[np.ndarray, " n"]
    amplitude: Float[np.ndarray, "n c"]
    """Mean |x| over each word's target span, per sensor."""
    regions: dict[str, list[int]]
    kinds: list[str]

    @classmethod
    def read(cls, export: Path) -> "Saliencies":
        manifest = json.loads((export / "manifest.json").read_text())
        samples = json.loads((export / "samples.json").read_text())
        channels = manifest["sensors"]
        maps: list[np.ndarray] = []
        correct: list[list[bool]] = []
        for model in manifest["models"]:
            file = export / "models" / model["saliency"]["file"]
            raw = np.fromfile(file, np.uint8).reshape(len(samples), -1).astype(np.float64)
            maps.append(raw / raw.sum(axis=1, keepdims=True).clip(min=1))
            records = json.loads(file.with_name(f"{model['name']}.json").read_text())["samples"]
            correct.append([records[s["id"]]["pred"] == s["label"] for s in samples])
        _, sentence = np.unique([s["sentence"] for s in samples], return_inverse=True)
        return cls(
            [model["name"] for model in manifest["models"]],
            np.stack(maps),
            np.array(correct),
            sentence,
            np.stack([cls.span(export, sample) for sample in samples]),
            channels["regions"],
            channels["kinds"],
        )

    @staticmethod
    def span(export: Path, sample: dict[str, object]) -> Float[np.ndarray, " c"]:
        meg: dict[str, float] = sample["meg"]  # type: ignore[assignment]
        x = np.fromfile(export / str(meg["file"]), np.float32).reshape(int(meg["channels"]), int(meg["times"]))
        onset, offset = ((np.array(sample["target_t"]) - meg["t0"]) * meg["sfreq"]).astype(int)
        lo = int(np.clip(onset, 0, x.shape[1] - 1))
        return np.abs(x[:, lo : max(lo + 1, offset)]).mean(axis=1)


@dataclass(frozen=True)
class Agreement:
    """The statistics one `Saliencies` supports."""

    saliencies: Saliencies

    @staticmethod
    def unit(rows: Float[np.ndarray, "... c"]) -> Float[np.ndarray, "... c"]:
        return rows / np.linalg.norm(rows, axis=-1, keepdims=True).clip(min=np.finfo(rows.dtype).tiny)

    @staticmethod
    def pairs(z: Float[np.ndarray, "m n c"], other: Int[np.ndarray, " n"], keep: Bool[np.ndarray, " n"]) -> float:
        """Mean over model pairs of one model's row i correlated with another's row `other[i]`, over `keep`."""
        m = len(z)
        return float(
            np.mean([(z[i] * z[j][other]).sum(axis=1)[keep].mean() for i in range(m) for j in range(i + 1, m)])
        )

    def residuals(self) -> Float[np.ndarray, "m n c"]:
        maps = self.saliencies.maps
        return self.unit(maps - maps.mean(axis=1, keepdims=True))

    def ratios(self) -> dict[str, float]:
        """Each region's and sensor kind's mean share per sensor, as a multiple of the uniform share."""
        maps = self.saliencies.maps
        mean = maps.mean(axis=(0, 1)) * maps.shape[2]
        kinds = np.array(self.saliencies.kinds)
        groups = {**self.saliencies.regions, **{kind: np.flatnonzero(kinds == kind).tolist() for kind in set(kinds)}}
        return {name: float(mean[index].mean()) for name, index in groups.items()}

    def replication(self, seed: int = 0) -> dict[str, float]:
        """Residual correlation across models for the same word, a word of another sentence, and that same-word
        figure after the input-amplitude direction is projected out; plus residual-to-amplitude correlation."""
        z = self.residuals()
        n = z.shape[1]
        same = np.arange(n)
        shuffled = np.random.default_rng(seed).permutation(n)
        sentence = self.saliencies.sentence
        loudness = self.saliencies.amplitude
        amplitude = self.unit(loudness - loudness.mean(axis=0))
        projected = self.unit(z - (z * amplitude).sum(axis=2, keepdims=True) * amplitude)
        everywhere = np.ones(n, dtype=bool)
        return {
            "same_word": self.pairs(z, same, everywhere),
            "other_sentence": self.pairs(z, shuffled, sentence[shuffled] != sentence),
            "amplitude": float((z * amplitude).sum(axis=2).mean()),
            "same_word_without_amplitude": self.pairs(projected, same, everywhere),
        }

    def per_model(self) -> dict[str, float]:
        """Each model's mean same-word residual correlation with every other model: who reads like the rest."""
        z = self.residuals()
        similarity = np.einsum("anc,bnc->ab", z, z) / z.shape[1]
        np.fill_diagonal(similarity, np.nan)
        return dict(zip(self.saliencies.names, np.nanmean(similarity, axis=1).tolist(), strict=True))

    def errors(self) -> dict[str, float]:
        """The temporal regions' total share on top-1 correct words and on wrong ones."""
        temporal = sorted({c for region in TEMPORAL for c in self.saliencies.regions[region]})
        share = self.saliencies.maps[:, :, temporal].sum(axis=2)
        correct = self.saliencies.correct
        return {"correct": float(share[correct].mean()), "wrong": float(share[~correct].mean())}

    def report(self) -> dict[str, dict[str, float]]:
        return {
            "ratios": self.ratios(),
            "replication": self.replication(),
            "temporal": self.errors(),
            "models": self.per_model(),
        }
