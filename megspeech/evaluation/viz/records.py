"""The sample set the viewer browses: each picked word's text, strata, MEG window and audio, written as files."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.runs import Column
from megspeech.data.raw.serialised import NAME
from megspeech.evaluation.viz.media import Meg, Stimulus
from megspeech.evaluation.viz.sample import Sentences, Strata

SAMPLES = "samples.json"
MEG, AUDIO = "meg", "audio"
ID_WIDTH = 4
KEY = [Column.RUN, Column.ONSET]
"""What names one word across models' word tables: its run and its onset there."""


@dataclass(frozen=True)
class Window:
    """The MEG shown per word: the model's own `length`-sample window from `offset` after its onset, with `pad`
    samples of context either side, at `rate`."""

    rate: float
    length: int
    offset: int = 0
    pad: int = 0

    @property
    def samples(self) -> int:
        return self.length + 2 * self.pad

    def start(self, at: int) -> int:
        """The first sample shown for a word whose onset is sample `at`."""
        return at + self.offset - self.pad


@dataclass(frozen=True)
class SampleSet:
    """The picked words (`KEY` columns and an `id`), their `samples.json` records and the media those name."""

    keys: pd.DataFrame
    records: list[dict[str, object]]
    meg: dict[str, Float[np.ndarray, "c s"]] = field(default_factory=dict)
    audio: dict[str, tuple[int, Int[np.ndarray, " a"]]] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return self.keys["id"].tolist()

    def write(self, out: Path) -> Path:
        """`samples.json`, `meg/<id>.f32` and `audio/<id>.wav` under `out`."""
        out.mkdir(parents=True, exist_ok=True)
        for name, window in self.meg.items():
            Meg.write(out / MEG / f"{name}.f32", window)
        for name, (rate, sound) in self.audio.items():
            Stimulus.write(out / AUDIO / f"{name}.wav", rate, sound)
        (out / SAMPLES).write_text(json.dumps(self.records), encoding="utf-8")
        return out


@dataclass(frozen=True)
class Builder:
    """Builds a `SampleSet` from picked rows of a word table, the runs' MEG and their stimuli."""

    window: Window
    sentences: Sentences

    @staticmethod
    def identifiers(count: int) -> list[str]:
        width = max(ID_WIDTH, len(str(max(count - 1, 0))))
        return [f"{i:0{width}d}" for i in range(count)]

    def record(self, name: str, row: pd.Series, strata: dict[str, str], start: int, channels: int) -> dict[str, object]:
        """`row`'s `samples.json` entry; its `channels`-row MEG window starts at run sample `start`."""
        text, span = self.sentences.text(row)
        parts = NAME.search(f"{row[Column.RUN]}_")
        if parts is None:
            raise ValueError(row[Column.RUN])
        t0 = start / self.window.rate
        return {
            "id": name,
            "source": strata["source"],
            "subject": int(parts["subject"]),
            "session": int(parts["session"]),
            "text": text,
            "target": list(span),
            "label": PRIMARY.words[int(row[Column.TARGET])],
            "target_t": [float(row[Column.ONSET]), float(row[Column.FINISH])],
            "strata": strata,
            "meg": {
                "file": f"{MEG}/{name}.f32",
                "channels": channels,
                "times": self.window.samples,
                "sfreq": self.window.rate,
                "t0": t0,
            },
            "audio": None,
        }

    def build(
        self,
        rows: pd.DataFrame,
        lengths: Int[np.ndarray, " p"],
        signals: dict[str, Float[np.ndarray, "c t"]],
        stimuli: dict[str, Stimulus],
    ) -> SampleSet:
        """`rows` (with run-local onset sample `at`) as samples, strata cut over the pool's sentence `lengths`, MEG
        cut from each run's `signals`, audio from its `stimuli`."""
        names = self.identifiers(len(rows))
        strata = Strata.of(rows, self.sentences.lengths(rows), lengths)
        records: list[dict[str, object]] = []
        meg: dict[str, Float[np.ndarray, "c s"]] = {}
        audio: dict[str, tuple[int, Int[np.ndarray, " a"]]] = {}
        seconds = self.window.samples / self.window.rate
        keys = Sentences.keys(rows)
        for name, (_, row), stratum, key in zip(names, rows.iterrows(), strata, keys, strict=True):
            start = self.window.start(int(row["at"]))
            signal = Meg(signals[row[Column.RUN]], self.window.rate)
            meg[name] = Meg.zscored(signal.window(start, self.window.samples))
            record = self.record(name, row, stratum, start, meg[name].shape[0]) | {"sentence": key}
            clip = stimuli[row[Column.RUN]].clip(start / self.window.rate, seconds)
            if clip is not None:
                audio[name] = clip
                record["audio"] = {"file": f"{AUDIO}/{name}.wav", "sfreq": clip[0], "t0": start / self.window.rate}
            records.append(record)
        keys = rows[KEY].reset_index(drop=True).assign(id=names)
        return SampleSet(keys, records, meg, audio)
