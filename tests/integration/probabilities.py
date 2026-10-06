from dataclasses import replace
from pathlib import Path

import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.evaluation.studies.uncertainty import Members, Saved
from megspeech.evaluation.text import RULER
from tests.unit.megspeech.evaluation.loaded import Favourite, loaded, runs
from tests.unit.megspeech.evaluation.studies.uncertainty import Pair, Reversed


def test_saved_probabilities_feed_uncertainty(tmp_path: Path) -> None:
    """Probabilities a training run saves, read back by the uncertainty study, give the tables it reads live."""
    domains, _ = loaded(tmp_path).domains({RULER: runs(tmp_path)[:1]})
    for name, net in (("one", Favourite()), ("two", Reversed())):
        (tmp_path / name).mkdir()
        Probabilities.of(loaded(tmp_path).evaluator, net, domains, torch.zeros(VOCAB_SIZE), 1.0).save(
            tmp_path / name / PROBS, "val"
        )
    frame, hits = Saved.load([tmp_path / "one", tmp_path / "two"], "val").partition()
    live, live_hits = Members(replace(loaded(tmp_path), model=Pair())).partition(runs(tmp_path)[:1])
    assert torch.equal(hits[RULER], live_hits[RULER])
    assert frame.equals(live)
