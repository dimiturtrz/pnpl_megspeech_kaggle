import math
from typing import cast

import torch

from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.studies.gap import Locator
from megspeech.evaluation.studies.sourcefit import Source, SourceFit
from tests.unit.megspeech.evaluation.studies.gap import Stub, split

SUBJECTS = {Source.LIBRIBRAIN: 0, Source.MASC: 101, Source.APPLESEED: 201, Source.KYMATA: 301}


def locator() -> Locator:
    fitted = split()
    ids = torch.tensor([list(SUBJECTS.values())[i % 3] for i in range(len(fitted.y))])
    fitted.subject.copy_(ids)
    return Locator(cast(LoadedModel, Stub()), fitted, {"d": split()}, 0.0, 0)


def test_owns_partitions_subject_ids_by_source() -> None:
    ids = torch.tensor([*SUBJECTS.values(), 50])
    owned = {source: source.owns(ids).tolist() for source in Source}
    assert owned[Source.MASC] == [False, True, False, False, False]
    assert sum(sum(mask) for mask in owned.values()) == len(SUBJECTS)


def test_of_scores_each_source_and_leaves_an_absent_one_nan() -> None:
    measured = SourceFit.of(locator())
    assert measured.fit[Source.MASC] == 1.0
    assert math.isnan(measured.fit[Source.KYMATA])
    assert measured.words[Source.KYMATA] == 0
    assert sum(measured.words.values()) > 0


def test_lines_report_every_source_and_validation() -> None:
    lines = SourceFit.of(locator()).lines()
    assert len(lines) == len(Source) + 1
    assert lines[-1].startswith("val macro")
