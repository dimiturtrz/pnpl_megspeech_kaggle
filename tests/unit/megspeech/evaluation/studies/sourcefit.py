import math
from typing import cast

import pytest
import torch

from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.studies.gap import Locator
from megspeech.evaluation.studies.sourcefit import UNSEEN_SUBJECT, Source, SourceFit, Variant
from megspeech.model.context.timing import KAGGLE_GAP
from tests.unit.megspeech.evaluation.studies.gap import Stub, split

SPAN = 4
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
    assert measured.fit[Variant.REAL][Source.MASC] == 1.0
    assert math.isnan(measured.fit[Variant.MEG][Source.KYMATA])
    assert measured.words[Source.KYMATA] == 0
    assert sum(measured.words.values()) > 0


def test_lines_report_every_source_and_validation() -> None:
    lines = SourceFit.of(locator()).lines()
    assert len(lines) == len(Source) + 2
    assert lines[-1].startswith("val macro")


def test_evenly_spaces_valid_words_from_zero() -> None:
    evened = Variant.evenly(split())
    assert evened.onset[0, :SPAN].tolist() == pytest.approx([0.0, KAGGLE_GAP, 2 * KAGGLE_GAP, 3 * KAGGLE_GAP])


def test_regapped_keeps_each_spans_gaps_as_a_multiset() -> None:
    base = split()
    moved = Variant.regapped(base, 0)
    before = torch.diff(base.onset[:, :SPAN]).sort(dim=1).values
    assert torch.allclose(torch.diff(moved.onset[:, :SPAN]).sort(dim=1).values, before, atol=1e-5)
    assert moved.onset[:, 0].tolist() == base.onset[:, 0].tolist()


def test_anonymous_sets_every_subject_unseen() -> None:
    assert Variant.anonymous(split()).subject.unique().tolist() == [UNSEEN_SUBJECT]


def test_apply_real_is_identity_and_others_shuffle() -> None:
    base = split()
    assert Variant.REAL.apply(base, 0) is base
    assert Variant.BOTH.apply(base, 0).subject.unique().tolist() == [UNSEEN_SUBJECT]
