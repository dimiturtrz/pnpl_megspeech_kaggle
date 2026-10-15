from typing import cast

import pytest
import torch

from megspeech.objective.auxiliary.aligned.acoustic import CHUNK, MIX, PRIOR, TAPS, Acoustic, AcousticTerm
from megspeech.objective.targets.loudness import LENGTH
from megspeech.objective.targets.speech import Speech
from megspeech.objective.term import Reading

WIDTH = 6
STEPS = 25
GENERATOR = torch.Generator().manual_seed(0)
TARGET = torch.tensor([[0, 1, -100], [2, -1, -100]])


def term() -> AcousticTerm:
    return AcousticTerm(2.0, Speech(torch.tensor([0, -1, 1, 2]), torch.randn(3, LENGTH, generator=GENERATOR).half()))


def test_forward() -> None:
    """Every window's envelope at the envelope's frames, whatever the encoder's own time resolution; the lag starts
    on its prior tap, so a constant feature channel reads out as that level."""
    head = Acoustic(WIDTH, LENGTH)
    for steps in (STEPS, LENGTH, 2 * LENGTH):
        assert head.forward(torch.randn(2, 3, WIDTH, steps, generator=GENERATOR)).shape == (2, 3, LENGTH)
    assert Acoustic(WIDTH, 3 * CHUNK).forward(torch.randn(2, WIDTH, STEPS, generator=GENERATOR)).shape == (2, 3 * CHUNK)
    with pytest.raises(ValueError, match="no whole number"):
        Acoustic(WIDTH, CHUNK + 1)
    assert head.lag.weight.shape == (1, MIX, TAPS)
    assert torch.count_nonzero(head.lag.weight).item() == MIX
    assert head.lag.weight[0, :, PRIOR].ne(0).all()


def test_head() -> None:
    assert isinstance(term().head(WIDTH), Acoustic)
    assert AcousticTerm.timed


def test_unweighted() -> None:
    """Positive over the words with a target, differentiable into the readout and the lag, 0 when none has one; the
    contrast is over the window's `CHUNK`-frame chunks, however many its cut holds."""
    aligned = term()
    head = aligned.head(WIDTH)
    output = head(torch.randn(2, 3, WIDTH, STEPS, generator=GENERATOR))
    value = aligned.unweighted(aligned.read(head, output, TARGET))
    direct = aligned.unweighted(Reading(head, output, torch.tensor([[0, 1, -1], [2, -1, -1]]), torch.ones(2, 3)))
    assert direct.item() > 0
    assert value.item() > 0
    value.backward()
    assert cast("Acoustic", head).project.weight.grad is not None
    assert cast("Acoustic", head).lag.weight.grad is not None
    assert aligned.unweighted(aligned.read(head, output, torch.full((2, 3), -1))).item() == 0.0
    wide = AcousticTerm(1.0, Speech(torch.tensor([0, 1, 2]), torch.randn(3, 3 * CHUNK, generator=GENERATOR)))
    widened = wide.head(WIDTH)
    features = torch.randn(1, 3, WIDTH, STEPS, generator=GENERATOR)
    rows = torch.tensor([[0, 1, -1]])
    assert wide.unweighted(Reading(widened, widened(features), rows, torch.ones(1, 3))).item() > 0


def test_learns_a_matched_envelope() -> None:
    """Predictions equal to the truth score below shuffled ones: the contrast prefers the right window."""
    aligned = term()
    head = aligned.head(WIDTH)
    truth = aligned.speech.vectors.float()[torch.tensor([[0, 1, 0], [2, 0, 0]])]
    wrong = truth.roll(1, dims=0)
    right_loss = aligned.unweighted(aligned.read(head, truth, TARGET))
    wrong_loss = aligned.unweighted(aligned.read(head, wrong, TARGET))
    assert right_loss.item() < wrong_loss.item()


def test_fits() -> None:
    """Features that carry the envelope `PRIOR` frames late are read back by the readout and lag: the held-in
    prediction correlates with the truth after a few steps of the contrast."""
    n = 12
    smooth = torch.nn.functional.avg_pool1d(torch.randn(n, 1, LENGTH + 8, generator=GENERATOR), 9, 1).squeeze(1)
    envelope = (smooth - smooth.mean(-1, keepdim=True)) / smooth.std(-1, keepdim=True)
    features = torch.nn.functional.pad(envelope, (PRIOR, 0))[:, None, :LENGTH].expand(n, WIDTH, LENGTH)
    fitted = AcousticTerm(1.0, Speech(torch.arange(n), envelope.half()))
    head = fitted.head(WIDTH)
    optimiser = torch.optim.Adam(head.parameters(), lr=0.02)
    target = torch.arange(n)[None]
    for _ in range(60):
        optimiser.zero_grad()
        fitted.unweighted(fitted.read(head, head(features[None]), target)).backward()
        optimiser.step()
    predicted = head(features[None])[0]
    r = torch.corrcoef(torch.stack([predicted.flatten(), envelope.flatten()]))[0, 1]
    assert r.item() > 0.5
