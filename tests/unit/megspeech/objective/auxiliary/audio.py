import math

import torch

from megspeech.objective.auxiliary.audio import DIM, TEMPERATURE, AudioTerm, Listening
from megspeech.objective.targets.loudness import LENGTH
from megspeech.objective.targets.speech import Speech
from megspeech.objective.term import Reading

GENERATOR = torch.Generator().manual_seed(0)
WIDTH = 8
BATCH = 64
TIMES = 40


def distinct(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.arange(n), torch.arange(n), torch.ones(n)


def test_forward() -> None:
    listening = Listening(WIDTH)
    assert listening.forward(torch.randn(3, 2, WIDTH, TIMES, generator=GENERATOR)).shape == (3, 2, DIM)
    assert math.isclose(listening.scale.exp().item(), 1 / TEMPERATURE, rel_tol=1e-5)


def test_tower() -> None:
    assert Listening.tower(1)(torch.zeros(4, 1, LENGTH)).shape == (4, DIM)


def test_hear() -> None:
    envelope = torch.randn(5, LENGTH, generator=GENERATOR).half()
    assert Listening(WIDTH).hear(envelope).shape == (5, DIM)


def test_loss() -> None:
    listening = Listening(WIDTH)
    meg = torch.randn(BATCH, DIM, generator=GENERATOR)
    audio = torch.randn(BATCH, DIM, generator=GENERATOR)
    kinds, spans, rows = distinct(BATCH)
    random = listening.loss(meg, audio, kinds, spans, rows)
    assert torch.isclose(random, listening.loss(audio, meg, kinds, spans, rows))
    assert abs(random.item() - math.log(BATCH)) < 0.6
    assert listening.loss(meg, meg, kinds, spans, rows) < 0.01
    lone = listening.loss(meg, audio, kinds, torch.zeros(BATCH, dtype=torch.long), rows)
    assert lone.abs() < 1e-6
    assert listening.loss(meg, audio, torch.full((BATCH,), -1), spans, rows) == 0


def test_loss_trains_the_scale_and_both_towers() -> None:
    listening = Listening(WIDTH)
    embedded = torch.randn(BATCH, WIDTH, TIMES, generator=GENERATOR)
    envelope = torch.randn(BATCH, LENGTH, generator=GENERATOR)
    optimiser = torch.optim.Adam(listening.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        optimiser.zero_grad()
        loss = listening.loss(listening(embedded), listening.hear(envelope), *distinct(BATCH))
        loss.backward()
        assert listening.scale.grad is not None
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_head() -> None:
    term = AudioTerm(1.0, Speech(torch.tensor([0, -1]), torch.zeros(1, LENGTH)))
    assert isinstance(term.head(WIDTH), Listening)


def test_unweighted() -> None:
    term = AudioTerm(1.0, Speech(torch.tensor([0, 1, 2]), torch.randn(3, LENGTH, generator=GENERATOR)))
    head = term.head(WIDTH)
    output = head(torch.randn(2, 2, WIDTH, TIMES, generator=GENERATOR))
    target = torch.tensor([[0, 1], [2, -1]])
    loss = term.unweighted(Reading(head, output, target, torch.ones(2, 2)))
    loss.backward()
    assert loss.item() > 0
    assert all(p.grad is not None for p in head.parameters())
    assert term.unweighted(Reading(head, output, torch.tensor([[0, 1], [-1, -1]]), torch.ones(2, 2))) == 0
