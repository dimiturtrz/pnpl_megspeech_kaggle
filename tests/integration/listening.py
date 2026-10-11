import torch

from megspeech.data.batching.split import SpanInput
from megspeech.model.auxiliary import WithAuxiliary
from megspeech.objective.auxiliary.audio import AudioTerm, Listening
from megspeech.objective.targets.loudness import LENGTH
from megspeech.objective.targets.speech import Speech
from megspeech.training.auxiliary import Joint
from tests.unit.megspeech.model.auxiliary import ONSET, SLOT, net, x

GENERATOR = torch.Generator().manual_seed(0)
ROWS = torch.tensor([0, 1, 2, -1])


def identity(model: WithAuxiliary) -> WithAuxiliary:
    return model


def test_label_free_pretrain_steps_train_the_encoder() -> None:
    """The audio term alone, wrapped by `Joint` around a sentence model, trains its encoder on a span batch: the
    pretrain an `--optim.heads 0 --optim.audio 1` run takes, on synthetic MEG and envelopes, on CPU."""
    term = AudioTerm(1.0, Speech(ROWS[:3], torch.randn(3, LENGTH, generator=GENERATOR)))
    model = Joint((term,)).trained(net(), identity).eval()
    assert isinstance(model.heads[0], Listening)
    inputs = SpanInput(x(), ONSET, SLOT, torch.zeros(3, dtype=torch.long))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        optimiser.zero_grad()
        loss = term.loss(model.heads[0], model(inputs)[2], ROWS[SLOT])
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert any(p.grad is not None and p.grad.any() for p in model.net.words.parameters())
    assert losses[-1] < losses[0]
