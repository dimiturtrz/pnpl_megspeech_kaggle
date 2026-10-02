import torch

from megspeech.data.windows import Windows
from megspeech.model.megxl import LATENT
from megspeech.model.scaler import Scaler
from megspeech.training.extract import Averaged, Extractor, Pooled, Read, Resolved
from tests.unit.megspeech.model.megxl_words import words

MEG = torch.arange(5 * 2 * 8, dtype=torch.float32).reshape(5, 2, 8).sin()
COMPONENTS = 3


def extractor(chunk: int = 5) -> Extractor:
    torch.manual_seed(0)
    encoder = words().eval()
    return Extractor(encoder, Scaler(MEG), Pooled(encoder), chunk)


def test_call():
    windows = Windows.stacked(MEG)
    whole = extractor()(windows)
    assert whole.shape == (5, 2 * LATENT, 1)
    assert whole.dtype == torch.bfloat16
    assert torch.allclose(extractor(2)(windows).float(), whole.float())
    averaged = Averaged(extractor().words)
    assert averaged(MEG).shape == (5, 2, LATENT)
    fitted = Resolved.fit(averaged, averaged(MEG), COMPONENTS)
    assert fitted(MEG).shape == (5, 2, COMPONENTS)


def test_scaled():
    base = extractor()
    assert torch.equal(base.scaled(MEG), base.scaler(MEG))


def test_fit():
    sample = torch.randn(400, 2, LATENT, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    fitted = Resolved.fit(Averaged(extractor().words), sample, COMPONENTS)
    projected = (sample.flatten(0, 1) - fitted.mean) @ fitted.basis
    assert torch.allclose(projected.std(0), torch.ones(COMPONENTS), atol=0.05)
    assert torch.allclose(fitted.mean, sample.flatten(0, 1).mean(0))


def test_reader():
    base = extractor()
    sample = Windows.stacked(torch.randn(20, 2, 8, generator=torch.Generator().manual_seed(0)))
    assert isinstance(Read.POOLED.reader(base, sample), Pooled)
    resolved = Read.RESOLVED.reader(base, sample)
    assert isinstance(resolved, Resolved)
    assert resolved.basis.shape[0] == LATENT
