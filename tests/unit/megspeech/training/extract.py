import torch

from megspeech.data.windows import Windows
from megspeech.model.megxl import LATENT
from megspeech.model.scaler import Scaler
from megspeech.training.extract import Extractor
from tests.unit.megspeech.model.megxl_words import words

MEG = torch.arange(5 * 2 * 8, dtype=torch.float32).reshape(5, 2, 8).sin()


def extractor(chunk: int) -> Extractor:
    torch.manual_seed(0)
    return Extractor(words().eval(), Scaler(MEG), chunk)


def test_call():
    windows = Windows.stacked(MEG)
    whole = extractor(5)(windows)
    assert whole.shape == (5, 2 * LATENT)
    assert torch.allclose(extractor(2)(windows).float(), whole.float())


def test_pooled():
    pooled = extractor(5).pooled(MEG)
    assert pooled.dtype == torch.bfloat16
    assert pooled.shape == (5, 2 * LATENT)
