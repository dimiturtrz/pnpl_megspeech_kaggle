"""Subjects grouped by the source they were recorded under."""

import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.raw.runs import SOURCES


class Sources:
    """The subjects grouped by recording source."""

    @staticmethod
    def means(weight: Float[Tensor, "s c c"]) -> Float[Tensor, "s c c"]:
        """Each subject's matrix replaced by the mean over the subjects of its source; a subject in no `SOURCES`
        range is its own source."""
        owner = {subject: group for group, ids in enumerate(SOURCES) for subject in ids}
        groups = torch.tensor([owner.get(i, len(SOURCES) + i) for i in range(len(weight))], device=weight.device)
        same = (groups[:, None] == groups[None]).to(weight.dtype)
        return torch.einsum("ij,jcd->icd", same / same.sum(dim=1, keepdim=True), weight)
