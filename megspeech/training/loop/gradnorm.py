"""The gradient norms of a run's steps, summed on the device and read once per epoch."""

from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class GradNorms:
    """Each step's whole-model gradient norm (what `clip_grad_norm_` returns, before it clips) and that of the
    `encoder` parameters alone, kept on the device until `drained` so a step never syncs to the host."""

    encoder: tuple[nn.Parameter, ...] = ()
    rows: list[Tensor] = field(default_factory=list)

    def record(self, total: Float[Tensor, ""]) -> None:
        """Keep this step's `total` norm and its encoder's; call between `backward` and the optimiser step."""
        grads = [p.grad.norm() for p in self.encoder if p.grad is not None]
        encoder = torch.linalg.vector_norm(torch.stack([*grads, total.new_zeros(())]))
        self.rows.append(torch.stack([total.detach(), encoder]))

    def drained(self) -> dict[str, float]:
        """The mean and the largest of each norm over the steps since the last drain, which is then cleared."""
        if not self.rows:
            return {}
        seen = torch.stack(self.rows).float()
        self.rows.clear()
        mean, peak = seen.mean(0).tolist(), seen.amax(0).tolist()
        return {
            "total_mean": mean[0],
            "total_max": peak[0],
            "encoder_mean": mean[1],
            "encoder_max": peak[1],
        }
