"""A multinomial logistic readout fit on the GPU, a drop-in sklearn classifier for the part-of-speech probes."""

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.base import BaseEstimator, ClassifierMixin

from megspeech.device import Device

MAX_ITER = 300
"""L-BFGS iterations a logistic readout may take."""
TOLERANCE = 1e-6
"""Gradient infinity-norm at which a logistic readout counts as converged."""


class TorchLogistic(ClassifierMixin, BaseEstimator):
    """Multinomial logistic regression fit by full-batch L-BFGS on the GPU: sklearn's `LogisticRegression(C)` objective
    (summed cross-entropy plus ||W||^2 / 2C, intercept free) over features standardised on the fit rows."""

    def __init__(
        self, c: float = 1.0, *, standardize: bool = True, max_iter: int = MAX_ITER, balanced: bool = False
    ) -> None:
        self.c = c
        self.standardize = standardize
        self.max_iter = max_iter
        self.balanced = balanced

    def fit(self, x: Float[np.ndarray, "n d"], y: Int[np.ndarray, " n"]) -> "TorchLogistic":
        """The weights minimising the penalised loss of `y` (any integer labels) on `x`."""
        self.classes_, targets = np.unique(y, return_inverse=True)
        device = Device.best().as_torch()
        features = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.mean_ = features.mean(0) if self.standardize else torch.zeros(features.shape[1], device=device)
        scale = features.std(0, correction=0) if self.standardize else torch.ones(features.shape[1], device=device)
        self.scale_ = torch.where(scale > 0, scale, torch.ones_like(scale))
        features = (features - self.mean_) / self.scale_
        labels = torch.as_tensor(targets, device=device)
        rows = len(labels)
        counts = torch.bincount(labels, minlength=len(self.classes_)).float()
        emphasis = rows / (len(self.classes_) * counts) if self.balanced else None
        weight = torch.zeros(features.shape[1], len(self.classes_), device=device, requires_grad=True)
        bias = torch.zeros(len(self.classes_), device=device, requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [weight, bias], max_iter=self.max_iter, tolerance_grad=TOLERANCE, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = F.cross_entropy(features @ weight + bias, labels, weight=emphasis) + 0.5 * weight.square().sum() / (
                self.c * rows
            )
            loss.backward()
            return loss

        optimiser.step(closure)
        self.weight_, self.bias_ = weight.detach(), bias.detach()
        return self

    def predict_log_proba(self, x: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n k"]:
        """Each row's log-probability of each of `classes_`."""
        features = torch.as_tensor(x, dtype=torch.float32, device=self.weight_.device)
        logits = ((features - self.mean_) / self.scale_) @ self.weight_ + self.bias_
        return logits.log_softmax(1).double().cpu().numpy()

    def predict_proba(self, x: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n k"]:
        """Each row's probability of each of `classes_`."""
        return np.exp(self.predict_log_proba(x))

    def predict(self, x: Float[np.ndarray, "n d"]) -> Int[np.ndarray, " n"]:
        """Each row's most probable of `classes_`."""
        return self.classes_[self.predict_log_proba(x).argmax(1)]
