"""The sampler training draws spans by, built in one place from the optimiser's config."""

from megspeech.data.batching.split import Split
from megspeech.training.config import OptimConfig
from megspeech.training.sampling.curriculum import Annealed, Curriculum, Prioritised, Robust


class Sampler:
    """A base distribution (`OptimConfig.sampler`) under an ordered chain of modifiers: annealed, then tilted by
    stratum (`robust`), then by span (`priority`, `visibility`). Each wraps the one before and forwards what training
    observes to it, so the loss-driven tilts compose."""

    @staticmethod
    def scheduled(optim: OptimConfig, train: Split) -> Curriculum:
        """The chain's loss-free head, the base annealed by `curriculum`: what sizes a batch's word rows."""
        return Annealed.of(optim.curriculum, optim.sampler.of(train, optim.stratified))

    @staticmethod
    def of(optim: OptimConfig, train: Split) -> Curriculum:
        """The whole chain over `train`: `scheduled`, then `robust`, then `priority` and `visibility`."""
        robust = Robust.of(Sampler.scheduled(optim, train), train, optim.robust)
        return Prioritised.of(robust, train, optim.priority, optim.visibility, train.reads())
