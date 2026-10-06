"""The sampler training draws spans by, built in one place from its config."""

from collections.abc import Mapping
from dataclasses import dataclass

from megspeech.data.batching.split import Split
from megspeech.training.sampling.curriculum import Annealed, Base, Curriculum, Prioritised, Robust


@dataclass(frozen=True)
class SamplerConfig:
    base: Base = Base.NATURAL
    """The distribution spans are drawn from before any tilt (`megspeech.training.sampling.curriculum.Base`): `natural`
    draws each span at its words' run weight (`DataConfig.read_weight`, `narrator_weight`, `repeat_power`, all 1 unless
    set, so uniform by default), `uniform` at 1 whatever those say, `stratified` balanced across source, subject, span
    length and word rate by `alpha`."""
    alpha: float = 1.0
    """Exponent of `base = stratified`: each span draws at `count ** -alpha` of its stratum's span count, so 1 gives
    every stratum an equal share and 0 draws every span alike."""
    curriculum: float = 0.0
    """The Deep subject's share of the spans drawn at the last epoch, annealed linearly from an equal share per
    subject at the first (`megspeech.training.sampling.curriculum`); 0 draws every subject's spans at the sampler's
    own weight. Only meaningful when `DataConfig.subjects` trains on more than the Deep subject."""
    robust: float = 0.0
    """Power of the tilt toward the strata (subject x span length x word rate) whose training loss ran highest last
    epoch (`megspeech.training.sampling.curriculum.Robust`): a stratum at twice the mean loss draws `2 ** robust` times
    its weight, and none draws below its own; 0 draws at the curriculum's weights alone."""
    priority: float = 0.0
    """Power of each span's difficulty (the EMA of its training loss) over the mean, tilting its draw weight
    (`megspeech.training.sampling.curriculum.Prioritised`); an unseen span counts at the highest difficulty so far,
    and none draws below a quarter or above four times its weight. Negative is self-paced, drawing the hard least
    (noisy trials may be unlearnable). 0 turns the tilt off; composes with `robust`."""
    visibility: float = 0.0
    """Exponent `beta` of a span's draw weight `(1 + seen) ** -beta` in the times training has drawn it, so unseen spans
    draw first and the corpus is rotated through; 0 turns it off; composes with `robust`."""

    @classmethod
    def migrated(cls, record: Mapping[str, object]) -> dict[str, object]:
        """A saved run config with its sampler knobs, flat under `optim` (or at the root, before the tree), moved into
        `sampler`, the retired `sampler` and `stratified` fields renamed `base` and `alpha`; a record that already has a
        `sampler` node passes unchanged."""
        renamed = {"sampler": "base", "stratified": "alpha"}
        knobs = set(cls.__dataclass_fields__) | set(renamed)
        if isinstance(record.get("sampler"), Mapping):
            return dict(record)
        optim = record.get("optim")
        nested: Mapping[str, object] = optim if isinstance(optim, Mapping) else {}
        moved = {k: v for source in (record, nested) for k, v in source.items() if k in knobs}
        rest = {k: v for k, v in record.items() if k not in knobs and k != "optim"}
        kept: dict[str, object] = {"optim": {k: v for k, v in nested.items() if k not in knobs}} if nested else {}
        return rest | kept | {"sampler": {renamed.get(k, k): v for k, v in moved.items()}}


class Sampler:
    """A base distribution (`SamplerConfig.base`) under an ordered chain of modifiers: annealed, then tilted by
    stratum (`robust`), then by span (`priority`, `visibility`). Each wraps the one before and forwards what training
    observes to it, so the loss-driven tilts compose."""

    @staticmethod
    def scheduled(config: SamplerConfig, train: Split) -> Curriculum:
        """The chain's loss-free head, the base annealed by `curriculum`: what sizes a batch's word rows."""
        return Annealed.of(config.curriculum, config.base.of(train, config.alpha))

    @staticmethod
    def of(config: SamplerConfig, train: Split) -> Curriculum:
        """The whole chain over `train`: `scheduled`, then `robust`, then `priority` and `visibility`."""
        robust = Robust.of(Sampler.scheduled(config, train), train, config.robust)
        return Prioritised.of(robust, train, config.priority, config.visibility, train.reads())
