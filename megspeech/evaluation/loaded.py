"""A trained model, or several averaged as one, loaded to score as their training scored them."""

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.config import ADAPTED
from megspeech.data.raw.runs import DEEP, Run
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Head, Score

ADAPT_ON_ELEVEN = 11


@dataclass(frozen=True, kw_only=True)
class LoadedModel(Reader):
    """Trained runs averaged as one model, with the prior and evaluator their training scored them by."""

    model: nn.Module
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]
    names: tuple[str, ...] = ()

    def context(self, split: Split) -> Float[Tensor, "n v"]:
        return self.evaluator.head(self.evaluator.logits(self.model, split))

    def labelled(self, split: Split) -> Score:
        """The context head on every labelled word of `split`."""
        labelled = split.labelled()
        return self.score(self.context(split)[labelled], split.y[labelled])

    def score(self, context: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> Score:
        """Context-head log-probabilities `context` against labels `y`, scored by this model's evaluator and prior."""
        return self.evaluator.evaluate(context, y, self.log_prior)

    @property
    def adapted_on(self) -> int:
        """The Sherlock1 session the model trained on as `DataConfig.adapt`; 0 for none."""
        return self.data.adapt

    def broad(self, runs: list[Run]) -> list[Run]:
        """Subjects 1-32's Sherlock1 run the model did not adapt on: session 11, or 12 when it adapts on 11."""
        held = 12 if self.data.adapt == ADAPT_ON_ELEVEN else 11
        return sorted((r for r in runs if r.subject != DEEP and r.key == (ADAPTED, held, 1)), key=lambda r: r.subject)

    def run_context(self, run: Run) -> tuple[Split, Float[Tensor, "n v"]]:
        """`run`'s split, read from its own MEG, and the context head's log-probabilities for its words."""
        _, split = self.splits([run])[run]
        return split, self.context(split)

    def unseen(self, run: Run) -> Score:
        """The context head on `run`'s unseen-span words, read from its own MEG."""
        split, context = self.run_context(run)
        return self.evaluator.unseen(context, split, self.log_prior)

    def domain_logits(self, runs: list[Run]) -> tuple[dict[str, Split], dict[str, dict[Head, Float[Tensor, "n v"]]]]:
        """Every domain of `runs` in the scored partition as the model reads it, and both heads' log-probabilities."""
        domains, _ = self.domains(self.ruled(runs))
        return domains, {name: self.evaluator.logits(self.model, split) for name, split in domains.items()}

    def report(self, runs: list[Run]) -> Epoch:
        """Every domain of `runs` in the scored partition, scored alone, and the macro that selects."""
        domains, logits = self.domain_logits(runs)
        return self.evaluator.summarised(logits, domains, self.log_prior, 0, 0.0)

    def graded(self, runs: list[Run], tau: float) -> Graded:
        """Every counted domain of `runs` in the scored partition: its unseen-span words' hits at `tau`."""
        domains, logits = self.domain_logits(runs)
        return self.evaluator.graded_logits(logits, domains, self.evaluator.calibration(self.log_prior, tau))
