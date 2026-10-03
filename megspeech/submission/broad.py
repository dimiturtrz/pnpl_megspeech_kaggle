"""Score a trained model on the local Broad ruler, then write its Broad submission at the ruler's adjustment.

Every holdout subject's rows are predicted from that subject's own MEG; a zero-shot subject (33-39) is read
through the `--unseen` fallback of a model with a subject layer, and with `--reading unseen` every subject is.

    uv run python -m megspeech.submission.broad sent40-s7 [--reading unseen]   # models/<name>/broad_submission.csv
"""

import logging

import torch

from core.paths import DataRoot
from megspeech.data.cache.holdout import Track
from megspeech.data.raw.runs import Run
from megspeech.evaluation.broad import BroadRuler, Options
from megspeech.submission.submission import Submitter

LOG = logging.getLogger(__name__)

SUBMISSION = "broad_submission.csv"


def main() -> None:
    """Log the Broad ruler and write models/<name>/broad_submission.csv at the adjustment it selects."""
    options = Options.parse(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    ruler = options.ruler(root, torch.device("cuda"))
    macro = BroadRuler.report(ruler.scores(Run.discover(root.raw())))
    pooling = ruler.pooling
    submitter = Submitter(pooling.evaluator, pooling.data, Track.BROAD, ruler.reading)
    path = submitter.submit(
        pooling.model, root, pooling.log_prior, macro.tau, root.models() / options.name / SUBMISSION
    )
    LOG.info("wrote %s", path)


if __name__ == "__main__":
    main()
