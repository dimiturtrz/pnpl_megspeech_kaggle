"""How a training run optimises: its schedule, its losses' weights and the groups of knobs each consumer owns."""

import math
from dataclasses import dataclass

from megspeech.training.adversary import AdversaryConfig
from megspeech.training.auxiliary import AuxiliaryConfig
from megspeech.training.teacher import TeacherConfig
from megspeech.training.transfer import TransferConfig


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    clip: float = math.inf
    seed: int = 0
    ema: float = 0.0
    """Decay of the weight average scored and kept; 0 keeps the raw weights. At 0.999 the average spans ~1000
    steps, about half an epoch of the ~1750 a sentence-path epoch takes."""
    heads: float = 1.0
    """Weight of the two 50-way heads' losses (context and word). 0 trains an expert (megspeech-6ow.3): the encoder
    learns from the auxiliary terms alone, never the 50 words, and the run keeps its last epoch, since the heads it
    would be selected on are untrained."""
    auxiliaries: AuxiliaryConfig = AuxiliaryConfig()
    adversarial: AdversaryConfig = AdversaryConfig()
    distillation: TeacherConfig = TeacherConfig()
    transfer: TransferConfig = TransferConfig()
    mi: float = 0.0
    """Weight of the OVMI auxiliary (`megspeech.objective.information`): minus the mutual information in bits between
    the word and the context head's softmax over each batch's labelled words, the confusion rows weighted by the
    training prior as `Ruler.information` weighs them. Maximising it rewards predictions both confident and spread
    over the 50 words; 0 adds no term."""
    timing: str = ""
    """A timing expert (`megspeech.training.cli.timing`) whose head is loaded into the model's `rhythm` head and frozen,
    its logits added to the context head's in training and in every read after: a product of experts in which the
    MEG path learns only what timing leaves (megspeech-7gf.100); empty trains no such expert."""
    balance: float = 0.0
    """Class-weight exponent of the cross-entropies: each class weighs `count ** -balance` over the training labels,
    mean 1, so 0.5 is 1/sqrt(count) (the PNPL 2025 phoneme winner's). The metric is balanced accuracy, and the
    ruler's `tau` already corrects the class prior after training (Menon et al. 2021 logit adjustment); balancing
    during training overlaps it, so an arm asks whether it adds beyond `tau`. SigLIP's context loss is unweighted."""
    topk: float = 0.0
    """Share of the smooth top-k loss (Berrada et al. 2018, k=10) blended into both 50-way heads' losses, over the same
    logits and `balance` class weights (`TopKMix`); 0 trains their own losses alone."""
    topk_temperature: float = 1.0
    """The top-k term's smoothing in raw logits: 1 is the cross-entropy's own, the paper's tau 1 is 10."""
    topk_margin: float = 1.0
    """The raw-logit gap the label must hold over the 10th competitor before its top-k term is ~0."""
    anchor: float = 1.0
    """Share of each subject matrix's deviation from the Deep subject's that the run's decay leaves by its end, were its
    gradient never to restore it (`megspeech.training.regularise.weights.anchor`); 1 decays the matrices toward 0 like
    every other weight. The Deep matrix is what a zero-shot subject is read through."""
    pull: float = 1.0
    """Share of each encoder weight's drift from where the run started (the `stem`'s) that a decoupled pull toward it
    leaves by the run's end, were its gradient never to restore it (`megspeech.training.regularise.weights.pull`,
    L2-SP); 1 never pulls. The subject matrices are the `anchor`'s."""
    anchor_toward: str = "deep"
    """The matrix `anchor` pulls toward, an `Unseen` value: `deep` the Deep subject's, `mean` the mean of every trained
    subject's matrix, `source` the mean over the subjects of its own recording source, `identity`."""
    last: bool = False
    """Keep the last epoch, not the best on validation: a stage-1 encoder trained to the end of its schedule, its word
    head converged where the context head selected at its best epoch reads an immature one (megspeech-7gf.117)."""
    fused_lr: tuple[float, ...] = ()
    """Each `context.fused` encoder's learning rate as a share of `lr`, its decay alike, one per encoder; 0 freezes it.
    Empty trains them all at `lr`."""
    isda: float = 0.0
    """Final strength `lambda` of the word head's implicit semantic data augmentation (`megspeech.objective.isda`,
    Wang et al. 2019), raised linearly as `isda * epoch / epochs`; 0 augments nothing."""
    sharpness: float = 0.0
    """SAM radius (`megspeech.training.regularise.weights.sharpness`): each step's gradient is taken `sharpness` along
    the normalised gradient, at twice a step's compute; 0 descends plainly."""
    adaptive: bool = False
    """Make the `sharpness` ball scale-adaptive (ASAM): each weight is perturbed in proportion to its magnitude."""
    compiled: bool = True
    """Compile the model's forward and backward into CUDA graphs; batches have fixed shapes, so they replay."""
    pretrained: float = 0.1
    """Retired with the end-to-end MEG-XL encoder it scaled; kept so the configs saved with it still load."""
