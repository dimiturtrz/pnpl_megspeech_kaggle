# Deep: a span refiner over word-head beliefs gains only when the neighbours are true (2026-10-10)

Study: `python -m megspeech.evaluation.studies lm.refine` (dev `c568c18`). Issue: megspeech-7gf.145.1. Single seed.

## Setup
- A small transformer reads a span's word beliefs (50-way) and re-predicts each word.
- It is pretrained on WikiText-103 spans, with beliefs simulated from MEG confusion. 229 articles that name the
  test-adjacent texts were dropped.
- It is fine-tuned on out-of-fold beliefs from `peerbase-s0`: kf5t folds 0-3 train it, fold 4 picks the checkpoint.
- It is scored with lmsoft's 2-fold VAL crossfit and transfers to the local test (LT).
- Base `m11xsp7hub6d2v3`: VAL .6537, LT .6970.
- An untrained refiner scores VAL −.009 [−.019, −.000]. That is the crossfit's own cost on this ruler.

## Result

| arm | realistic VAL Δ | realistic LT Δ | oracle VAL Δ | oracle LT Δ |
|---|---|---|---|---|
| pretrain + fine-tune (p1) | −.009 | −.020 | +.120 | +.128 |
| MEG-only fine-tune (p1) | −.008 | −.018 | +.080 | +.096 |
| pretrain only (held) | −.010 | +.012 | +.035 | +.047 |
| pretrain + fine-tune (held) | −.008 | −.008 | +.092 | +.123 |

In the realistic arms the neighbours are the model's own beliefs. In the oracle arms the true neighbour words are
shown and only the target is hidden.

## Reading
- **The harness works.** The oracle beats the lmbound backoff/pmi oracle (+.044) and matches pmi-right (+.091).
  So the refiner, the ruler and the crossfit path can all show a gain when one exists.
- **The realistic arms are at the crossfit floor.** None clears the +.015 bar, and LT is mostly negative.
- **Context only helps here if the neighbours are decoded right.** At about 65% top-10 and much lower top-1, the
  neighbour beliefs are too noisy to carry the context a text LM would add. The base context head already uses
  what context is recoverable.
- **Fine-tuning on real data barely moves held-out loss** (2.409 → ~2.37). The pretrained model alone is worse on
  real beliefs (2.70-2.78). The simulated confusion does not match the real belief distribution.

## What this does NOT rule out
- A belief sampler with correlated errors. Independent per-word draws ignore that the context head already used
  context.
- A refiner trained on the base's own out-of-fold beliefs. No m11x OOF exists, and the peerbase → m11x shift likely
  explains the LT −.01 to −.02.
- A pretraining corpus whose label mix matches the MEG text: 33% in-vocabulary words against about 40%.
- Iterating the refiner, or using it as a reranker of top-k joint span hypotheses instead of a per-word refiner.
