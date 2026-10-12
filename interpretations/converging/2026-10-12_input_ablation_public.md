# Input ablation on the public test — what the score is made of

Single members, one input channel removed at inference by a forward hook (no retraining). Tau re-picked on VAL per
arm. Beads `megspeech-7gf.105.1` (Deep), `megspeech-7gf.147.17` (Broad).

## Deep — member `ema-cd25-xsp10sem05-s4-ft-s5` (clock `order`, intervals `log`)

| Arm | Local test | Public |
|---|---|---|
| intact | .694 | .671 |
| intervals (IOI) := batch mean | .384 | .363 |
| all timing (`onsets` output) := 0 | .338 | .306 |
| order-clock onset := 0 | .590 | – |
| MEG-blind (`words.pool` := train mean) | .590 | – |

- The local→public shrink is ~.02, near-uniform across arms: the Kaggle timing path is live and IOI's full local
  gain (.31) is on public. `7gf.105`'s "no public gain from IOI" came from comparing different models, not plumbing.
- Per-feature split (local, CPU, `agentA/timing.md`): the gap to the **next** onset carries .18 of it (`nextnone`
  .510), the previous gap .05, per-span speech rate nothing (`rowmean` ≈ `bothmean`). The model reads the gap
  near-continuously (4-bin quantisation −.04, σ=.10 log noise −.02).
- 25.8% of Kaggle Deep rows have no next gap (156 epoch-last words, 92 isolated windows), against almost none in
  training — the largest timing lever left (`--context.intervals ended`; a next-gap dropout flag).

## Broad — member `ema-bgmix05-all-w15-s0-adapt12-mask-ep40` (as `b-s0k25`)

| Arm | Broad local test | Public |
|---|---|---|
| intact | .620 | .607 |
| MEG-blind (`words.pool` := batch mean) | .586 | .534 |

- MEG is worth .073 on Broad public, **twice** its .034 on the Broad local test. The Broad local test (seen
  subjects, stimuli near training) lets timing and context carry more than the Kaggle Broad rows do, so it
  under-prices encoder changes: small Broad local deltas on encoder changes are not predictive of public, either
  sign (cf. per-subject levels: local −.002, public −.015).

## What this does not show

- MEG-blind replaces the pooled embedding by a mean, so it prices the encoder *given* a model trained with it
  present; a model trained without timing would lean on MEG more (the timing-blind experts queued in `blv5`).
- One member per track; arms are diagnostic instruments, not score candidates.
