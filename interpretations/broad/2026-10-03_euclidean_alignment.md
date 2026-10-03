# Broad: Euclidean alignment of other subjects to subject 0 (2026-10-03)

Question (bd `megspeech-7gf.56`): a subject-0 model reads 32 other subjects' MEG on the Broad track. Their
sensor covariance differs from subject 0's (head position, noise). Does re-colouring each recording to subject
0's mean covariance (Euclidean alignment, He & Wu 2019) recover Broad accuracy, and at what cost on Deep?

## Answer

Yes on Broad, at both eval time and train time. Aligning subject 0's own runs costs Deep accuracy, so each
track keeps its own model.

| model | alignment | Deep ruler (n=2512) | Broad ruler (n=25322) | Broad LB |
|---|---|---|---|---|
| ens-ema9998-s0to7 (8 seeds) | none | 0.6626 | 0.528 | 0.55810 |
| ens-ema9998-s0to7 (8 seeds) | eval: subjects ≠ 0 | 0.6626 (sub-0 untouched) | 0.565 | 0.56611 |
| sent40-ioi-ema9998-s0 | eval: subjects ≠ 0 | 0.6453 | 0.5389 | — |
| sent40-ea-s0 | train + eval: every run | 0.6247 | 0.5604 | — |

Paired deltas, all on the same rows:

- **Eval-time EA, 8-seed ensemble, Broad:** +0.038, CI [+0.030, +0.046].
- **Train-time vs eval-time EA, single model, Broad:** +0.021, CI [+0.011, +0.032].
- **Train-time EA, single model, Deep:** −0.021, CI [−0.053, +0.002]. The run plateaus at 0.61–0.62 from epoch
  18 and does not recover. Read this as a real cost.

The LB moved +0.008 for a local +0.038. That is inside the public LB's ±0.04 resolution (see bd memory
`ruler-lb-rank`), so it is consistent with the local gain but does not confirm it.

## Mechanism

- **Eval-time EA helps.** A subject-0 model learned spatial filters in subject 0's sensor geometry. Matching
  another subject's second-order statistics to subject 0's moves that subject's evoked field closer to where
  the filters look.
- **Aligning subject 0 at eval hurts** (0.644 → 0.588). The model never saw re-coloured subject-0 data, so this
  is pure distribution shift.
- **Train-time EA buys Broad.** It trains in the same geometry every subject is then mapped into, so train and
  test share one distribution.
- **Train-time EA costs Deep.** Recolouring each subject-0 run to the mean covariance removes run-to-run spatial
  variance. That variance is either informative or was acting as augmentation. This is untested, and it is the
  open question.

## What it does not rule out

- A Deep model trained with EA plus covariance augmentation.
- Riemannian (log-Euclidean) rather than arithmetic reference means.
- EA as the enabler for cross-dataset pretraining (external MEG; the rules allow public data). Classic EA use is
  exactly cross-subject training, which this repo cannot do with LibriBrain alone: subjects 1–32 only have the
  forbidden ses-11/12.
