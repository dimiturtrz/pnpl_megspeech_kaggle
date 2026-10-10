# Where the Broad ruler loses: subject, word frequency, span position (2026-10-10)

**Scope.**

- **Source.** Saved `broad_hits.pt` per-word top-10 hits, written by `megspeech/submission/report.py`. Nothing
  was re-inferred. Each model's saved macro reproduces exactly from the hits.
- **Ruler.** Subjects 1–32 on Sherlock1 ses-11, words of unseen multi-word spans only, n = 25322. The score per
  subject is balanced top-10 accuracy (mean per-class hit rate). The ruler is the unweighted mean over subjects.
- **Models.**
  - **xc5w6all6:** the 29-member blend, scored raw (no unseen fallback, no Sinkhorn), .6329.
  - **xc5a32k25s:** the same 29 members plus the all32 adapt member, with mean fallback k .25 and Sinkhorn
    β .25, .6595.
  - The two differ in both the balance step and the extra member. `xc5k25s` (balance, no extra member,
    .6591) separates them in Reading 1.
- **Buckets.** Inside a bucket, "macro" is per-subject balanced accuracy over the classes present, averaged over
  subjects. The ruler holds only 41–47 classes, so the frequency quartiles are about 12 classes each and move in
  coarse steps.

## Subject group

| subjects | ruler words each | classes | xc5w6all6 | xc5a32k25s |
|---|---|---|---|---|
| 1–12 | ~1296 | 47 | .6607 | .6732 |
| 13–22 | ~665 | 44 | .6389 | .6667 |
| 23–32 | ~310 | 41 | .5935 | .6358 |

Raw, the worst single subject is 29 at .5181 and the best is 4 at .6807. The groups are different recording
sources, so subject, source, word count and class set are confounded.

## Training-frequency quartile of the class (Q1 rarest)

| quartile | classes | share of words | xc5w6all6 macro / pooled hit | xc5a32k25s macro / pooled hit |
|---|---|---|---|---|
| Q1 | 12 | 3.5% | .521 / .530 | .536 / .551 |
| Q2 | 12 | 10.7% | .668 / .686 | .688 / .702 |
| Q3 | 11 | 16.7% | .609 / .632 | .683 / .662 |
| Q4 | 12 | 69.0% | .714 / .848 | .710 / .772 |

## Position in the span

The raw blend is flat over position: first word .649, middle .635, last .643. There is no deficit for
first or last words.

## Reading (inferences)

1. **The step from the raw blend to the submitted blend trades common-word hits for rare-class recall.**
   - Q4's pooled hit rate falls from .848 to .772 while its macro stays level.
   - Q2 and Q3 rise.
   - This is what Sinkhorn balancing toward the prior should do under a balanced metric.
   - Isolated with `xc5k25s`, which has the same 29 members and the balance step but no all32 member:

| model | 1–12 | 13–22 | 23–32 | Q1 | Q2 | Q3 | Q4 | Q4 pooled hit | ruler |
|---|---|---|---|---|---|---|---|---|---|
| xc5w6all6 (raw) | .6607 | .6389 | .5935 | .521 | .675 | .608 | .714 | .848 | .6329 |
| xc5k25s (balanced) | .6721 | .6655 | .6371 | .536 | .697 | .675 | .710 | .772 | .6591 |
| xc5a32k25s (+all32) | .6732 | .6667 | .6358 | .536 | .697 | .675 | .710 | .772 | .6595 |

   - The whole shift is the balance step. The all32 member changes no bucket by more than .002.
2. **The subjects with the fewest ruler words gain most:** 23–32 gain .042, 1–12 gain .013.
   - Fewer words per class make per-subject balanced accuracy noisier and more sensitive to calibration.
   - It is also possible that their source is the most mis-calibrated by the raw prior.
3. **Position carries no lever.** A span-position refiner would target nothing.
4. **The rarest classes remain the floor (.52–.54).** They are only 3.5% of words but a quarter of the classes,
   so in a balanced metric they weigh as much as Q4.
   - A sampler that draws rare words more often was tried on Deep (`--sampler.balance 0.5`). It was null at
     the macro: rare recall went up and common-word top-1 went down.

## Word-count-matched subsample

Each subject in 1–12 was cut to 310 random ruler words, the 23–32 mean, and rescored. This was repeated 200
times. A second arm kept only words of classes that subjects 23–32 also have.

| model | 1–12 full | 1–12 at 310 words | 1–12 at 310, 23–32 classes | 23–32 |
|---|---|---|---|---|
| xc5w6all6 (raw) | .6607 | .6834 ± .011 | .6787 ± .012 | .5935 |
| xc5k25s (balanced) | .6721 | .6960 ± .011 | .6910 ± .012 | .6371 |

- Fewer words raise the score, because the rarest classes drop out of a small sample. So word count does not
  explain the deficit; the matched gap is wider than the raw one (.085 raw, .054 balanced).
- The deficit belongs to subjects 23–32 themselves, meaning their source or recordings. The lever is per-source
  (calibration, anchoring, or a source token), not more words.

## Not ruled out

- **A per-source prior or calibration step.** Sinkhorn per source rather than pooled is untested. Any such
  step has to be fitted on training statistics, never the Kaggle test MEG.
- **The cause of subjects 23–32's deficit:** source acoustics or recording quality. Word count is ruled out
  below.
- **Rare-word balanced sampling on Broad.** Only its Deep use was measured.
