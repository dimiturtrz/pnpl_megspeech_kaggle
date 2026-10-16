# Final-test readiness: how the scored picks generalise

megspeech-e1f · 2026-10-16 · read from existing measurements; no new runs except where marked.

The final test uses stimuli unlike anything released, and possibly subjects unlike them too. The scored picks are
the Kaggle default (best public): Deep `d-ac05xfam16` (public .69483) and Broad `b-xc5k25b0ps` (public .63097). This
note rates each against what could change, from the numbers we already hold.

## Deep: d-ac05xfam16

| what changes | rating | evidence |
|---|---|---|
| New sentences | good | VAL on sentences never trained matches VAL on all: Sherlock1 .724 / .727, TheMoth .571 / .571. No text recall. |
| New recordings | known drop | Local test ~.72, public .695: about −.03 on a third recording set. Expect at least as much. |
| New source or narrator | weak | VAL Sherlock1 .73, TheMoth .57. New stimuli will likely land nearer the TheMoth figure. |
| Sentence context | largest exposure | Word head alone .44 / .31 (Sherlock1 / TheMoth); in sentence context .73 / .57. Around 40% of the score comes from the span. If the final test is cut into sentences as public is, it holds. |
| Selection | low overfit | 16 seeds averaged, tau .5 from a 3-point grid, nothing fitted on test rows, never tuned on public. Top Deep entries sit within .004 on public, below what one score can resolve; the 16-seed family has the lowest prediction variance of them. |
| VAL inflation | caveat | MOCHATIMIT VAL .926 is recited TIMIT text. The VAL macro reads high. |

## Broad: b-xc5k25b0ps

| what changes | rating | evidence |
|---|---|---|
| Transductive balancing | moderate, measured | Per-subject Sinkhorn to equal class mass. Gained on two separate natural-speech sets: Broad ruler +.026 (`xc5k25s` vs raw, [ruler breakdown](../broad/2026-10-10_ruler_breakdown.md)), public +.011 alone and +.019 with the mean fallback ([fallback shrink](../broad/2026-10-10_subject_fallback_shrink.md)). On 310-word subsamples it still gains +.013, and the subjects with fewest words gain most. Not measured: a test with far fewer rows per subject, or a much more skewed word distribution. |
| New subjects | unmeasured, being measured | The 0% group (subjects 33–39, 7/39 of public rows) already reads through the mean subject matrix. But no number scores a subject the adapt never saw: the .6137 "zero-shot" read kept each subject in the backbone and in the mean. megspeech-cbb trains the adapt without 10 of subjects 1–32 and scores them zero-shot. |
| New text | covered | The Broad ruler scores only unseen-span words (`split.novel`), so the context head cannot recall the story there. |

## Not ruled out

- A final-test sentence format different from public's (spans cut differently, isolated words). Deep loses up to
  ~.3 if context is gone; nothing we hold measures a partial loss.
- Very small per-subject row counts in the final Broad test, where Sinkhorn's class masses would be noisy.
