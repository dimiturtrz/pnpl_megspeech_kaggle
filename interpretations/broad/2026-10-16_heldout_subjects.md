# Broad: subjects the adapt never saw

megspeech-cbb · 2026-10-16 · `train --origin ema-bgmix05-all-w15-s0-adapt12-mask-ep40 --data.subjects 0 1 2 4 5 7 8 10 11 13
14 16 17 19 20 22 23 25 26 28 29 31 32 --name bho10-adapt12-mask-ep40`, both arms scored by
`megspeech.submission <model> --track broad --fallback.unseen mean --fallback.keep 0.25`

## Question

The final Broad test may hold subjects we have no labelled data for. The pooled stem never trained on subjects 1–32, so
an adapt that leaves 10 of them out (3, 6, …, 30) reads those 10 exactly as it would read a new subject: through the
mean subject matrix. How much does that cost against reading them after adapting on them?

Pre-registered: gap ≤ .03 means new subjects are a small exposure; > .05 means the zero-shot read is the lever.

## Result (Broad ruler, bacc@10, tau .5)

| | held-out 10 | the other 22 |
|---|---|---|
| control (adapted on all 32) | .6273 | — |
| bho10 (adapted on 22) | .5907 | — |
| bho10 − control | −.037 | −.022 |

Ruler over all 32: control .6199, bho10 .5934.

## Reading

The raw gap (−.037) sits between the two pre-registered bars, but most of it is not about the subjects being unseen.
The 22 subjects bho10 did adapt on lost −.022 too: an adapt on 22 subjects is a weaker model than one on 32, for
everyone. The part specific to never having seen a subject is the difference, about −.015, with a sampling error of
about ±.011 (per-subject SD ~.03, 10 against 22 subjects). That is a small exposure.

The seen-subject loss is largest on subjects 28–32 (−.04 to −.10), the ones with about 300 words each: those lean
hardest on what the other subjects teach the shared layers.

The training run's own end-of-run ruler reported .5712 for the held-out 10. That number used the default fallback
(subject-0 matrix, keep 1.0), not the control's mean k.25, so it is not comparable; matched, it is .5907.

## What it means for the final test

A subject outside 1–39 should land about .015 below an adapted one, not .05. Picks built only to protect against new
subjects buy little; the mean-subject read already does most of that work. More labelled subjects help everyone,
which the seen-subject loss shows.

Not ruled out: subjects recorded on a different system or with a different story, where the mean matrix itself may not
fit.
