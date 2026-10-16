# Deep: a source the model never trained on

megspeech-c5t · 2026-10-16 · base recipe `ema-cd25-xsp10sem05-ac05x` s1, trained with `--data.tasks` minus TheMoth (30 of
162 subject-0 runs), fine-tuned s1, scored at tau .5

## Question

The final test's stimuli are entirely new. TheMoth is the closest stand-in we hold: a different narrator and genre from
the Sherlock books. Leave it out of training and score it: what does an unseen source cost?

Pre-registered: TheMoth drop ≤ .05 means Deep transfers across sources; > .10 means source-general training is the Deep
lever.

## Result (bacc@10, tau .5)

| | Sherlock1 VAL | TheMoth VAL | Sherlock1 local test | TheMoth local test |
|---|---|---|---|---|
| control (base s1, all sources) | ~.72 | .571 | — | ~.66 |
| nomoth s1 | .7242 | .5024 | .7162 | .5822 |
| nomoth − control | ~0 | −.069 | — | ~−.08 |

## Reading

TheMoth drops .07 on VAL and about .08 on the local test, between the two pre-registered bars. Leaving TheMoth out also
cut 18% of the training data, but Sherlock1 is unchanged, so the data loss did not weaken the model in general (unlike
the Broad held-out-subject run, where seen subjects lost too). The .07 is the cost of the source being unseen.

Even trained on, TheMoth is the weak source (.57 against Sherlock1's .72); unseen it falls to .50. A final test of new
stories by new narrators should be expected to land well below the Sherlock-weighted local test, and the public's
recordings sit in between.

## What it means

- A moderate exposure, not the dominant one: sentence context (~.3 if lost) is larger.
- Picks whose training covers more sources and narrators have a real, if unmeasured, case on the final test. The
  pooled-to-Deep carry (d6r.11) tested one form of this and lost on the local test and public, both of which share the
  training sources, so neither could see the axis it would help on.

Not ruled out: the control here is a single seed against single-seed numbers (seed SD ~.013 on the local test); the
direction is clear, the size is ±.02.
