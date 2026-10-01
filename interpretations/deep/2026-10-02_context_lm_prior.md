# context-v0: where the sentence-context gain comes from, and why it does not reach the LB

context-v0 has a CNN word embedding, a 4-layer transformer over the span, and onset encoding. It was trained on 44 subject-0 runs, nearly all Sherlock.
Its holdout LB score is **0.307**. For comparison, cnn-v0 (isolated words) scores 0.221 and the frequency prior scores 0.200 (chance).

## Val, split by task (context head, BAcc@10, logit-adjusted)

| val run | words | context | word head | zero-MEG context | dense (≥4-word spans) |
|---|---|---|---|---|---|
| Sherlock1 ses-11 run-2 | 1308 | 0.549 | 0.439 | 0.333 | 0.641 (n=603) |
| MOCHATIMIT ses-2 run-1 | 518 | 0.384 | 0.316 | 0.315 | 0.513 (n=116) |
| combined | 1826 | 0.472 | 0.375 | 0.331 | 0.571 |

Ablations on the combined val:

| input | context |
|---|---|
| real | 0.472 |
| zero-MEG | 0.331 |
| shuffled word order | 0.339 |
| each word isolated | 0.323 |

## Reading

- **About 0.13 of the context score above chance is a word-order language-model prior, not MEG.** A model fed the scaler median instead of MEG still scores 0.33. Shuffling word order or isolating words removes most of the gain.
- The prior is learned from Sherlock text. MOCHA-TIMIT read sentences carry a different word order, so on MOCHA val the MEG-driven part is small:
  - context minus zero-MEG is only 0.07 on MOCHA, against 0.22 on Sherlock;
  - the word head scores 0.316 on MOCHA.
- The LB (0.307) sits at MOCHA val (0.384) minus a margin, not at Sherlock val (0.549). The holdout sentences behave like read sentences, not narrative.
  - Scale shift was ruled out: the holdout/train IQR ratio is 0.97 and the median shift is 0.02 IQR.
- **Denominator caveat:** MOCHA val has 518 words with a mean span of 1.74 words, so its dense subset has only n=116. Per-class BAcc on that n is noisy at roughly ±0.03.

## Consequences

- Selection now uses the macro over val tasks instead of pooling words (train.py, 82faf10). Pooling let Sherlock decide 72% of the score.
- The next lever is data: MOCHATIMIT/TIMIT runs in train (LibriBrain2 fetch), and a check that per-task val moves.
- A word-head-only LB submission would separate MEG transfer from LM-prior transfer. Spend one only if two models disagree on it.

## Follow-up: context-v1 (all 129 runs, incl. 14 TIMIT + 2 MOCHATIMIT)

The same ruler is used for both models: three-task val, macro over tasks, and a prior from all train runs, unweighted.

| model | MOCHA (n=518) | Sherlock1 (n=1308) | TheMoth (n=1226) | macro | LB |
|---|---|---|---|---|---|
| context-v0 | 0.387 | 0.529 | 0.347 | 0.421 | 0.307 |
| context-v1 | 0.527 | 0.546 | 0.439 | 0.504 | 0.385 |

- The local macro gain (+0.083) matches the LB gain (+0.078), so the macro-over-tasks val tracks the LB. Most of the gain is on MOCHA (+0.14), where the read-sentence data is in-domain, and on TheMoth (+0.09).
- v1's word head on MOCHA (0.264) is *below* v0's (0.314), while its context head rose. The extra data mostly bought read-sentence word-order prior, not better single-word MEG decoding. The read-sentence share is still only 5.8% of train words. That motivates `--read-weight` (c4fefd3), which tests whether oversampling those runs moves MOCHA and the LB further.

## Correction: MOCHA val is memorised material; the macro mis-selected

The val spans were matched against training: a span's exact sequence of vocab labels was looked up among every training span's.

| val run | spans ≥3 words | of those, verbatim in train | words in unseen multi-word spans |
|---|---|---|---|
| MOCHATIMIT | 61 | 61 (100%) | 2 / 518 |
| Sherlock1 | 192 | 91 (47%) | 499 / 1308 |
| TheMoth | 188 | 33 (18%) | 980 / 1226 |

MOCHA-TIMIT reuses TIMIT prompts, so every MOCHA val sentence is also read in a TIMIT train run. Sherlock1 val repeats a book the subject heard in earlier sessions.

The macro over tasks mis-selected because of this. It ranked a v1+v2 ensemble first (0.542), but the ensemble scored 0.310 on the LB, against 0.385 for v1. The ensemble is dominated by context-v2 (`--read-weight 4`), which memorised TIMIT word order: v2 scores 0.646 on MOCHA val but 0.446 on unseen spans. The earlier reading that "the LB tracks MOCHA val" was wrong. That correlation came from v0→v1 adding the same TIMIT data that both helps recitation and helps a little genuinely.

Two views do NOT separate the models: shuffling word order within spans, and isolating each word. On both, the ensemble ≥ v1. So the ensemble's MEG evidence is not worse. What hurts it on new sentences is a stronger, wrong word-order prior.

Context BAcc@10 on words in unseen spans, as the mean of Sherlock1 and TheMoth:

| model | unseen-span mean | LB |
|---|---|---|
| v0 | 0.479 | 0.307 |
| v1 | 0.525 | 0.385 |
| ens | 0.496 | 0.310 |

This ranking matches the LB order; the magnitudes do not (ens is closer to v1 locally than on the LB). That makes it a better ruler, not a calibrated one. It is the selection metric since e06fbeb.
