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
