# OVMI auxiliary loss and word-embedding geometry (2026-10-09)

**Scope.** This is a CPU analysis of saved checkpoints and `probs.pt` files. No training was run.

- **Ruler.** As in `2026-10-08_error_anatomy.md`: novel words, top-10 balanced accuracy where ties count
  against the true word, Sherlock1 + TheMoth macro.
- **Denominators.** val n=2512, LT n=2133.
- **Scripts.** The scripts are not committed: `why/q23.py`, `why/q4.py` and `why/q4c.py` in the job scratch dir.

## Q3 — OVMI (mi 0.1): val up, LT down, best epoch 5

All four runs fine-tune from the same warm start with the same seed. Training logs were parsed per epoch.

| run | val | LT | kcut test | best epoch |
|---|---|---|---|---|
| base s1 | .6600 | .6857 | .6606 | 23 |
| mi003 | .6587 | .6823 | .6531 | — |
| mi005 | .6542 | .6658 | .6612 | — |
| mi01 | .6660 | .6686 | .6626 | 5 |

**Measured.**

- **The best epoch is a selection artifact.**
  - The val curves of all four runs move together, noisily between .635 and .667, with no trend over 40
    epochs.
  - mi01's epoch-5 best (.6673) is a one-epoch spike; its neighbours are .6414 and .6561. The baseline has
    equal local peaks at epoch 5 (.6580) and at epochs 22–23.
  - The best epoch is therefore the maximum of about 40 noisy reads, not an earlier convergence.
- **The LT drop does not replicate.** On the kcut test, mi01 is +.002 over the baseline.
  - The sentence-LT drop is all TheMoth: .5967 vs .6428 (n=775). Sherlock1 goes the other way, .7405 vs .7287.
  - Within TheMoth, the 11 classes with fewer than 5 words lose −.159 mean per-class recall. The classes with
    5 or more words lose only −.012. In Sherlock1, the 14 small classes *gain* +.036.
  - Balanced accuracy weights each tiny class as much as a frequent one. A few flipped words therefore move
    the macro by about .02.

**Measured: the mechanism did fire.** Output-distribution stats on LT, Sherlock / TheMoth (entropy is
normalised by log 50):

| run | entropy | max p | NLL | top-1 marginal entropy | KL(mean p ‖ train prior), TheMoth | k-occurrence Gini |
|---|---|---|---|---|---|---|
| base | .634 / .721 | .358 / .277 | 2.303 / 2.623 | .806 / .792 | .080 | .30 / .36 |
| mi003 | .627 / .716 | .365 / .281 | — | — | — | — |
| mi005 | .614 / .690 | .372 / .303 | — | — | — | — |
| mi01 | .597 / .676 | .387 / .315 | 2.322 / 2.651 | .818 / .811 | .098 | .29 / .36 |

- Entropy falls monotonically with the MI weight. This is the conditional-entropy half of the MI objective.
- The top-1 marginal flattens slightly. This is the marginal-entropy half.
- NLL *rises* slightly, so the model becomes sharper without becoming more correct.
- Hubness, measured as the k-occurrence Gini, is unchanged. Micro top-10 hit rates are within ±.02 at every tau.

**Interpretation (inference).**

- OVMI does exactly what it was designed to do, and that has no detectable effect on ranking.
- Top-10 recall depends on where the true word ranks, not on how peaked the distribution is. Sharpening plus
  a marginal pushed slightly away from the prior mostly reshuffles the tail.
- The "val +, LT −" pattern is two noisy reads, one of which is driven by small TheMoth classes. It is not a
  val/test trade-off.

**Does NOT rule out.**

- OVMI on a model whose marginal is actually collapsed. Here the marginal is already about 0.8 normalised
  entropy and hubness is moderate.
- OVMI as a calibration aid for an LM-fusion stage. Sharper and better-ordered probabilities might matter
  there; that is untested.

## Q4 — word-embedding geometry

**What was measured.**

- The embedding is the stage-1 word-head embedding (512-d), extracted on subject 0's local-test runs.
- The statistics below are for Sherlock1 + TheMoth only (n=2163; MOCHA excluded).
- The Broad model is measured on subject 0, not on Broad subjects. For Broad-subject geometry, see
  `2026-10-08_attribution_embeddings.md`.

| | masc27w-s1 | xsp10sem05-s4-ft-s5 | Broad all-w15-s0-adapt12 |
|---|---|---|---|
| mean pairwise cosine, raw | .778 | .792 | .783 |
| same, after subtracting run centroid | .003 | .001 | .001 |
| PC1 variance share | .175 | .150 | .216 |
| effective rank (participation ratio) | 16.3 | 18.2 | 11.4 |
| Sherlock–TheMoth centroid distance / within-run spread | .98 | .85 | 1.14 |
| eta² of label (run-centred) | .061 | .067 | .069 |
| eta² of position bin | .035 | .043 | .041 |
| top-5 classes' share of all top-10 slots (uniform = .10) | .42 | .39 | .47 |
| r(k-occurrence, train log prior) | .83 | .84 | .77 |
| r(norm, log prior of true word) | .05 | .06 | .09 |
| word-head hit rate by prior tercile, rare / mid / frequent | .02 / .21 / .85 | .02 / .24 / .85 | .00 / .13 / .83 |
| leave-one-out same-word cosine (run-centred), hit / miss | .049 / .018 | .057 / .015 | .062 / .023 |
| distance to run centroid, hit / miss | 3.77 / 3.47 | 4.91 / 4.58 | 2.43 / 2.35 |
| max p, hit / miss | .145 / .123 | .163 / .135 | .162 / .151 |

**Measured.**

- **The anisotropy is a session offset.** The raw mean cosine is about 0.78. Subtracting each run's centroid
  brings it to about 0.
- **Hubs are frequent words.** The word head's top-10 hubs are the frequent classes: r = .77–.84 with the
  train prior. The top 5 classes fill 39–47% of all top-10 slots.
- **The word head is close to a frequency ranker.** Rare-tercile words are almost never in its top 10
  (0–4%).
- **Norm is not frequency.** Embedding norm carries no frequency signal (|r| < .1).
- **Hits and misses differ in signal, not direction.** Misses sit closer to their run's centroid, by 7–8% for
  the Deep models, and carry about one third of the same-word coherence of hits.
- **xsp vs masc.** Compared with masc, xsp is more isotropic (rank 18.2 vs 16.3), has a smaller session shift
  (.85 vs .98) and slightly more word structure (eta² .067 vs .061). This matches its better word head
  (+.015 to +.02 LT).
- **Broad vs Deep.** The Broad pooled stem has the lowest effective rank (11.4), the largest session shift
  (1.14) and the strongest hubness (.47). On subject 0 its word head is weaker (micro .632 vs .661–.669).

**Interpretation (inference).**

- A miss is a low-signal embedding that collapses toward the session mean. From there the prior-shaped head
  emits the frequent-word hubs. That is the same function-word crowding seen in the error anatomy.
- The errors are not "wrong direction" errors. Cosine to other words' centroids is about 0 for both hits and
  misses.
- Raising the per-word signal-to-noise matters more than reshaping the output distribution. This is also why
  OVMI (Q3) did nothing.

**Does NOT rule out.**

- The context head's geometry differs. Only the stage-1 embedding was extracted, and the context head reads
  that embedding after the span transformer.
- These are single seeds per model, and no confidence intervals were computed for the geometry statistics.

## Next experiments (ranked by plausible gain)

1. **Per-run embedding centring or normalisation before the heads** (test-time, label-free).
   - Rationale: the raw anisotropy is entirely the run offset, and Sherlock–TheMoth shift is about 1× the
     within-run spread. The final test is unseen sessions.
   - Cost: cheap to test on saved embeddings, by refitting a linear head on centred embeddings.
   - Plausible gain: +0–.02. It is the only knob here sized near the bar.
2. **Better-than-hubs handling of low-signal rows.** An example is confidence-gated LM rescoring: when max p
   is low, defer to the language prior over the span.
   - Rationale: misses are the low-signal rows, already flagged by max p (.12–.15 vs .15–.16).
   - Plausible gain: only together with the LM lever from the frontier sweep.
3. **Drop OVMI as a score lever.** It works mechanistically but leaves ranks unchanged. Keep it only if
   LM fusion needs calibrated probabilities.
