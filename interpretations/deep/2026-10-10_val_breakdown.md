# Where the Deep blend loses on VAL: source × word frequency (2026-10-10)

**Scope.**

- **Source.** The saved `hits.pt` of `m11xsp7hub6d2v3` (public .68883): per-word top-10 hits on VAL, tau .5. Nothing
  was re-inferred.
- **Denominator.** VAL only (subject 0, 10% stratified split): Sherlock1 n = 1296, TheMoth n = 1216. VAL selected the
  members' epochs, so these numbers flatter the blend; the comparison between buckets is what is read here.
- **Buckets.** The training-frequency quartiles of the class, with the cuts of the Broad breakdown
  (`../broad/2026-10-10_ruler_breakdown.md`). About 12 classes each. Cells read "macro / pooled hit".

| source | macro | Q1 (rarest) | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| Sherlock1 | .7314 | .537 / .667 (n 39) | .803 / .757 (n 136) | .742 / .747 (n 225) | .850 / .885 (n 896) |
| TheMoth | .5756 | .464 / .519 (n 54) | .527 / .653 (n 147) | .570 / .606 (n 203) | .738 / .831 (n 812) |

## Reading (inferences)

1. **The .156 source gap is spread over every quartile, and is widest in Q2 (.276) and Q3 (.172).** Common words
   (Q4) lose only .112. TheMoth's loss is in the mid-frequency classes, not at the rare tail.
2. **The rare tail is the floor in both sources** (.54 and .46), as on Broad.
3. Q2 and Q3 hold 23 classes and about 11–29 words per class per source. A gap this size across about 300 words
   per source is far outside word-sampling noise, but a single model-selection split is all there is.

## Not ruled out

- **Why TheMoth is weaker.** Fewer subject-0 training sessions, a different speaker and rhythm, or more varied
  text. A count of training words per source and a per-source timing-only probe would separate them.
- **Whether the Kaggle Deep test looks like TheMoth or Sherlock1.** Its stimuli differ from both. If it resembles
  neither, TheMoth's gap is the better estimate of the transfer cost.
