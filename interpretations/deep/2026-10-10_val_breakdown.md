# Where the Deep blend loses on VAL: source × word frequency (2026-10-10)

**Scope.**

- **Source.** The saved `hits.pt` of `m11xsp7hub6d2v3` (public .68883): per-word top-10 hits on VAL, tau .5. Nothing
  was re-inferred.
- **Denominator.** VAL only. The blend's members are legacy full-data models (`data.val: official`), so VAL is pnpl's
  official validation runs `sub-0_ses-11_task-Sherlock1_run-2` and `sub-0_ses-29_task-TheMoth_run-1`, not the 10%
  stratified split: Sherlock1 n = 1296, TheMoth n = 1216 scored hits. VAL selected the members' epochs, so these
  numbers flatter the blend; the comparison between buckets is what is read here.
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

- **Why TheMoth is weaker.** An earlier version of this note ruled out training quantity from the VAL word counts
  (1308 / 1226). That was wrong: those are VAL counts, not training counts. Subject 0 trains on 265,113 labelled
  Sherlock words and 24,510 TheMoth words, 11× fewer. Training quantity stays a candidate.
- **Whether the Kaggle Deep test looks like TheMoth or Sherlock1.** Its stimuli differ from both. If it resembles
  neither, TheMoth's gap is the better estimate of the transfer cost.

## Timing alone carries about half the source gap

A probe that uses timing only, with no MEG, trained on subject 0's training words and scored on the same VAL runs.

- **Features:** log duration, log IOI to the previous word, place in sentence, sentence length, log time to
  sentence end, and edge flags.
- **Model:** a spline basis, then a multinomial logistic regression.
- **Script:** scratch only, not committed.

Top-10 balanced accuracy on the 47 classes present:

| reader | Sherlock1 | TheMoth | gap |
|---|---|---|---|
| chance / train-frequency prior | .200 / .213 | .200 / .213 | 0 |
| timing only, trained on all sources | .514 | .440 | .074 |
| timing only, trained on its own source | .555 | .464 | .091 |
| the blend (MEG + timing) | .731 | .576 | .155 |

**Reading (inferences, one probe):**

1. **TheMoth's words are less predictable from timing alone.** About half the blend's .155 gap is already there
   without MEG.
2. **The gap holds when each source trains its own probe** (.091), even though TheMoth then trains on 11× fewer
   words. The cause is more likely the stimulus itself, many tellers with varied rhythm against one reader, than
   the pooled fit favouring Sherlock.
3. **What MEG adds is .217 on Sherlock1 and .136 on TheMoth,** read as the blend minus the pooled probe. The other
   half of the gap sits in what MEG adds, and that is where 11× less TheMoth training could act.

**Not ruled out:** whether the MEG half is quantity or stimulus.

- **Prior test.** `--data.narrator_weight 0.5` (megspeech-7gf.47) shifts draws from the Sherlock narrator's runs
  toward the other narrative runs. On one seed it left TheMoth flat (.541 vs .542) and cost Sherlock1 .032.
- **What that shows.** Re-weighting the same 24.5k TheMoth words does not buy TheMoth, so the MEG half is not a
  sampling-share problem.
- **What it leaves open.** More distinct TheMoth-like words could still help. Only new data would test that.
