# Deep error anatomy: where the local-test score is lost (2026-10-08)

Scope: saved predictions only (`probs.pt`, `hits.pt`, `metrics.json`), CPU, no model reloaded, no MEG read. Kaggle test:
onset/mask arrays only. Scripts (one-off, not committed): `C:/Users/User/.claude/jobs/44912819/tmp/research/`
(`frame.py` joins `probs.pt` ids to the text table, label agreement 1.0 on all 150 model×section×domain blocks;
`anatomy.py`, `extra.py`, `kstats.py`).

**Denominator unless stated:** LOCAL TEST novel (unseen-span) words of the two counted domains, Sherlock1 n=1358 +
TheMoth n=775 = **2133 words**, 45 + 47 = 92 domain-classes. MOCHATIMIT has 36 novel words (< `min_novel` 100) and
does not count. Metric = balanced top-10 accuracy per domain, macro over the 2 domains. Chance = 0.20.

**Ensemble:** masc7lf16 has 31 members; 8 (`ema-cdrop25-s0..7`) have no `probs.pt`. ENS23 = mean of the other 23
members' adjusted probabilities. Check against the exact ensemble on official val (`masc7lf16/hits.pt`): exact .6487,
ENS23 .6535, per-word hit agreement 98.9% (2512 words). ENS23 LT = **.7000**. Close enough for anatomy; it is not
masc7lf16. `sent40-masc27-s0` (the stem) has no `probs.pt` and is not analysed.

## 1. Which words carry / fail (ENS23, LT)

- Domain split: Sherlock1 .743, TheMoth .657. TheMoth owns **57%** of the lost macro (0.300 total).
- **Small-support classes dominate the loss.** Domain-classes with n<5: 25 of 92 classes, 57 of 2133 words (2.7%),
  **33.8%** of the lost macro. n<10: 44 classes, 187 words (8.8%), **54.3%** of the loss. Balanced accuracy gives a
  1-word class the weight of `the` (n=265); e.g. `good` (TheMoth n=1, missed) alone = 3.5% of the loss, `has`
  (0/9 in both domains) = 7.3%.
- Biggest single losses (share of lost macro): has .073, him .044 (TheMoth 0%), an .042 (TheMoth 0%), my .040
  (Sherlock 0%), be .038, for .037, do .037 (n=2), not .037, good .035, as .034. Top 10 words = 41.6% of the loss;
  top 20 = 68.5%.
- Carry (≥.90): a .99, i .95, and .91, the .90, plus several rare content words at 1.00 on n≤11 (very, time, really,
  any, always, new, am) — rare content words are not the problem; rare/ambiguous **function words and pronouns** are.
- Domain flips: `my` Sherlock 0.00 / TheMoth .92; `have` .36/.80; `they` .42/1.00; `him` .77/.00; `an` .83/.00;
  `as` .89/.17 — word identity alone does not predict failure; the narrative's style does (inference).
- Training frequency: Spearman(class accuracy, train log prior) = −0.008 over 92 domain-classes, 0.115 over 50 pooled
  words. After logit adjustment (tau .5) frequency does not predict accuracy.

**Confusions.** 408 misses of 2133 (19.1%). Top-1 on misses falls on high-frequency function words: and→i 7,
they→the 6, of→the 5, he→and 4, for→and 4, in→and 4. Words most often among the 10 that outrank a missed word
(408 misses): i 154, we 145, it 141, that 140, to 138, the 136, my 134, is 130. The failure mode is a pronoun /
function-word cluster crowding the top 10, not a specific pair.

**Covariates** (hit rate raw / class-adjusted = mean of hit minus its domain×word mean, so word identity is removed):

| covariate | finding (n per bin ≈ 130-600) |
|---|---|
| position in sentence | flat: adjusted within −.064 (pos 5-7, n=292) … +.034 (pos 0, n=249); no trend |
| sentence length | flat: adjusted −.070 at length 1-3 (n=51), otherwise within ±.05 |
| time to sentence end | −.033 in the longest quintile only (the table has no per-word offset, so no true word duration) |
| **onset gap to next word** | monotone: quintile 1 (≤.09 s) +.045, Q5 (>.18 s) −.053; Q1−Q5 = **.098, 95% CI [.051, .148]** (sentence-block bootstrap, 300 resamples); not driven by sentence-final words (excluded, gap undefined) |
| preceding word | sentence start +.034 (n=249); prev OOV .000 (n=1208); prev in vocab −.012 (n=676). After `it` −.127 (n=30), after `of`/`to` +.075 (n=55/34) — n too small, directional |

Inference: a long gap after a word = slow delivery or a phrase-boundary pause; both the MEG window (next word's
response not inside it) and the context (phrase boundary, weaker right-context constraint) are weaker there.

## 2. Clip model `ema-cd25-masc27w15kg-s3` vs sentence model `ema-cd25-masc27w15-s3` (same 2133 words)

LT macro: sentence .7074, clip .6615 (−.046). Both scored under the grouping in their `probs.pt` (kg model's saved
grouping is `kaggle`; the notes on 7gf.132 report the 5-model kg ensemble at .6621 on the sentence ruler, consistent
with these numbers — I could not confirm from the file which cut the kg model's LT spans used).

| sentence position | n | sent | clip | Δ |
|---|---|---|---|---|
| 0 | 249 | .855 | .843 | −.012 |
| 1 | 191 | .827 | .827 | .000 |
| 2 | 129 | .698 | .674 | −.023 |
| 3-4 | 247 | .818 | .777 | −.040 |
| 5-7 | 292 | .733 | .627 | −.106 |
| 8-11 | 292 | .825 | .767 | −.058 |
| 12-19 | 393 | .802 | .730 | −.071 |
| ≥20 | 340 | .853 | .735 | −.118 |

By sentence length: ≤11 words Δ −.01 to −.04; ≥20 words −.080 (n=599) and −.106 (n=566). By position within a
Kaggle-cut piece: −.04 to −.08 everywhere, no edge concentration. Per word the clip model loses most on
context-predictable function words: that −.175 (n=63), they −.167, had −.149, so −.130, it −.125, the −.109 (n=265).
Reading: the clip model loses what long left context buys (positions ≥5 in long sentences), and does not win back
anything at piece edges. That explains its local-test loss; it does **not** by itself explain the public loss (the
public test IS clips). Untested: the clip model's public loss could come from a weaker context head overall (each
training word sees ≤~6 context words), which the per-position table cannot separate.

## 3. Language context vs MEG

Means over 25 single models (23 masc7lf16 members + the two s3 models):

| head | LT (2133) | official val (2512) |
|---|---|---|
| context head (full) | .686 | .645 |
| context head, MEG windows shuffled within domain (`best.shuffled`) | — | .527 |
| word head (MEG-only, no context), tau .5 | .388 | .344 (val tau-selected) |
| chance | .200 | .200 |

Above chance on val: full .445; shuffled-MEG .327 → **~73% of the above-chance score survives destroying the
word-to-MEG pairing** (language prior + onset timing); MEG word identity adds ~.118 (~27%). The word head alone is
.19 above chance on LT. Two frozen-encoder members (`*-frz-s1`) have a word head at chance-level .15-.20 yet LT
context .68-.69: the context head carries them. Per word, word-head and context accuracy correlate weakly (Spearman
.30, 50 words): MEG-only is strong on long/rare content words (think .82, do .85, time .98, people/really/always/new
1.0, n small) and near zero on the frequent function words (were .015, was .016, had .027, the .077) — exactly the
words the context head must carry.

## 4. Kaggle Deep test structure (subject 0; onset/mask arrays only) vs local test

| | Kaggle test | local test (Sherlock1+TheMoth, all words) |
|---|---|---|
| sentence epochs / units | 156 epochs, 868 words + **92 isolated-word epochs** (9.6% of 960 rows) | 444 sentences, 5389 words; ~0 lone words |
| words per unit | 1:2, 3:2, 4:24, 5:55, 6:37, 7:23, 8:11, 9:2 (mean 5.56, max 9) | whole sentences median 8, p90 29, max 50; Kaggle-cut pieces mean 5.23, mode 5-6 |
| first→last onset span | mean 1.18 s | Kaggle-cut pieces mean 1.21 s |
| intra-unit onset gap | median .21, mean .256, p95 .57, max .92 s | median .20, mean .287, p95 .84, max 1.71 s (Sherlock median .23, TheMoth .165) |
| first onset in epoch | 0 in ≥75% of epochs | — |
| last onset → epoch end | 1.02-2.78 s, median 1.36 (`sentence_extra_end_s` = 1.0) | runs continue |

Window overflow: with a 1.0 s window no Kaggle sentence word reads past its epoch; with **1.5 s, 176/868 = 20.3%**
do (2.0 s: 53.9%). Local-test whole sentences have no such cut. Inference: this is a concrete, onset-only mechanism
for the w1.5 public misses (m15x5 public .6261 vs masc7lf16 .6678), matching the kcut-ruler diagnosis in 7gf.132.
Kaggle's lack of >.92 s gaps (local p95 .84, max 1.71) suggests epochs are cut inside fluent stretches (inference).

## Headroom ranking (inference, not measured gains)

1. **Small-support classes** — 54% of the LT loss sits in classes with <10 words (8.8% of words). Part of this is
   ruler noise, not model error: the Kaggle test has 960 rows, so its rare classes are equally thin. A per-class
   decision rule (tau / prior per class, e.g. balancing top-10 inclusion rates of the pronoun/function cluster)
   targets it directly; 7gf.154 per-label weights on Broad did not transfer to public — a warning.
2. **Function-word / pronoun crowding** — misses' top-10 is filled by i/we/it/that/to/the; the/to/i/and sit in the
   top 10 of 39-41% of ALL 2133 rows, of/it/in/that/we 30-36%. The metric rewards spreading the 10 slots; a calibrated top-10 set selector (not
   argmax-prob order) is the lever I would bet on most (≥1.5% plausible, ~35% odds; needs val-only fit).
3. **Isolated words (9.6% of Kaggle rows)** get only the word head (.39 LT-style, vs .69 with context); local test
   has no such rows, so it under-measures them. Any word-head gain is worth ~0.1× on public directly.
4. **Long-gap words** (−.10 Q1→Q5) — small and not obviously actionable.

## 5. Zero-tail hypothesis for the w1.5 public losses (tested 2026-10-08)

Hypothesis: 1.5 s models lose on public (m15x5 .626, m15kgx5 .577 vs masc7lf16 .668) because Kaggle sentence
epochs end ~1 s after the last word, so a 1.5 s window is zero-filled past the epoch end — a condition training and
the local test never show.

**Plumbing facts (code).** The premise is false as stated: zero-fill past a bound is applied everywhere, not only on
Kaggle. `Windows.__call__` zeroes every sample from `end` on (`megspeech/data/batching/windows.py:76-79`), for
training, val, local test and Kaggle alike. `end` is `Grouping.bound` = `start + words[rule.stop]`
(`megspeech/data/spans/grouping.py:60-64`): `SENTENCE` → `stop` = sentence's last word finish + `TAIL` 1.0 s
(`run_cache.py:51,154,192`), `KAGGLE` → `piece_stop` = the 5.56-word piece's last finish + 1.0 s (`run_cache.py:193-197`);
both floored at onset + 1 s and capped at run end (`run_cache.py:205`). Kaggle: `end` = the epoch's end
(`kaggle.py:122`), samples past it zeroed by the same `Windows`; the zeroing is after cache-time decimation and
(optional) standardisation, then the per-window mean is taken over valid samples only (`windows.py:80-84`). So
the kcut ruler (`--grouping kaggle`) already zero-fills past piece ends. Not simulated by it: the 92 isolated-word
epochs, which are exactly 1.0 s (`pnpl/competition/holdout.py:20-22`), so a 1.5 s window there is 1/3 zeros.

**Exposure (1.5 s window, measured).** Kaggle Deep file: 27.1% of 960 rows read some zeros, mean zeroed share
5.2%, room past onset q10/50/90 = 55/94/132 samples. kcut local test: Sherlock1 21.7% / 3.0% (64/99/148), TheMoth
35.2% / 4.6% (61/82/112). Comparable — kcut is already a faithful simulation of the sentence rows.

**Scores** (local test = pnpl official test, Sherlock1+TheMoth, kcut spans, tau .5 from val, CPU,
`C:/Users/User/.claude/jobs/44912819/tmp/research/zerotail2.py`; macro bAcc@10 with bootstrap CI):

| model | public | open (continuous MEG) | zero past piece end (kcut) | Δ zero−open | word head open → room 1 s |
|---|---|---|---|---|---|
| masc27w-s3 (1.0 s, sentence) | ~.668 (masc7lf16) | .6754 | .6754 | 0 (no zeros; control) | .399 → .399 |
| masc27w15-s3 (1.5 s, sentence) | ~.626 (m15x5) | .6664 | .6509 | −.016 | .406 → .341 |
| masc27w15kg-s3 (1.5 s, clip) | ~.577 (m15kgx5) | .6117 | .6829 | **+.071** | .414 → .328 |

**Verdict.** Measured: zero-tail costs the sentence-trained 1.5 s model .016 (below the 1.5% bar's lower end, CI
±.03; directional), the 1.0 s model nothing, and *helps* the clip-trained model by .071 (it was trained on exactly
these zeros; continuous MEG is the off-distribution condition for it). Under the simulated Kaggle condition the
ranking is w15kg .683 > w1.0 .675 > w15 .651 — **not** the public order (1.0 > w15 > w15kg). The isolated-word
1 s room costs the 1.5 s word heads .065-.086 on those rows; at 92/960 rows that is ≈ .007-.008 of public
(inference, linear weighting). Inference: zero-tail explains at most ~.02 of w15's .042 public gap and none of
w15kg's .09-.11 gap (kcut .683 vs public .577); the hypothesis is not supported as the main cause. Untested suspects
for the w15kg gap: Kaggle epoch construction beyond length (per-epoch decimation edges, `begin` bound at epoch
start, how pnpl cut the ≤9-word stretches), or public-row sampling of the clip model's specific errors.
