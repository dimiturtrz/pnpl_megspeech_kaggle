# Deep error anatomy of m11xsp7, span ends and the Kaggle cut (2026-10-09)

Follows [2026-10-08_error_anatomy.md](2026-10-08_error_anatomy.md) (masc7lf16/ENS23). This one: the 18-member blend
m11xsp7 (11 `ema-cd25-masc27w-s*` + 7 `ema-cd25-xsp10sem05-*`), equal-weight mean of context-head log-probabilities,
tau 0.5. It is a legacy `val = official` model: VAL = pnpl validation (selected), LOCAL TEST (LT) = pnpl test (reports).
Scratch scripts (not committed): `C:/Users/User/.claude/jobs/44912819/tmp/anat/`.

**Denominator unless stated:** novel-span words of Sherlock1 + TheMoth. Sentence cut: VAL 2512 (1296 + 1216), LT 2133
(1358 + 775). Metric = balanced top-10 accuracy, macro over the two domains. "Headroom" = macro points regained if a
bucket reached recall 1; "gap" = points regained if it reached the mean. Total headroom VAL 34.1, LT 30.7 points.

**Reproduction:** a CPU fp32 re-forward under the saved sentence cut gives VAL .6591 / LT .6929, matching the logs exactly.

## 1. Where the points are (sentence cut)

| cut | VAL | LT |
|---|---|---|
| TheMoth vs Sherlock1 recall | .584 vs .734, gap 3.75 | .648 vs .739, gap 2.27 |
| rarest train-prior quartile | 68% of weight, headroom 26.4, gap +3.2 | 63% of weight, headroom 20.2, gap +0.9 |
| domain-classes with n = 1-4 | 24 cells, 63 words, headroom 12.9, gap +4.2 | 25 cells, 57 words, headroom 11.0, gap +2.6 |
| both heads miss | headroom 30.1 | headroom 22.3 |
| word head hits, context head misses (head-choice ceiling) | 4.0 | 8.4 |

Worst-10 words differ between VAL and LT except `has`. Per-word ranks are small-n noise (inference).

## 2. Span-last words: ≤1.7 points, and mostly an artefact of our cut

- Span-last recall .506 (VAL) / .547 (LT) vs middle .68 / .71. Weight .10, gap **+1.61 / +1.54 points** → the ceiling
  for any span-end fix is ≈1.7 points; any realistic partial recovery falls below the 1.5-point bar.
- Under the sentence cut, 183/250 (VAL) and 144/213 (LT) span-last words are *mid-sentence*: a >9-word PNPL sentence
  split into equal pieces (`Spans.by_sentence`). Kaggle epochs are never split mid-sentence like that. Truly
  sentence-last words: LT n=69, recall .62, gap +0.35.
- Overlapping-span inference (megspeech-7gf.163) cannot transfer. Each Kaggle sentence epoch is its own MEG array
  that ends at the sentence, in shuffled order (`pnpl/competition/holdout.py`). Joining epochs needs the
  `shuffle_seed`/timing loophole the organisers flagged (`docs/competition/rules.md:57`). Closed as not transferable.

## 3. Lone words and short spans

The Kaggle Deep test has 92 isolated-word epochs (9.6% of rows, see the 2026-10-08 doc §4); the local sentence cut has
almost none. Local spans of ≤3 words: LT n=51, recall .45, gap +0.94; VAL n=12. Under the Kaggle cut every span is
shorter (mean ≈5.6 words), so these conditions are no longer rare.

## 4. The Kaggle cut as a ruler (megspeech-7gf.132 note)

Re-scored with `--grouping kaggle` (pieces of `KAGGLE_WORDS`, MEG window bounded at the piece end), same 18 members:

| | sentence | kaggle | Δ |
|---|---|---|---|
| VAL macro | .6591 | .6287 | −.030 |
| LT macro | .6929 | .6726 | −.020 |
| LT Sherlock1 / TheMoth | .739 / .647 | .710 / .636 | −.029 / −.012 |
| LT span-last hit rate (unweighted) | .676 (n=213) | .594 (n=283) | |
| LT other words hit rate | .823 | .794 | |
| member mean VAL, M vs XF | .660 vs .641 | .620 vs .619 | M lead .019 → .001 |
| member mean LT, M vs XF | .682 vs .693 | .664 vs .669 | XF lead .011 → .005 |

- Public for m11xsp7 is .680–.692. It sits between the two rulers. Public SE ≈ .023 cannot pick one.
- The VAL lead of M over XF disappears under the Kaggle cut. A kind decision taken on the sentence-cut VAL can
  therefore reflect the cut, not the kind (inference). H and D kinds were not in m11xsp7 and were not re-scored.
