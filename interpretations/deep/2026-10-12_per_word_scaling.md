# Per-word data scaling: the macro curve is concave, and the per-word split is mostly single-seed noise (megspeech-d6r.4)

Scratch scripts (not committed): `C:/Users/User/.claude/jobs/44912819/tmp/scale/{score_gpu,score_fast,analyse,hetero,consistent,counts,inv2}.py`.
No training. The scale models of megspeech-7gf.104 are rescored with the package `LocalTest` path (val tau, then
`record` val + local test). Issue: megspeech-d6r.4. Builds on 7gf.104 (about +3 points per doubling) and the
m11 per-word table (`2026-10-10_per_word_decodability.md`).

## Setup (what the numbers are)
- **Models.** `scale-share{25,50}-s0` (one seed each) and `scale-share100-s{0..5}` (six seeds). All are trained on
  subj0 only, with nested session subsets (`Sources.thinned`) and the balanced ruler partitions (80/10/10). tau is
  0.5 for all eight.
- **Rows.** LOCAL TEST is pnpl validation + test pooled, novel-span words, in domains with at least `min_novel` novel
  words. share100-s0 has **28,186** test words: Sherlock 24,722 + TIMIT 391 + TheMoth 3,073. share25 also scores
  MOCHATIMIT (127 words) because its novel set is larger. VAL is the carved val, **28,961** words for share100-s0.
  Every test-side number is replicated on val.
- **Ruler.** Balanced top-10 with tie-pessimistic hits, per domain, then the mean over domains.
- **Per-word rate.** Hit10 is pooled over domains, so it is Sherlock-dominated. Bits are log2 p(true) + log2 50,
  after one temperature fit on val NLL per model (T = 1.00 / 0.90 / 0.85 for 25/50/100-s0).
- **Training count per word.** From `Heard.of` at each share. Counts double cleanly between shares, e.g. `the`
  8.3k / 16.6k / 33.2k at 25 / 50 / 100%. At 100%, `new` (261) is the rarest word and `the` (33,169) the most frequent.
- **Slope.** d(hit10) / d(log2 n_train_word).
  - 50→100 uses the 6-seed mean at 100 against single-seed r50.
  - SE = sd_seed(100) · sqrt(1 + 1/6) / log2(ratio). This assumes the seed sd at 50 (and 25) equals the measured sd at
    100; it probably underestimates the low-data seed noise.
  - 25→100 is the same formula over 2 doublings.

## Macro: real and concave (headline, clear of seed noise)
| share | test macro | val macro |
|---|---|---|
| 25 % | 0.4796 | 0.5060 |
| 50 % | 0.5514 | 0.5628 |
| 100 % (6 seeds) | **0.5895 ± 0.0066 sd** | 0.5951 ± 0.0056 |

- The test numbers match 7gf.104: .4775 / .5520 / .5830 for s0, and our s0 is .5831.
- 25→50 adds **+.072** and 50→100 adds **+.038**. Concavity ratio 0.53 (test); val gives .057 then .032, ratio 0.57.
  Each step is many seed sd's, so the curve and its bend are real. One caveat: share25 and share50 are single seeds.
- **Extrapolation, at subj0-equivalent data:**

  | Assumption | 2× (200%) | 4× (400%) | Cumulative at 4× |
  |---|---|---|---|
  | Geometric decay (ratio 0.53 per doubling) | +0.020 | +0.011 | **+0.031** |
  | Constant slope (upper bound; the curve is visibly bending) | +0.038 | +0.076 | |

  - With ratio 0.53 per doubling the limit is about +0.04 above share100: the curve saturates at about 0.63 on this
    ruler.
  - Production Deep (m11, val = official) already trains on about 1.25× the share100 sessions (+0.3 doubling) and
    warm-starts from MASC. Its remaining 2× headroom is therefore below +0.020. This is directional: it applies the
    scale models' ratio to a different recipe.

## Per word: the two halves of the curve disagree
- **Heterogeneity is real but not stable.**
  - Across 50 words the 50→100 slopes exceed the seed SE: chi2 = 161 on 49 df, 13 words at |z| > 2 against 2.3
    expected (val: 199, 12).
  - The 25→50 slope correlates **−0.64** with the 50→100 slope (val −0.57). Words that dipped at the single-seed
    share50 point "steepen" afterwards.
  - Examples: `is` .554 → .428 → .607, `at` .657 → .511 → .644, `do` .230 → .681 → .558.
  - That is a single-seed redistribution of top-10 slots among confusable function words (is/it/at/an/do/can), not
    a per-word data limit. The 50→100 steep/flat list (15 steep, 6 flat, 29 noisy at 2 SE) is therefore **not
    reported as a finding**.
- **Val-test agreement is not independent evidence.** The slopes agree at Spearman 0.78 (50→100) and 0.83 (25→100),
  but the same single-seed share25/50 models score both, so the r25/r50 noise is shared.
- **Robust lists.** These are words whose 25→100 slope clears 2 SE on test and val, and whose two half-slopes keep
  the same sign on both.

  STEEP (10 words, 9 function + 1 content). Columns: 25→100 slope per doubling, test ± SE / val ± SE, then n_train
  at 100%, n_test, r25 → r100.

  | word | test slope | val slope | n_train | n_test | r25 → r100 |
  |---|---|---|---|---|---|
  | so | +.162 ± .024 | +.139 ± .032 | 2,514 | 260 | .34 → .66 |
  | all | +.156 ± .044 | +.116 ± .045 | 2,830 | 268 | .24 → .55 |
  | out | +.151 ± .024 | +.137 ± .027 | 1,750 | 209 | .35 → .65 |
  | am | +.138 ± .043 | +.147 ± .045 | 870 | 100 | .28 → .55 |
  | this | +.129 ± .017 | +.167 ± .011 | 3,608 | 375 | .40 → .66 |
  | any | +.127 ± .031 | +.150 ± .027 | 839 | 93 | .43 → .69 |
  | not | +.126 ± .030 | +.118 ± .040 | 3,289 | 359 | .48 → .73 |
  | it | +.102 ± .018 | +.114 ± .015 | 9,423 | 1,064 | .54 → .74 |
  | for | +.092 ± .023 | +.094 ± .034 | 4,397 | 491 | .50 → .69 |
  | i | +.084 ± .017 | +.087 ± .017 | 15,451 | 1,740 | .63 → .79 |

  Pooled 25→100 slope over all 50 words: test .042, val .061.

  FLAT (25→100 slope + 2 SE below the pooled slope, on test and val):

  | word | test slope | val slope | r100 | Pattern |
  |---|---|---|---|---|
  | the | +.006 ± .006 | +.008 | .89 | at ceiling |
  | and | +.016 ± .006 | +.022 | .87 | at ceiling |
  | of | +.012 ± .013 | +.021 | .82 | at ceiling |
  | but | +.004 ± .018 | +.020 | | U-shaped through 50% (.77 → .70 → .78) |
  | at | −.007 ± .021 | −.009 | | U-shaped through 50% (.66 → .51 → .64) |

  - the/and/of are flat for headroom: hit10 is near 0.9 and the remaining errors are each other.
  - but/at are flat end to end but swing about 15 points through the single share50 seed. Their rates are set by
    the class reallocation, not by count.
  - Macro weight: steep 0.185, flat 0.104, the other 35 words 0.711.
- **The SE assumption matters.** If seed noise at 25% is larger than at 100% (likely), even the STEEP list is
  generous. Treat it as **directional**; a second seed at 25/50 decides it (see Next).

## What steepness relates to: nothing measured
Spearman of the 50→100 slope against each property (n = 50):

| Property | test | val |
|---|---|---|
| log2 n_train | +.03 | −.08 |
| level r100 | +.03 | −.01 |
| function word | −.07 | −.18 |
| duration | +.13 | +.23 |
| m11 hit10 (local test, n median 26) | −.05 | +.03 |
| n_test | +.01 | −.10 |

- None clears noise, so it is not a data-count story: the steep ten span 839 to 15,451 training examples.
- Mean slope is .044 over 40 function words and .058 over 10 content words. With 10 content words that is not a
  finding.
- The steep words sit at mid level (r100 .55 to .79). Words at 0.9 have no headroom, and the lowest
  (they .33, will .44, were .44) move erratically.
- Per-word bits follow hit10 (`this` +.59 ± .15 bits per doubling, `not` +.95 ± .34). Bits SEs are 2 to 4× larger
  relative to the effect, so they add nothing beyond hit10.

## Reweighting prediction ("only the steep words get more examples") — not licensed
- **Arithmetic.**
  - Steep weight 0.185 × mean steep slope 0.127 = +.023 at constant slope, or about +.012 with the 0.53 decay.
  - The full per-word reconstruction over-predicts the measured macro step by 25% (.047 vs .038), because pooled
    per-word rates are not the per-domain balanced macro. Scaled down, that gives **+.010 to +.019**.
  - All words doubled gives +.020 geometric, so steep-only would carry roughly half the gain at about 20% of the
    data.
- **Why the curve does not license it.**
  1. Collinearity: every word's count doubles together with the total data, the session count, the speaker hours
     and the context. A word's own-count slope cannot be separated from the general-representation slope.
     Steepness does not track own count (ρ ≈ 0), which points to the shared part.
  2. Oversampling is not new data. Repeating a steep word's existing examples adds no new MEG. The owner forbids
     oversampling read runs. Class-balance loss weighting is a different mechanism: it moves the prior or the
     margin and was not measured here.
  3. New data would have to arrive as whole sentences, so "more `so` and `this`" means more of every word anyway.
- **What it licenses.** More subj0-like data helps with a concave return of about +2 points for the next doubling.
  The gain concentrates in mid-level function words (so/all/out/this/not/it/for/i/am/any), not the ceiling words
  (the/and/of).

## Legal data not yet in Deep training (vocab-word tokens for the 50 words)
Rules: `docs/competition/rules.md` says "Not addressed: external data, pretrained models, language models, use of
stimulus text, test-time adaptation". 4b forbids "hand labeling or human prediction of the validation dataset or
test data records". The 2026-09-18 announcement makes cross-track mapping disqualifying. The restrictions below are
**ours** (owner):
- Deep excludes subj1–32 ses-11/12, while Broad may use them.
- Never exploit subj1–32 transcripts.
- No fitting on unlabelled test MEG.

Denominator: subj0 train = **307,335** vocab tokens (765k words in total). The multiplier is the source's count
divided by subj0's count, per word.

| Source | subjects | vocab tokens | per-word × subj0 (median, max) | steep-ten range | status for Deep |
|---|---|---|---|---|---|
| subj0 held-out (pnpl val+test) | 1 | 5,733 | 0.017 | 0.008–0.032 | refit=all already; ≈ +0.0006 |
| subj1–32 Sherlock1 ses-11 pool | 32 | 41,853 | 0.11 (0.38) | 0.03–0.17 | **Broad only** (ours) |
| subj1–32 Sherlock1 ses-12 | 32 | 43,645 | 0.11 | 0.01–0.23 | **Broad only** (ours) |
| MASC | 27 | 131,614 | 0.39 (1.18) | 0.09–0.69 | already used: m11 warm-start pretraining |
| Kymata | 21 | 80,514 | 0.29 (3.13) | 0.00–0.74 | only in bgmix "all" pretraining arms |
| Appleseed | 12 | 30,022 | 0.067 | 0.00–0.13 | only in bgmix "all" arms |
| SCANS, SMN4Lang | — | 0 | — | — | non-English |
| LPP ds007523 | — | 0 | — | — | French, 478 GB, not on disk |
| Armeni 2022 | 3 | (10 h Sherlock each) | — | — | owner dropped 2026-10-08 (DUA denied); also overlaps Sherlock1 text |
| MOUS | — | — | — | — | Dutch, dropped |

- **No legal source doubles subj0.**
  - MASC + Kymata + Appleseed together is 0.79× subj0 in tokens. They are other subjects on other scanners, so a
    token there is not a subj0 token.
  - The scale curve is single-subject and does not price cross-subject tokens. Their exchange rate is the bgmix or
    warm-start measurement, not this curve.
- The only same-subject addition (the held-out sessions) is 1.9% and already in refit.

## Next
- Train `scale-share{25,50}-s{1,2}`, 4 runs on GPU. This decides whether any per-word steepness is real: a stable
  word has the same-sign slope across seeds at 50%.
- Price class-balance weighting on the steep ten as its own mechanism, separately from data. Its expected gain is
  small: the steep ten already score .55–.79.
