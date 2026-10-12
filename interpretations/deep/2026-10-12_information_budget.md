# Information budget: where the bits are, what the model captures, which channels are unexploited (megspeech-d6r.1)

Blend `m11xsp7hub6d2v3` (27 members, legacy full-data: VAL = pnpl official validation, LOCAL TEST = official test).
CPU-only scratch analyses (not committed) that import the studies' classes: `crowding.Saved/Section/Finishes`,
`perword.Place`, `lmbound.Bound/Table`, `lmsoft.Dumped`, `templatemeans.Sums/Reliability`, `templates.Experiment`.
Four parts: E1 bits budget, E2 confusion GLM, E3 cross-narrator templates, E4 channel table.

## Setup (what the numbers are)
- **Words.** The unseen-span words scored by perword. MOCHA is excluded (fewer than 100 novel words).
  - VAL: **2512** words (Sherlock1 1296, TheMoth 1216).
  - LOCAL TEST (LT): **2133** words (Sherlock1 1358, TheMoth 775).
- **Sanity check.** `Finish(tau=0.5).scores` on LT reproduces perword exactly:
  - macro bacc10 **.697** (Sherlock1 .738, TheMoth .656);
  - plain top-10 **.811**;
  - top-1 **.345**.
- **Calibration.**
  - Every model below is a logit `z = Σ θ_k·feature_k − θ_b·log_prior`. θ is fit by LBFGS on VAL only and then applied
    unchanged to LT. This is the "temperature per head" plus the prior-correction weight.
  - Two fits:
    - **balanced** weights 1/(n_class·K·D), which matches bacc;
    - **natural** weights (per word).
- **Bits.**
  - `info_bal` = log2 50 (5.644) − the balanced-weighted CE of the balanced fit. This is the metric-aligned number.
  - `info_nat` = CE(training prior) − the word-weighted CE of the natural fit.
  - The training prior's natural CE is 4.89 bits on LT, so the prior alone gives 0.76 bits over uniform when words are
    weighted naturally. Under balanced weighting the prior's CE is 6.30, worse than uniform: it hurts, as tau already
    implies.
- **CIs.** 95%, span-block bootstrap (1000 multinomial span-count draws per domain). Paired deltas use the same draws.
- **LM channels** come from GPT-2-large PMI tables (lmbound / lmsoft).
  - "full" gives the LM the true neighbouring text, OOV words included. This is an oracle that cannot be built: the
    test has no text.
  - "backoff" masks OOV neighbours and gives in-vocabulary neighbours as truth. Also an oracle.
  - "lmsoft-oracle" is close to backoff.
  - "lmsoft-soft" is the realistic one: the neighbours' identities come from the model's own beliefs.
  - "-right" adds the right context.

## E1 — bits budget (LT; VAL in brackets for the point; [95% CI] on LT)

| model | info_bal (bits) | info_nat (bits) | bacc10 | VAL info_bal / bacc10 |
|---|---|---|---|---|
| uniform | 0 (CE 5.644) | — | — | — |
| training prior | −0.65 (CE_bal 6.30) | 0 (CE_nat 4.89) | — | −0.70 |
| timing-only (other model, VAL, top-1 MI proxy) | **0.63** top-1 MI | — | top-1 .267 | see note |
| word head | 0.309 [0.262, 0.356] | 0.448 [0.401, 0.490] | 0.424 [0.384, 0.460] | 0.211 / 0.375 |
| context head, 1 member | 1.208 [1.143, 1.379] | 1.403 [1.315, 1.491] | 0.679 [0.657, 0.730] | 1.081 / 0.652 |
| **context head (blend)** | **1.301 [1.231, 1.478]** | 1.478 [1.387, 1.563] | 0.696 [0.672, 0.746] | 1.137 / 0.649 |
| ensemble ctx + word | 1.293 [1.220, 1.480] | 1.478 [1.387, 1.564] | 0.694 [0.669, 0.743] | 1.151 / 0.650 |
| ctx + LM full/pmi (oracle) | 2.453 [2.325, 2.659] | 2.173 | 0.869 [0.849, 0.896] | 2.158 / 0.824 |
| ctx + LM full/pmi-right (oracle) | 3.712 [3.534, 3.915] | 3.205 | 0.949 [0.937, 0.966] | 3.326 / 0.912 |
| ctx + LM backoff/pmi (oracle in-vocab) | 1.612 [1.515, 1.798] | 1.614 | 0.748 [0.726, 0.793] | 1.417 / 0.713 |
| ctx + LM backoff/pmi-right | 1.923 [1.791, 2.115] | 1.816 | 0.779 [0.760, 0.822] | 1.707 / 0.750 |
| ctx + LM lmsoft-oracle/pmi-right | 1.909 [1.778, 2.106] | 1.808 | 0.776 [0.757, 0.821] | 1.697 / 0.750 |
| **ctx + LM lmsoft-soft/pmi-right (realistic)** | 1.289 [1.218, 1.473] | 1.473 | 0.694 [0.669, 0.743] | 1.143 / 0.653 |

**Timing-only note.** This row uses a different denominator and a different quantity.
- `models/confusion/timing_counts.npy` holds top-1 counts only, with no probabilities. So its CE cannot be recovered.
- The number shown is the plug-in MI of the true × predicted top-1 joint on VAL (2512 words, model sent40-ioi), minus a
  label-permutation null:
  - plug-in 1.106 − null 0.473 = **0.63 bits**.
- The same proxy for comparison:

| Model | Top-1 MI (proxy) |
|---|---|
| Another real model (VAL) | 0.81 bits |
| Ours, VAL | 0.91 bits |
| Ours, LT | 0.92 bits |

- So timing alone carries about **70%** of our top-1 MI (0.63 / 0.91). This matches 7gf.71: 0.32 of 0.445 above chance
  is timing.
- Top-1 MI is a lower bound on the information in a model and is not on the same scale as info_bal.

**Paired deltas, LT (info_bal bits; bacc10).**

| Comparison | info_bal (bits) | bacc10 |
|---|---|---|
| Context − word head | +0.992 [0.933, 1.161] | +.272 [.240, .333] |
| Blend − 1 member | +0.093 [0.062, 0.122] | +.017 [−.000, .032] |
| Ensemble (ctx + word) − ctx | −0.008 [−0.015, 0.014] | −.002 [−.016, .013] |
| LM backoff/pmi-right − ctx | +0.622 [0.496, 0.719] | +.083 [.057, .110] |
| LM full/pmi-right − ctx | +2.410 [2.193, 2.544] | +.253 [.205, .280] |
| LM lmsoft-soft/pmi-right − ctx | −0.012 [−0.019, −0.000] | −.002 [−.011, .004] |

- Context − word head: 77% of the bits come from context and timing, not from the word's own window.
- Ensemble − ctx: the word head adds nothing on top of the context head.

The VAL deltas have the same signs: backoff-right +0.570 bits / +.100 bacc, soft-right +0.006 / +.004.

**By domain, LT, context head (info_bal; bacc10).**

| Domain | info_bal (bits) | bacc10 |
|---|---|---|
| Sherlock1 | 1.595 | .730 |
| TheMoth | 1.007 | .662 |

On VAL the TheMoth gap is wider: 1.494 vs 0.781 bits.

**By position and class, LT, context head.** Word-weighted uniform-referenced info of the balanced fit; bootstrap CI;
hit10 = plain top-10.

| slice | n | info (bits) | hit10 |
|---|---|---|---|
| first in span | 404 | 2.03 [1.84, 2.20] | .817 |
| middle | 1516 | 1.96 [1.85, 2.06] | .806 |
| **last in span** | **213** | **1.14 [0.92, 1.36]** | **.653** |
| function (Attributes.function) | 2054 | 1.89 [1.79, 1.98] | .790 |
| content | 79 | 1.98 [1.65, 2.34] | .873 |

- The word head alone shows no last-word penalty: first .88, middle .60, last .61 bits. The −0.8 bits at the last word
  is lost right context, not a weaker word code.
- The oracle LM with right context has last at 3.00 bits vs middle 4.02.

**Bits → bacc10 (empirical).**
- Across the 12 models on LT: about **0.14 bacc10 per bit** (least squares). It saturates: 0.13/bit from ctx to
  backoff-right, about 0.10/bit beyond 2.5 bits.
- Per class (43 classes with n ≥ 5, context head): hit10 = 0.476 + 0.179·info, Spearman 0.78.
- The word head is steeper (0.50 per bit, Spearman 0.92) because it sits on the low, linear part of the curve.
- Use 0.13-0.14 per bit to convert a near-current-model channel into bacc.

## E2 — confusion GLM (LT top-1 errors)

**Data.**
- 50×50 LT confusion; **1397** off-diagonal errors out of 2133 words.
- Soft version: calibrated probability mass off the diagonal. The temperature on the tau-finished context score is
  T = 1.32, fit on VAL. Off-diagonal mass = 1679.

**Model.** Poisson quasi-independence on the 2450 off-diagonal cells: log μ_ij = a_i + b_j + β·X.

**Covariates**, standardised:

| Covariate | Definition |
|---|---|
| same_pos | Same part of speech |
| phoneme_edit | Normalised Levenshtein over the first 4 phonemes |
| dlogdur | \|Δ log duration\| |
| dlogioi | \|Δ mean log IOI-to-next\| on LT |
| lm_subst | Mean over LT occurrences of true i of logP_LM(j) − logP_LM(i) (full/pmi left context) |
| sem_cos | GPT-2-large wte cosine (first token) |

**Fit.**
- Hard: marginals-only null deviance 2334 → 1746 with all six covariates (−588). Pearson φ = 1.19.
- Soft: 802 → 402.

**CIs** come from 200 span-block bootstrap draws, refit each time. A drop-one CI is the percentile CI of a
non-negative statistic, so the quasi-χ² test (Δdev/φ against 3.84 for 1 df) is the significance check.

| covariate | hard β [CI] | hard drop-one Δdev [CI] | Δdev/φ | soft β [CI] | soft Δdev [CI] |
|---|---|---|---|---|---|
| dlogioi | −0.61 [−0.74, −0.48] | **60.9 [36.9, 95.5]** | 51.3 | −0.33 [−0.38, −0.28] | 23.8 [15.6, 32.7] |
| dlogdur | −0.40 [−0.51, −0.29] | **40.4 [20.5, 63.9]** | 34.0 | −0.33 [−0.38, −0.28] | 34.5 [25.5, 44.2] |
| same_pos | +0.12 [0.06, 0.18] | 14.5 [3.4, 32.0] | 12.2 | +0.15 [0.12, 0.17] | 25.9 [16.7, 34.1] |
| lm_subst | +0.41 [0.16, 0.62] | 14.4 [2.2, 33.2] | 12.1 | +0.32 [0.24, 0.39] | 10.6 [5.8, 16.4] |
| sem_cos | +0.11 [0.03, 0.20] | 4.7 [0.4, 16.7] | 4.0 (borderline) | +0.06 [0.03, 0.09] | 2.0 [0.4, 4.7] |
| phoneme_edit | −0.02 [−0.08, 0.04] | 0.7 [0.0, 8.9] | 0.6 (none) | −0.02 [−0.03, 0.00] | 0.5 [0.0, 2.2] |

Correlated covariates: dlogdur and dlogioi r = .65; same_pos and sem_cos r = .44; lm_subst and sem_cos r = .36.

**Timing-only confusion** (VAL, `models/confusion`):
- Off-diagonal row-rate Pearson with ours: **.62** on VAL (Spearman .57); .30 with our LT.
- Another real model vs timing: .56. Ours vs that real model: .73.
- Adding the timing confusion rate as a covariate to the hard GLM: Δdev = 28.7, on top of the six covariates.

**Spectral clusters.** Soft LT, eigengap k = 4:

| Cluster | Words |
|---|---|
| A — prepositions, pronouns, conjunctions | to it he that they of there and in but for at out him as on |
| B — auxiliaries, pronoun-auxiliaries | is was we be have this will my she were am it's had has can new |
| C — adverbs, quantifiers, content | not so all any really very time think always people |
| D — determiner-like | the a i are our an do good |

The hard LT+VAL clustering (k = 7) gives the same skeleton: {a, are, do} stays a tight group at every k, and so do
{the, to, of, in, for, at, our, an} and {any, really, very, good}.

## E3 — cross-narrator per-word templates

**Data.**
- Raw-MEG per-word templates from `templates.Experiment` (8 reps), over 156 training runs (subject 0).
- Words with ≥ 20 trials in both groups: 50.
- Median trials per word: Sherlock 2672, TheMoth 238.
- "Centred" removes the across-word mean template (the word-agnostic evoked response).
- Ceiling = sqrt(SB reliability_S × SB reliability_M), from split halves.
- CIs: word bootstrap, 2000 draws.

| pair (centred) | cross same-word r | other-word r | SB rel A | SB rel B | ceiling | cross / ceiling | cross-ID top-10 (chance .20) |
|---|---|---|---|---|---|---|---|
| Sherlock–TheMoth | .120 [.076, .163] | −.004 | .490 [.421, .556] | .141 [.083, .199] | .232 [.177, .289] | **.52 [.38, .66]** | .52 (top-1 0/50) |
| Sherlock–TIMIT | — | — | — | — | — | .43 [.30, .58] | .45 |
| TheMoth–TIMIT | — | — | — | — | — | .62 [.36, .95] | .49 |

- Without centring (Sherlock–TheMoth): same .230 vs other-word .148, difference .082 [.033, .132]; ratio .64
  [.55, .75].
- The median per-word disattenuated ratio is .48. The per-word mean is unusable: near-zero ceilings blow it up.

**Verdict.** The word code is **partly shared, partly narrator/story/session-specific**.
- About half of the reliable word-specific template transfers across stories. The ratio is well above 0, and its CI
  clearly excludes 1.
- Cross-identification is above chance at top-10 (.52 vs .20) but not at top-1.
- Together with 7gf.116: TheMoth's flat trial-averaging curve against Sherlock's steep one matches TheMoth's low
  split-half reliability (.14 vs .49). TheMoth has about 11× fewer trials per word.
  - With 238 trials the TheMoth template is mostly noise.
  - The domain gap (1.01 vs 1.60 bits LT) is consistent with data starvation on a code that is about half shared, and
    does not need a qualitatively different code.

## E4 — information channels

**How the columns are built.**
- "Available" = the measured upper bound in bits (info_bal on LT unless stated), with its bacc10 conversion.
- "Captured" = what the current blend or a tested method realises.

**Legality basis.**
- rules.md: "Not addressed: external data, pretrained models, language models, use of stimulus text, test-time
  adaptation."
- Rule 4b: no "hand labeling or human prediction of the validation dataset or test data records".
- Announcement 741919: mapping via `word_onset_s` / `shuffle_seed` is cross-track leakage.
- OWNER restrictions (ours, not theirs):
  - no fitting on unlabelled test MEG;
  - no shuffle_seed/onset mapping;
  - no cross-subject test pooling;
  - LM prior and GPT-2 text targets approved 2026-10-08.

| # | channel | bits available | captured now | legality | status | rank (bits × feasibility by Oct 16) |
|---|---|---|---|---|---|---|
| 1 | word-sequence LM over in-vocab neighbours (realistic beliefs) | +0.62 [0.50, 0.72] bits / +.083 [.057, .110] bacc (backoff-right oracle); +0.31 [0.23, 0.37] left only | −0.012 [−0.019, 0.000] (lmsoft-soft) | LM approved by owner; rules "not addressed" | lmsoft/lvj: soft beliefs too weak (in-vocab neighbour q(true) .08); 7gf.60 BERT harmful; 7gf.6 gate unmet | 1 by bits; feasibility low. The bound needs the neighbours' identities, which is the decoding problem itself; joint/iterative decoding is untried |
| 2 | narrator/TheMoth gap (data starvation) | TheMoth 1.01 vs Sherlock 1.60 bits LT → about 0.3 macro-bits ≈ .034 macro bacc if closed | about half of the template shared (E3 .52) | legal (training data only) | 7gf.47 voice balance null; 7gf.109.6 open | 2; needs more TheMoth-like data or invariance training. Not doable by Oct 16 |
| 3 | right context for last-in-span words | 213/2133 words × 0.82 bits ≈ 0.08 bits mean; upper bound about .015 hit10 (.153 × .10) | +.002 LT (route-b, 7gf.164) | legal; Kaggle epochs end at sentence end (7gf.163) | mostly captured as far as reachable | 3; small ceiling |
| 4 | ensemble size / members | 1 → 27: +0.093 [0.062, 0.122] bits, +.017 [−.000, .032] bacc | captured | legal | done; saturating | 4; more members would give diminishing returns |
| 5 | class marginals / Sinkhorn | not measured in bits; +.008 local, +.002 / +.011 public | partly (157.13) | needs test-row coupling; owner OK'd Sinkhorn, nothing beyond it (f4o) | 7gf.29.3, 157.13 | 5; cheap, legal, small |
| 6 | timing / IOI | about 0.63 bits top-1 MI on its own (70% of ours) | captured (IOI in the model; ablation −3.6pt LT) | legal as within-row timing; onset→stimulus mapping is forbidden (741919) | 7gf.71, 7gf.110 (sub-20 ms word-specific) | already in; drives the confusions (E2 dlogioi/dlogdur largest) |
| 7 | POS / syntax auxiliary target | +.010 / +.012 directional | partial | legal | 7gf.93 directional | small |
| 8 | word head on top of context | −0.008 [−0.015, 0.014] | nothing left | legal | E1 | none |
| 9 | a/an and next-word phonology | ≤ about 0.07 bits mean (about 7% of words, ≤ 1 bit each); E2 phoneme_edit adds Δdev 0.7, so no phonological confusion structure | 0 | the next word's phonemes need its text (OOV) or neural decoding of it | untried | low |
| 10 | OOV neighbour positions from onset gaps | already in as gap_prev/gap_next (IOI) | captured | gaps OK; mapping onsets to the stimulus is not | 7gf.53 killed (OOV neighbours) | — |
| 11 | pre-onset predictive signal | lead 0.1 s .609 vs .617 (in band); not measured in bits | 0 | legal | 7gf.115/38; 0.5 s and 3 s leads untested | unknown, cheap to test |
| 12 | frequency bands (high gamma) | high-gamma LDA .23 vs low .32; fusion −1.1pt | 0 | legal | 7gf.80; 16c and x30 open | low |
| 13 | repetition / test trial averaging | Sherlock centroid .38 → .98 at k = 32 (7gf.116; class sets shrink with k) | 0 | needs test-row coupling, so the owner must OK it; same-word grouping needs test labels (5e6) | f4o kNN +.002 | not reachable without labels |

## Reading
- **E1.** The blend carries about **1.30 bits** of metric-aligned information per word on LT, out of 5.64.
  - About 77% of that comes from context and timing, not from the word's own window (word head 0.31 bits).
  - Timing alone already accounts for about 70% of our top-1 MI. This is a proxy on a different model and a
    different denominator, but it agrees with 7gf.71.
- **The LM is the only large pool of bits.**
  - The oracle in-vocabulary LM adds +0.62 bits (+.083 bacc). With the true OOV text it adds +2.4 bits (+.25).
  - The realistic soft-belief LM captures none of it (−0.012 bits).
  - The gap is a neighbour-identity problem, not an LM problem: the LM needs words the decoder cannot yet name.
- **E2.** Confusions are organised by timing first: similar IOI and duration are the largest drop-one terms.
  Syntactic/LM substitutability comes second (same POS, lm_subst). Phonology does not contribute: phoneme edit
  distance adds Δdev 0.7.
  - The model confuses words that occur in similar temporal and syntactic slots, not words that sound alike.
  - This fits a decoder that mostly reads timing and context rather than an acoustic-phonetic code.
- **E3.** About half of the word-specific MEG template is shared across Sherlock and TheMoth; the rest does not
  transfer. TheMoth's weak code is consistent with having 11× fewer trials.

## What this does NOT show
- **Timing-only CE is unmeasured.** Only top-1 counts were saved. The 0.63-bit MI is a plug-in proxy with
  permutation bias correction, on VAL, from a different model (sent40-ioi). It is not comparable to info_bal.
- **The covariates are coarse.**
  - phoneme_edit uses only the first 4 phonemes (unknowns stripped).
  - sem_cos uses the first wte token only.
  - A null phonology result is about this ruler. It does not rule out finer acoustic similarity.
- **Narrator, story and session cannot be separated.** All datasets are subject 0, with different sessions and
  stimuli. E3's "narrator-specific" part may equally be story- or session-specific.
- **The LM rows mostly use oracle text and cannot be built.** They bound the bits available; they do not show a
  reachable gain. lmsoft-soft is the only realistic arm.
- **Sinkhorn, pre-onset, frequency bands and repetition** were not measured in bits here. Their rows quote bead results
  in bacc.
- **The bits → bacc slope is model-to-model and saturating.** It is a conversion for small changes near the current
  model, not a law.
- LT here is pnpl's official test (legacy full-data model). It is a local test, not the Kaggle public or final test.
