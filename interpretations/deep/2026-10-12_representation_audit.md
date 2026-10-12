# Representation audit: where word information appears inside one member, and what crowds it (megspeech-d6r.3)

One member of blend `m11xsp7hub6d2v3`: `ema-cd25-xsp10sem05-s4-ft-s5` (legacy full-data: VAL = pnpl official
validation, LOCAL TEST = official test). A blend has no single hidden state, so the audit hooks one member.
The scratch scripts (not committed) are in `C:/Users/User/.claude/jobs/44912819/tmp/rep/`:
- `capture.py` uses forward hooks and changes no package code;
- `harness.py`, `audit.py`, `checks.py`, `summary.py` and `geometry.py` do the analysis.

Everything ran on CPU, inside a 5 GB RAM cap.

## Setup (what the numbers are)
- **Scored words.** The unseen-span words that perword scores. MOCHA is not counted.
  - LT: **2133** (Sherlock1 1358, TheMoth 775).
  - VAL: **2512** (Sherlock1 1296, TheMoth 1216).
- **Plumbing checks.**
  - The blend through `perword.Scene` gives LT macro bacc10 **.697**. The harness's bacc10, run on the same scores,
    gives .6970.
  - This member's own context head, with T and tau fit on VAL, gives **.693** [.668, .744].
  - The heads re-applied to the captured tensors reproduce the captured logits, so the hooks sit on the right tensors:
    - `words.head(pool)` reproduces the word logits: max abs error 1.2e-3 (float16 storage), argmax agreement 100%;
    - `net.head(headin)` reproduces the context logits: argmax agreement 99.96%.
  - Two hook pairs are identical by construction: `b5.mm ≡ pool` (MeanMax pool) and `L3 ≡ headin` (no final norm).
- **Stages.**
  - Time-resolved maps (input after the subject layer, the spatial conv, blocks b0..b5) are reduced to two views:
    `.bins` (5 adaptive-average time bins × channels) and `.mm` (mean and max over time, which is what the pool sees).
  - `pool` is the 512-d word embedding.
  - `mix` is the transformer input: `project(pool)` plus the timing encoding. The timing encoding is clock = *order*
    (span position) and intervals = *log* onset/finish intervals. There are no fusion branches.
  - `L0..L3` are the transformer layers.
  - `word` and `ctx` are the two heads' logits.
- **Probe.** Standardise, then PCA, then Ledoit-Wolf shrunk LDA (`shrunklda.Lda`).
  - The PCA size is chosen from {64, 128, 256, 512} by VAL bacc10.
  - It reads `s/T + log prior`, with T fit on VAL by NLL. Tau ∈ {0, .5, 1} is fit on VAL.
- **Training rows for the probe.**
  - Fit on **20,000** training rows, a subsample of 25,997 captured labelled rows from 15 training runs:
    - one middle run per Sherlock book (8 books in the 15 runs);
    - 6 TheMoth runs, all of whose rows are kept.
  - The subsampling is forced by the RAM cap. The member trained on about 150 runs.
- **Bits.** `log2 p(y|x) − log2 prior(y)` per LT word, averaged naturally.
- **CIs.** 95%, span-block bootstrap: 2000 multinomial span-count draws per domain. Paired deltas use the same draws.
  A delta is *directional* unless its CI excludes 0 and it is ≥ .015.
- **Nuisance probes.**
  - A 2-fold span-parity cross-fit inside the 20k training rows, PCA ≤128.
  - Bits are measured against the fit-fold prior. The table shows bits / entropy of the target.
  - The run target has 15 classes (3.72 bits) and source has 2 (0.84 bits). Subject is not applicable: only subject 0
    has labels.

## 1. Layer-wise word information (LT)

| stage | width | bits [CI] | probe bacc10 [CI] | VAL-fit probe | run η² | word η² |
|---|---|---|---|---|---|---|
| in.bins | 1530 | .036 [.022, .050] | .276 [.246, .308] | .269 | .001 | .004 |
| in.mm | 612 | .000 | .221 | .217 | .159 | .005 |
| spatial.bins | 1280 | .065 | .300 | .264 | .003 | .007 |
| b0.bins / .mm | 1280 / 512 | .120 / −.001 | .283 / .219 | .274 / .236 | .018 / .210 | .009 |
| b1.bins / .mm | | .162 / .042 | .317 / .226 | .251 / .250 | .046 / .214 | .011 |
| b2.bins / .mm | | .131 / .138 | .330 / .297 | .261 / .235 | .100 / .228 | .014 |
| b3.bins / .mm | | .205 / .152 | .349 / .330 | .284 / .261 | .126 / .224 | .017 |
| b4.bins / .mm | | .249 / .297 | .355 / .393 | .281 / .268 | .116 / .191 | .023 |
| b5.mm = **pool** | 512 | **.317** [.273, .361] | **.378** [.346, .414] | .309 | .148 | .040 |
| **mix** (transformer input) | 256 | **.658** [.607, .708] | **.527** [.492, .563] | .398 | .095 | .098 |
| L0 | 256 | .884 | .615 | .448 | .089 | .100 |
| L1 | 256 | 1.021 | .632 | .489 | .083 | .111 |
| L2 | 256 | 1.180 | .641 | .496 | .068 | .120 |
| L3 = headin | 256 | 1.291 [1.201, 1.379] | .662 [.638, .711] | .501 | .057 | .134 |
| *word head (member)* | 50 | .446 [.397, .493] | .400 [.360, .428] | | | |
| *ctx head (member)* | 50 | 1.454 [1.363, 1.548] | .693 [.668, .744] | | | |

Paired steps along the `.mm` chain, Δ probe bacc10 with [CI]:

| step | Δ bacc10 [CI] |
|---|---|
| b1 → b2 | +.071 [+.029, +.105] |
| b3 → b4 | +.063 [+.023, +.090] |
| b4 → b5 | −.015 [−.039, +.017] (the only negative step; directional) |
| **pool → mix** | **+.149** [+.111, +.186] (the largest step) |
| mix → L0 | +.089 [+.062, +.116] |
| L0 → L3 | +.046 in total; each layer step is directional |

**Readout lever** (does a probe on the penultimate layer beat the trained head?):

| comparison | Δ bacc10 [CI] |
|---|---|
| probe(pool) − word head | −.023 [−.050, +.020] |
| probe(headin) − ctx head | **−.031 [−.062, −.003]** (the head wins) |
| probe(ctx logits) − ctx head | −.014 [−.051, +.013] |
| VAL-fit probe(pool) − word head | −.091 [−.118, −.032] |
| VAL-fit probe(headin) − ctx head | −.192 [−.224, −.145] |

## 2. Nuisance information per stage (bits / target entropy)

| stage | word | run (3.72) | source (0.84) | position (1.11) | duration quintile (2.32) | duration\|word (2.32) | previous word (2.91) |
|---|---|---|---|---|---|---|---|
| in.bins | .019 | .011 | .010 | .053 | .017 | .012 | −.004 |
| b2.mm | .081 | **3.250** | .837 | .070 | .116 | .098 | .005 |
| **pool** | **.334** | **3.008** | **.842** | .131 | .154 | .135 | .046 |
| mix | .661 | 2.828 | .842 | **1.109** | .213 | .184 | **.822** |
| L3 | 1.364 | 2.435 | .841 | 1.108 | .554 | .492 | .857 |
| ctx logits | 1.460 | 1.220 | .804 | .699 | .235 | .197 | .401 |

Run-centring: Δ probe bacc10, centred minus plain, with [CI].
- How centring was done:
  - Training rows are centred per run.
  - VAL and LT are centred per domain, using the mean of that domain's held-out rows. Each official VAL/LT domain is
    one run.
  - This is **transductive**: it uses the test rows' own mean.

| stage | Δ bacc10 [CI] |
|---|---|
| pool | +.023 [−.010, +.055] (directional) |
| L1 | +.030 [+.003, +.048] (one of 25 stages tested, with no multiplicity correction; treat as directional) |
| headin / L3 | **−.056 [−.077, −.018]** |
| word logits | −.027 [−.054, −.001] |
| ctx logits | −.003 [−.015, +.010] |
| all other stages | inside ±.03, every CI spans 0 |
| **member's word head on centred pool** (pool − run mean + train mean, VAL-calibrated) | **+.009 [−.017, +.026]**: Sherlock1 −.009, TheMoth +.028 |

## 3. Cross-narrator transfer

The probe is fit on 5,418 rows from one source (equal sizes) and scored on each LT domain. Transfer is the off-source
score as a fraction of the in-source score on the same test domain, both measured above the chance level of a top-10
ruler on 50 classes (.2): `(off − .2) / (in − .2)`. These are point estimates; the per-row hits were not saved, so
there are no CIs.

| stage | Sherlock-fit on Sherlock1 | TheMoth-fit on Sherlock1 | transfer to Sherlock1 | TheMoth-fit on TheMoth | Sherlock-fit on TheMoth | transfer to TheMoth |
|---|---|---|---|---|---|---|
| b2.mm | .299 | .247 | .47 | .253 | .241 | .77 |
| b4.mm | .339 | .294 | .68 | .256 | .243 | .77 |
| **pool** | .363 | .270 | **.43** | .298 | .256 | **.57** |
| mix | .493 | .364 | .56 | .387 | .319 | .64 |
| L0 | .579 | .407 | .55 | .518 | .417 | .68 |
| L3 | .653 | .562 | .80 | .543 | .443 | .71 |
| ctx logits | .680 | .664 | .97 | .592 | .535 | .85 |

## 4. Per-word geometry at the pool (`geometry.json`)
- **Fisher ratio.** Squared nearest-other-mean distance over within-class spread, in the probe's whitened 256-d space.
  It tracks rarity, not decodability:
  - Spearman with train share −.91;
  - with pool bits −.42.
  - Rare words sit far from their neighbours only because their means are estimated from few rows. It is an
    estimation artefact, not separability.
- **Pool bits vs context-head bits** per word: Spearman −.02. The words the encoder carries are not the ones the
  context head gets right.
- **Classification.** Per word, using the paired hit gap against 2·SE. There are 50 tests, uncorrected, so expect
  about 1 false positive.
  - **13 readout-limited**: the probe beats the matched head on the pool or the head input. Examples: they, were,
    there, have, for, she, he, in, was, we, this, at, a.
  - **6 signal-limited**: pool bits < .1 and head hit10 < .6. These are has, good, be, not, can, people.
  - **31 neither.**
  - Most "readout" wins are small (+.03 to +.19), on words of mid frequency.

## Reading
- **No word information is lost inside the member; it is never there to begin with.**
  - Probe bits rise almost monotonically: input .04, pool .32, head input 1.29.
  - The only negative step (b4 → b5, −.015) is directional.
  - The MEG-driven encoder delivers just **.32 bits** (probe bacc10 .38) at the pool. The member's own word head reads
    .45 bits.
- **The biggest single gain is pool → mix: +.149 bacc10, +.34 bits.**
  - The only things added there are a linear projection of the pool and the timing encoding: span position and log
    onset/finish intervals.
  - The nuisance probes say what came in:
    - span position jumps from .13 to 1.11 bits (its full entropy);
    - previous word jumps from .05 to .82 bits.
  - So of the 1.29 probe bits at the head input, only .32 are present in the per-word MEG embedding. The rest arrives
    with timing/structure (where in the span, how long the gaps are) and the transformer's mixing of neighbours. They
    are not more single-word MEG evidence. The split is inferred, not ablated; see below.
  - This matches the d6r.1 budget's finding that the context and prior channels carry the macro.
- **Run identity dominates the encoder.**
  - At the pool, a 15-way run probe recovers **3.01 of 3.72 bits** and source **.84 of .84 bits**, against **.33 bits**
    of word.
  - Run means explain **14.8%** of pool variance; word means explain **4.0%**.
  - The `.mm` (mean/max) view carries the run offset from the first block on (η² .21). The time-binned view does not
    (η² ≤ .13). The DC/max statistic is where the run fingerprint sits.
- **Centring does not cash out this capacity.**
  - At the pool the gain is directional (+.023).
  - The real word head gains +.009 [−.017, +.026].
  - At the transformer output it hurts (−.056). That is expected: those features were trained uncentred, so removing
    the LT mean removes span-level context, not only run identity.
  - The run information looks orthogonal to the word directions, as an LDA would see them. A shared-covariance probe
    already ignores a pure run offset, which is why removing it buys little.
- **There is no readout lever.**
  - A shrunk-LDA probe on the penultimate layer does not beat the trained heads: ctx −.031 [−.062, −.003], word −.023
    (directional).
  - Even a probe on the logits themselves does not beat the head (−.014).
  - The VAL-fit probes, about 2.5k rows, are far worse: −.09 and −.19. Read that as the size of the training-row
    advantage, not as a lever.
- **Cross-narrator transfer breaks at the encoder.**
  - It is lowest at the **pool**: .43 to Sherlock1, .57 to TheMoth.
  - It recovers through the transformer to .85–.97 at the context logits.
  - The word evidence the MEG encoder produces is mostly narrator- or session-specific.
  - What transfers is the timing/position structure added at `mix` and refined after it.
  - This is consistent with run and source being near-saturated at the pool.

## What this does NOT show
- **One member, not the blend.** Blend averaging may change the per-word picture. The member's head (.693) is close to
  the blend (.697).
- **A linear-Gaussian probe is a lower bound on information.** "No readout lever" means no lever *for a shrunk-LDA
  readout trained on 20k rows from 15 runs*. The heads were trained on about 150 runs, with the encoder adapting to
  them.
  - A nonlinear head, or a head refit on all training rows, is not tested.
  - Train-fit probes also see rows the encoder trained on, which favours them; they still lose.
- **Timing vs MEG at mix is inferred, not ablated.** The split comes from the nuisance probes, not from zeroing the
  timing encoding. The ablation would be: rerun the forward with `onsets(...)` zeroed and probe mix/L3 again. That
  needs a hook that replaces an output; it is cheap and is the next test.
- **The run-centring test is transductive.** It uses each LT domain's own mean.
  - The owner allows Broad EA-style per-run whitening. Whether Deep may use test-run statistics is **undecided**.
  - Nothing here argues for spending a submission on it anyway: the real-head delta is null-sized.
  - Uncentred-trained features were centred post hoc. Training *with* `RunCentring(active=True)` is a different
    question and is not answered here.
- **Cross-narrator has no CIs**, and its fits are small (5,418 rows). "Transfer" mixes narrator, session and recording
  date: each source has its own sessions.
- **Duration bins** come from aligned word END − ONSET, which the model does not see directly. The interval features
  see onsets and finish.
- **Not run:**
  - a second member (the RAM/time budget went to the full 25-stage pass);
  - a GPU nonlinear probe.
