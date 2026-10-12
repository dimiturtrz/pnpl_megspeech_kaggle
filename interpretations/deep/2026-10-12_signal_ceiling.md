# Signal ceiling: how much word identity a linear read of the MEG carries, and where (megspeech-d6r.2)

A model-free question: how much word-identity information can a linear decoder read straight from the processed MEG,
at which latencies, sensors and bands, and do the model's window and receptive field cover it?
The comparison model is `m11xsp7hub6d2v3`: LT macro bacc10 .697 (Sherlock1 .738, TheMoth .656), top-1 .345.

The scratch scripts are in the job tmp dir `sig/` and are not committed. They import `shrunklda.Lda`,
`templates.Noise` and `metric.Ruler`. The engine is a GPU sufficient-statistics LDA with Ledoit-Wolf shrinkage:
- Every window is a sub-block of one within-class scatter.
- It agrees with the in-RAM `Lda.fit` to 3e-5 (on scores of size ~59).
- Peak VRAM was 0.9 GB and RAM stayed under 5 GB.

## Setup (what the numbers are)

**Words.** The perword novel-span split; MOCHA is excluded.

| Split | Words | Sherlock1 | TheMoth | Used for |
|---|---|---|---|---|
| train | 307,335 | 265,113 (Sherlock) | 24,510 (TheMoth) | fit |
| VAL | 2,512 | 1,296 | 1,216 | calibration and selection |
| LOCAL TEST (LT) | 2,133 | 1,358 | 775 | reporting |

The train domain counts are each dataset's full pool. The train total also includes rows from other sources, which
count only toward the pooled fits. Every number below was fit on train, calibrated and selected on
VAL, and reported on LT.

**Features.**
- Processed 64-component recordings at 50 Hz, epoched from −0.5 s to +1.5 s around word onset.
- Usually reduced to 40 ms bin means, so a 1 s window is 25 bins × 64 = 1600 dimensions.
- **masked**: samples outside the sentence epoch are zeroed. **cont**: no zeroing.
- No row's epoch ends before +1 s (stop < +1 s for 0.0% of rows in every split). So masked and cont are the same
  post-onset, and the mask only changes the pre-onset part (begin > −0.5 s for 20–25% of rows).

**Metric.**
- bacc10 is the house `Ruler` computed on uniform-prior LDA scores, then macro-averaged over the two domains.
- Chance is **0.20**.

**Bits.**
- Calibration fits `logp = log_softmax(β·s + γ·log_prior)` on VAL CE.
- Bits are the LT CE reduction over the VAL-calibrated label-only prior.
- "Pooled" calibration fits one (β, γ) on both domains. Pooled calibration leaves TheMoth slightly negative (~−0.03)
  because Sherlock dominates. So the headline bits are **per-domain calibrated**: each domain's (β, γ) and prior baseline
  are fit on that domain's VAL rows.

**CIs.** 95% span-block bootstrap within each domain, 1000 draws. Paired deltas use the same draws.

**Reading rule.** An effect is "directional" unless its CI excludes 0 and it is at least about .015.

## M1: information vs latency

### Sliding 100 ms windows (5 samples, step 40 ms; masked; word target)

| Window centre (s) | VAL bacc | LT bacc | bits S1 | bits Moth | bits macro |
|---|---|---|---|---|---|
| −0.22 (pre) | .232 | .238 | .007 | .004 | .006 |
| +0.06 | .234 | .260 | .027 | .023 | .025 |
| +0.10 | .212 | .248 | .031 | .020 | .025 |
| +0.26 | .237 | **.296** (LT peak) | — | — | .021 |
| +0.30 | .230 | .278 | .047 | −.001 | .023 |
| **+0.38** | .252 | .289 | .049 | .014 | **.032** (bits peak) |
| +0.66 (2nd bump) | .209 | .282 | .021 | .029 | .025 |
| +0.90 | .248 | .231 | −.001 | −.007 | −.004 |

- Bits are per-domain calibrated.
- Pre-onset windows sit at LT .20–.25 and below .007 bits.
- Information rises from the +0.02 s window, which spans 0 to +0.1 s.
- It is above .02 bits from +0.02 s to +0.70 s.
- It returns to about 0 after +0.85 s.

**Previous-word target, as a control.** Training n = 97,619 rows that have an in-vocabulary previous word (VAL 815,
LT 676).
- The previous word is decodable before the current onset (cont −0.38 s: LT .326) and early after it (+0.06 to +0.26 s:
  .25–.31).
- This is the previous word's own evoked response, still present.
- So the current word's early window partly contains the previous word.

### Cumulative windows (masked unless noted; per-domain-calibrated bits)

| Window (s) | LT macro bacc [95% CI] | S1 | Moth | VAL bacc | bits S1 / Moth / macro |
|---|---|---|---|---|---|
| [0, 0.4) | .307 [.269, .339] | .365 | .249 | .284 | .127 / .025 / .076 |
| [0, 0.8) | .325 [.287, .352] | — | — | .284 | .161 / .027 / **.094** |
| [0, 1.0) | .342 [.298, .365] | .419 | .265 | .278 | .162 / .018 / .090 |
| [0, 1.2) | .332 [.287, .353] | — | — | .293 | .155 / .017 / .086 |
| [0, 1.48) | .318 [.272, .343] | — | — | .287 | .141 / .009 / .075 |
| [−0.2, 0.8) (VAL-selected) | .339 [.296, .366] | .398 | .280 | .288 | .156 / .030 / .093 |
| [−0.2, 1.0) | .344 [.299, .368] | — | — | .278 | .158 / .021 / .090 |
| [−0.48, 1.0) | .347 [.301, .374] | .431 | .264 | .283 | .154 / .015 / .084 |
| model-like [0, 1), centred | .336 [.288, .357] | .410 | .262 | .274 | .159 / .015 / .087 |
| cont [−0.2, 1.0) | .353 | — | — | — | — |

- Accumulation **saturates by about 0.8–1.0 s**. Windows longer than 1.0 s add nothing: bits fall slightly, because more
  dimensions overfit.
- Adding pre-onset context does not move bits (−0.48 start .084 vs 0 start .090).
- The model-like [0, 1) window is within .01 of the best window.
- The ceiling agrees with 7gf.69, where an LDA on [0, 1) scored .3247.

## M2: sensors and bands (cont; [0, 1); bits vs broadband on LT)

| Feature | Dims | VAL bacc | LT bacc | LT bits | Bits vs broadband [CI] |
|---|---|---|---|---|---|
| broadband 40 ms bins | 1600 | .278 | .342 | .043 | — |
| + envelope delta | 2240 | .277 | .340 | .040 | −.003 [−.012, +.006] |
| + envelope theta | 2240 | .251 | .297 | .033 | −.011 [−.020, −.001] |
| + envelope alpha | 2240 | .267 | .284 | .042 | −.001 [−.010, +.008] |
| + envelope beta | 2240 | .269 | .335 | .039 | −.005 [−.013, +.004] |
| + all envelopes | 4160 | .244 | .321 | .030 | −.013 [−.030, +.003] |
| envelopes alone (each band, all) | 640 / 2560 | .17–.20 | .21–.27 | ≈0 | ≈ −.043 |
| evoked band-passed delta | 1200 | .277 | .271 | .032 | −.011 [−.034, +.013] |
| evoked theta / alpha / beta | 1200 | .19–.21 | .18–.23 | ≈0 | ≈ −.044 |
| region: Left-temporal | 400 | **.290** | .246 | .040 | −.003 [−.030, +.024] |
| region: Right-temporal | 400 | .253 | .292 | .027 | −.017 [−.039, +.005] |
| region: Left-parietal | 400 | .274 | .241 | .022 | −.021 [−.047, +.003] |
| region: Left-frontal | 400 | .264 | .220 | −.003 | −.046 [−.072, −.022] |
| region: Right-parietal | 400 | .241 | .276 | .011 | −.032 [−.055, −.008] |
| region: occipital L/R | 400 | .25 / .21 | .25 / .25 | ≈0 | ≈ −.042 |
| region: Right-frontal | 400 | .246 | .237 | .001 | −.042 [−.067, −.020] |

Region and band-evoked rows use their own decompositions (16 region-whitened components; 24 components at 20 ms), so
their "vs broadband" values compare different feature sets on the same rows.

- **All the linearly readable information is in the slow evoked response, delta below 4 Hz.**
- Theta, alpha and beta evoked responses and every band envelope are at chance.
- No band adds bits on top of broadband. Every CI includes 0 or is negative.
- **Left temporal carries the most information** (400 dimensions recover the broadband bits: −.003 [−.030, +.024]). Then
  come left parietal, right temporal and left frontal (on VAL). Occipital and right frontal are about 0.
- High-gamma cannot be measured at 50 Hz; see 7gf.80 and qi7.1.

## M3: bits beyond timing (LT, macro; bits per-word over the scale-calibrated covariate model)

The covariates are known per word at test time: log duration, log IOI to the next and previous word (with missing flags),
log position, log sentence length, and first/last flags. "+prev" adds the previous word's label as a one-hot. That is an
oracle; it is not available at test.

| Model | LT bacc (S1 / Moth) | LT bits |
|---|---|---|
| covariates: timing | .548 (.619 / .476) | .55 (VAL bacc .501) |
| covariates: timing + prev label (oracle) | .678 | .93 |
| MEG [0, 1) masked | .342 | .043 |
| MEG residualised on timing | .282 | .010 |
| MEG residualised on timing + prev | .266 | — |

| MEG window joined to covariates | Bits beyond timing [CI] | Bits beyond timing + prev [CI] |
|---|---|---|
| post [0, 1) masked | +.006 [−.011, +.021] | −.002 [−.017, +.012] |
| pre [−0.48, 0) cont | −.003 [−.010, +.005] | −.004 [−.009, +.002] |
| full [−0.2, 1.2) cont | +.009 [−.008, +.024] | −.003 [−.018, +.012] |

- **Timing alone beats the linear MEG read by a wide margin** (.548 vs .342 bacc; .55 vs .04 bits).
- **Once timing is known, the linearly readable MEG adds no bits whose CI excludes 0** (+.006 [−.011, +.021]).
- Residualising the MEG on timing keeps part of its bacc (.282 > .20). The bits drop to .010, and the joint model does
  not improve on timing.
- Reading: most of what a linear LDA reads from the MEG is word length or duration and rhythm, which the evoked
  response encodes. It is not lexical identity beyond that.

## M4: cross-narrator transfer ([0, 1) masked; LT bacc [CI])

| Train pool (n) | on Sherlock1 | on TheMoth |
|---|---|---|
| Sherlock matched (24,510) | .288 [.254, .329] | .222 [.188, .280] |
| TheMoth (24,510, its full pool) | .263 [.234, .303] | .209 [.184, .256] |
| all matched (24,510) | .277 [.244, .319] | .252 [.210, .293] |
| Sherlock full (265,113) | .387 [.338, .431] | .279 [.213, .317] |

| Paired delta | Delta [CI] |
|---|---|
| Sherlock- minus Moth-trained, on Sherlock1 | +.024 [−.018, +.064] |
| Sherlock- minus Moth-trained, on TheMoth | +.013 [−.051, +.077] |
| Sherlock full minus matched, on Sherlock1 | **+.100 [+.042, +.142]** |
| Sherlock full minus matched, on TheMoth | +.056 [−.031, +.099] |

The per-word TheMoth recall correlation between Sherlock-trained and Moth-trained probes is 0.49.

- **In-domain training gives no advantage at matched n.** Moth-trained probes do no better on TheMoth than
  Sherlock-trained ones; the delta points the other way and includes 0.
- The linear word code transfers across narrators, subjects and sessions as well as it holds within one.
- **Volume is what pays.** 11× more Sherlock rows add +.10 on Sherlock1, with a CI that excludes 0. On TheMoth the gain
  is +.056, directional.
- TheMoth's low ceiling (.21–.28) comes from the Moth recordings: their signal and reliability (7gf.116, 7gf.119). It
  does not come from a narrator mismatch.

## M5: is there a lever? Probe plus model (LT)

| | VAL bacc | LT bacc |
|---|---|---|
| model (context head blend) | .654 | .697 (S1 .738, Moth .656); calibrated bits 1.47 |
| VAL-selected probe [−0.2, 0.8) masked, fusion weight .201 | +.0053 gain | **−.0081 [−.0187, +.0021]** |
| best LT probe window [−0.48, 0.8) (not VAL-selected) | — | +.0063 [−.016, +.020] |
| per-latency fusions | ≤ +.01 | within ±.025 (noise) |

**Agreement.**

| Domain | Both hit | Model only | Probe only |
|---|---|---|---|
| Sherlock1 | .399 | .432 | .047 |
| TheMoth | .218 | .559 | .045 |

The per-word recall correlation between model and probe is −0.11 on Sherlock1 and 0.13 on TheMoth.

- **No lever.** Model + probe has no window whose LT CI excludes 0.
- The probe's hits are almost a subset of the model's (4–5% probe-only).
- The model already reads everything the linear probe reads, and about 2× more.

## Reading

1. **Information vs latency.** Word-identity information from a linear read starts at onset, at the +0.02 s window. It
   peaks at **+0.26 to +0.38 s** (LT bacc .296, .032 bits per 100 ms), has a second bump near +0.66 s, and is gone by
   about +0.85 s.
2. **Saturation.** Cumulative windows saturate by **0.8–1.0 s** (LT .34, ~.09 bits, VAL-best [−0.2, 0.8)). The model's
   [0, 1) window covers the readable signal; extending it is not a lever.
3. **Where.** Left temporal, then parietal. Slow evoked delta only. Envelopes and higher bands add nothing linearly.
4. **Timing.** The linear MEG word code is mostly timing. Beyond the duration, IOI and position covariates it adds
   +.006 [−.011, +.021] bits. Timing alone (.548 bacc) already beats the whole linear MEG read (.342).
5. **Transfer.** Transfer is not narrator-limited: in-domain training gives +0 at matched n. Data volume dominates
   (+.10 [.04, .14] on Sherlock1 from 11× data).
6. **Lever.** The model, at .697, is about 2× the linear ceiling (.34) and contains it. Fusing the probe gives
   −.008 [−.019, +.002]. The model's remaining gap to perfect is not something a linear evoked read can supply.

## What this does NOT show

- **Linear LDA only.** A nonlinear or deep read of the same features can and does (the model) get far more. ".34" is the
  ceiling of shrinkage-LDA on these features, not an information-theoretic ceiling of the MEG.
- **50 Hz cap.** Nothing above 25 Hz was measured, so high-gamma is unmeasured here; see 7gf.80 and qi7.1.
- **Mask.** masked equals cont post-onset in this data. The mask test only covers the pre-onset part.
  - The "masked" window is a linear zeroing approximation of the model's epoching, not its exact input.
- **Timing covariates.** These include word duration. The competition releases `word_onset_s`, so IOIs are derivable at
  test. I did not find duration in the competition docs, so the timing covariate may be partly an oracle.
  - "Bits beyond timing" still answers the information question.
  - The prev-label covariate is an oracle: the test has no labels.
- **Calibration.** Pooled and per-domain calibration differ for TheMoth (pooled is slightly negative, per-domain ~+.02).
  The bits depend on which one is used; bacc does not.
- **LT bacc noise.** LT bacc is noisy, with CI widths around ±.04, and VAL and LT orderings of windows disagree (for
  example region Left-temporal: VAL .290 vs LT .246).
  - The window and region rankings are directional. Only the large contrasts are findings: timing vs MEG,
    volume vs domain, model vs probe.
- **TheMoth's low numbers** reflect its recordings (7gf.116, 7gf.119). This study cannot separate fewer subjects or
  sessions from lower SNR.
- **Limits.** Nothing was skipped for RAM or VRAM: every planned measurement ran within 5 GB RAM and an 8 GB VRAM cap.
