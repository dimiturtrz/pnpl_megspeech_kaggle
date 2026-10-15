# MEG data augmentation (and time-frequency inputs) for word decoding — field synthesis

**Scope.** What augmentations the field has used, what evidence exists, and which of them are physically valid for
OUR input. Our input is single-subject LibriBrain MEG (PNPL 2026 Deep track, BAcc@10 over 50 words). It was
Maxwell-filtered by the provider (temporally non-extended SSS) and so has rank about 75-93 of 306 sensors. It is
decimated to 50 Hz, cut into 1 s windows from word onset, and fed to a CNN encoder with a sentence-transformer head.
The train set is about 307k labelled words. This document covers external literature plus our own closed results
from `.beads/issues.jsonl`. It runs no new experiment.

**Verification tags.**
- **[FT]**: the claim was checked in the paper's full text, downloaded as PDF and searched.
- **[SUM]**: the claim comes only from a WebFetch summary of the full page.
- **[AB]**: the claim comes only from an abstract or search snippet. Treat these as unverified.
- **(arg)**: the claim is my own physical argument, not a measurement.

---

## 1. Headline

1. **The frontier MEG speech decoders use no data augmentation.**
   - Défossez 2023 [FT], d'Ascoli 2024/25 [FT, arXiv version], MEG-XL [FT] and the PNPL 2026 baselines [FT] all
     have zero augmentation sections.
   - The only augmentation-like regulariser in this line is Défossez's *spatial-attention (regional) dropout*.
     Removing it costs 1.7 pt on Gwilliams MEG and 0 on Schoffelen MEG [FT].
   - Their real levers were preprocessing: robust scaling plus clamping (without the clamp, Gwilliams falls from
     70.7 to 1.5), long windows, and the target representation.
2. **The best-documented LibriBrain 2025 augmentation ablation is null at its final configuration.**
   MEGConformer (1st place, phoneme Standard track) reports 87.58±0.55 without and 87.59±0.70 with MEGAugment
   over 10 seeds, p=0.16 [FT]. The lever there was the sliding-window stride: +2.4 F1.
3. **Rommel 2022, the canonical EEG augmentation benchmark, shows gains shrink with data.**
   - Gains are 5-12% at small n and about 1-2% at larger n [FT].
   - Its winners (TimeReverse, SignFlip, FTSurrogate) win on *oscillatory/spectral* tasks: sleep staging and
     motor imagery. On a *time-locked evoked* response they are physically invalid (see table).
4. **Our own arms are null or harmful.**
   - mask+noise (megspeech-cho): 0.6023, inside the seed band 0.598-0.619.
   - ±20 ms jitter: null. ±40 ms jitter: −5.4 pt (7gf.18).
   - ±1-sample TTA: −0.004 (7gf.39).
   - (arg) Our noise, per-sensor gain and per-sensor dropout all push the sample **off the rank-~80 SSS
     subspace**, which a first linear layer can trivially project away. Those nulls therefore never tested
     in-subspace robustness. This is the one specific, fixable reason the existing arms may have been weak
     *instruments*. It is a hypothesis, not a finding.
5. **Honest prior: no augmentation here is likely to deliver ≥1.5 pt on the local ruler.**
   - We are deep in Rommel's large-n regime: 307k words, one subject.
   - The more plausible payoff is on the *holdout shift*. 7gf.78 measured that the local ruler over-states the
     holdout by about 5 pt, and augmentation targets robustness more than in-distribution fit.
   - That shift cannot be measured locally. LB submissions are the only ruler for it.

---

## 2. Evidence by source

| Source | Setting | Augmentation | Measured effect | Tag |
|---|---|---|---|---|
| Rommel et al. 2022, J Neural Eng — https://arxiv.org/abs/2206.14483 · code https://github.com/eeg-augmentation-benchmark/eeg-augmentation-benchmark-2022 | SleepPhysionet (78 subj, 2 EEG ch); BCI IV 2a (9 subj, 22 ch). 13 transforms, learning curves, 10-fold CV | TimeReverse, SignFlip, GaussianNoise, SmoothTimeMask, FTSurrogate, FrequencyShift, BandstopFilter, ChannelsDropout/Shuffle/Symmetry, SensorsRotations | TimeReverse: up to +13% (sleep) and +25% (MI) relative, at small n. SmoothTimeMask: best for MI at large n, up to +20% relative at 60 windows/class, but hurts N2/N3 sleep. GaussianNoise: no robust gain in either task, and the worst time-domain transform in sleep. SignFlip: second-best in sleep, but equal to baseline in MI because the network's squaring layer already encodes sign invariance. FTSurrogate: top in sleep, and +45% relative on a 2^-3 MI subset. ChannelsDropout: significantly helps MI, especially at larger n. ChannelsShuffle/Symmetry: harm MI. SensorsRotation: not significant. "No single best augmentation"; gains 5-12% small-n vs 1-2% large-n | [FT] |
| Défossez et al. 2023, Nat Mach Intell — https://arxiv.org/abs/2208.12266 | 4 datasets (2 EEG, 2 MEG), 175 subj, contrastive to wav2vec2 | **None.** Spatial-attention dropout removes every sensor within distance 0.2 of a random location | Ablation without spatial dropout (top-10 seg acc): Broderick 17.7→16.0*, Brennan 25.7→26.8, Schoffelen 67.5→67.5, Gwilliams 70.7→69.0*. Clamp at 20 SD essential | [FT] |
| d'Ascoli et al. 2024/25 — https://arxiv.org/abs/2412.17829 (Nat Commun Nov 2025) | 723 subj, word decoding, 0.1-40 Hz, 50 Hz, 3 s from word onset, RobustScaler, clamp ±5 | **None** (zero mentions in arXiv text) | — | [FT] arXiv only; journal version not checked |
| MEG-XL — https://arxiv.org/abs/2602.02494 | Foundation model, masked-block pretraining (3 s blocks, 40% masked, 150 s context) | **None** besides the masking objective itself | Frozen MEG-XL scored 0.36 on our ruler (7gf.50) | [FT] |
| PNPL 2026 competition paper — https://arxiv.org/abs/2609.03231 | Our task: BAcc@10/50 words, chance 20% | None in baselines | Deep baseline (d'Ascoli) 73.23%; Broad baseline (MEG-XL) 42.96%. **Raw FIF is on HF** (`libribrain`, `libribrain2` repos), "not preprocessed and include head movement artefacts" | [FT] |
| LibriBrain100 — https://arxiv.org/abs/2608.25204 | Dataset paper | None | Raw BIDS/FIF released alongside h5; h5 is head-movement-corrected and realigned to a common reference head position, bad channels interpolated | [FT] |
| LibriBrain 2025 — https://arxiv.org/abs/2506.02098 | Dataset paper, sub-0 | — | Head placed within a few mm of a standard reference each session; temporally non-extended SSS; notch 50/100 Hz; 250 Hz | [FT] |
| MEGConformer (LibriBrain 2025, 1st phoneme Standard) — https://arxiv.org/abs/2512.01443 | Raw 306 ch @ 250 Hz | MEGAugment: 2 time masks ≤0.72 s each; band-stop (Theta/Alpha/Beta/Gamma/HGA, p=0.4); stride-60 sliding windows | Speech task, 10 seeds: aug 87.59±0.70 vs none 87.58±0.55 (p=0.16; earlier model versions +1.8). Stride: 85.18→87.59. Window 2.5 s vs 0.5 s: 87.59 vs 78.13. Instance norm essential for holdout | [FT] |
| MEBM-Phoneme — https://arxiv.org/abs/2603.02254 | LibriBrain 2025 phoneme | Onset jitter ±3 samples @250 Hz (±12 ms); same-class averaging | Jitter not ablated separately | [FT] |
| Robustness under distribution shift — https://arxiv.org/abs/2604.04129 | LibriBrain phoneme, test vs holdout | Each transform p=0.3: Gaussian noise 0.01·std; circular shift ±40 ms; time mask ≤80 ms; 10% channel dropout; amplitude U(0.9,1.1); band-amplitude [0.8,1.2] | Mixed. One augmented CNN went 39.53→47.74 test but **3.60 holdout**. Simpler augmented run: 34.03 test / 18.80 holdout. InstanceNorm CNN without aug: 60.95 test. InstanceNorm +13.48 pt | [FT] |
| Guo et al. 2023, mix-based aug for physiological signals — https://arxiv.org/abs/2309.09970 | 6 datasets incl. Sleep-EDF, MMIDB EEG; ResNet18-1D | Mixup, CutMix, manifold mixup | Beat baseline in 16 of 18 trials; exceptions: mixup on Apnea-ECG, CutMix on MMIDB-EEG. Per-dataset deltas not extracted | [AB]/[SUM] |
| BGMix (SSVEP) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12501431/ | 10 subj×8 ch and 35 subj×64 ch | Class template (trial mean) of class A + background (trial − template) of class B, label mixed λ~Beta | +4.8 to +25% vs no aug; reported above noise, TimeReverse, SmoothTimeMask, FrequencyShift, FTSurrogate, ChannelsShuffle, mixup | [SUM] |
| Braindecode augmentation API — https://braindecode.org/stable/api.html | Library | Transform list used in table below | — | [SUM] |
| Euclidean alignment + DA for BCI — https://arxiv.org/abs/2405.14994 | Cross-subject EEG | — | not read | [AB] |
| MNE movement compensation — https://mne.tools/stable/auto_examples/preprocessing/movement_compensation.html | Docs | SSS re-expansion at a reference head position (`maxwell_filter(head_pos=…, destination=…)`) | Mechanism reference for virtual head position | [SUM] |

---

## 3. Transform table

"Ours?" refers to `megspeech/training/regularise/inputs/augment.py` (`Augment`: `dropout`, `gain`, `jitter`, `mask`,
`band`, `noise`, all off by default) and to the beads results. Run prices assume our warm-start harness: about
6 min per sent40 seed, about 25 min for 4 seeds. 4 seeds is the minimum to read a 1.5 pt delta against a ~2 pt
seed band.

| Transform | Physical validity for SSS'd, time-locked MEG | Evidence | In our code? | Recommendation (price → plausible gain) |
|---|---|---|---|---|
| **SignFlip** | **invalid** (arg): field polarity encodes current direction; −B is a reversed-current source, not the same word. Rommel's sleep gain came from sign-agnostic spectral features | Rommel: helps sleep; = baseline in MI [FT] | no | **skip** — 0 h, negative expected value |
| **TimeReverse** | **invalid** (arg): breaks causality and the N100/N400 latency order the decoder reads | Rommel: biggest small-n gains, on oscillatory tasks [FT] | no | **skip** |
| **FTSurrogate** (phase randomisation) | **invalid** (arg): an evoked response *is* phase-locking (LB100 shows low-frequency ITC as the speech signature [FT]); randomising phase removes the signal | Rommel: top in sleep [FT] | no | **skip** |
| **FrequencyShift** | **invalid** (arg): shifts spectral content, so changes waveform shape and latency | Rommel: helps both of its tasks [FT] | no | **skip** |
| **BandstopFilter** | valid (arg): removing a band leaves a physically possible signal | Rommel: +17% MI low-n [FT]; MEGConformer band-stop part of a null aug [FT] | yes (`band`) | **skip as a lever** — its family was null in MEGConformer |
| **GaussianNoise, iid per sensor** | **questionable** (arg): iid noise is full-rank; data are rank ~80 after SSS, so ~226 of the noise dimensions are off-subspace and trivially filtered by the first layer | Rommel: no robust gain [FT]; ours (cho, with mask): null | yes (`noise`) | **replace** with in-subspace noise (shortlist #1) |
| **Covariance-matched / in-subspace noise** | **valid** (arg): noise drawn from N(0, Σ_train) or projected onto the train PCA top-k basis lives where real noise lives | no direct MEG evidence found | no | **try** (shortlist #1) |
| **Background mixing (BGMix-style)** | **valid** (arg): adds real non-word-locked MEG from the same subject; superposition is linear | BGMix +4.8–25% on SSVEP, small n [SUM] | no | **try** (shortlist #1 variant) |
| **AmplitudeScale (global)** | valid (arg): head-sensor distance and gain drift scale all channels | arXiv 2604.04129 bundle, mixed [FT] | no (only per-sensor) | cheap rider on #1; no standalone run |
| **Per-sensor gain** | **questionable** (arg): independent per-sensor gains leave the SSS subspace; real calibration drift was absorbed by SSS | none | yes (`gain`) | **skip** or project back into the subspace |
| **ChannelsDropout, iid per sensor** | **questionable** (arg): after SSS, any sensor is reconstructable from the others (rank 80/306), so the information is not removed | Rommel: helps 22-ch EEG [FT] (no SSS redundancy there) | yes (`dropout`) | **replace** with regional dropout |
| **Regional / spatial dropout** (all sensors within a radius) | **valid** (arg): mimics a cortical region being weak or noisy; forces spatially distributed codes | Défossez: −1.7 pt without it on Gwilliams MEG, 0 on Schoffelen [FT] | no; needs sensor positions, which the repo lacks (`cleaning.py` builds info without locations) | **try** (shortlist #2) |
| **ChannelsShuffle / ChannelsSymmetry** | **invalid** (arg): destroys spatial structure; language is lateralised; gradiometer pairs are not L/R mirrors | Rommel: harmful in MI [FT] | no | **skip** |
| **SensorsRotation (braindecode)** | **questionable→invalid** (arg): interpolates scalar values over rotated positions, ignoring that MEG coils measure an oriented field component | Rommel: not significant [FT] | no | **skip**; the physical version is SSS re-projection |
| **SSS re-projection / virtual head position** | **valid** (arg): fit internal multipole moments by least squares from the 306 channels with the true coil geometry, re-expand at a displaced or rotated head origin. Needs FIF `info` (coil defs + `dev_head_t`); raw FIF is on HF [FT, paper text; not downloaded] | MNE docs [SUM]; no augmentation paper found | no | **Deep track: skip.** Head is held within a few mm and data are realigned [FT], so little variance to simulate. **Broad track: candidate** |
| **ChannelsReref** | **invalid**: EEG-only (reference electrode) | — | no | **skip** |
| **SmoothTimeMask / time masking** | valid (arg): a transient loss of signal, though our 1 s window holds the word's key latencies | Rommel: best MI large-n [FT]; MEGConformer null [FT]; ours (cho): null | yes (`mask`) | **skip**, already tested null |
| **Onset jitter / temporal shift** | valid only below alignment error (arg); our onsets are precise to <20 ms | ours: ±20 ms null, ±40 ms −5.4 pt (7gf.18); MEBM ±12 ms unablated [FT] | yes (`jitter`) | **skip**, already closed |
| **Time warping** | **questionable** (arg): latency carries information; global stretch ≈ jitter at the response's tail, which we measured harmful | no MEG evidence found | no | **skip** |
| **Mixup (input or embedding)** | valid as a signal (arg): linear superposition of two MEG windows is physical. Label-mixing is the weak part: two words heard at once is not the brain state of either | Guo 2023: 16/18 wins across physiological sets [AB/SUM]; not in Rommel | no | **try** (shortlist #3) |
| **CutMix (time splice)** | **questionable** (arg): splice discontinuity; the 1 s window already overlaps neighbouring words | Guo 2023: lost on MMIDB-EEG [AB/SUM] | no | **skip** |
| **SegmentationReconstruction** (splice same-class segments) | **questionable** (arg): same reason as CutMix | braindecode only [SUM] | no | **skip** |
| **Same-word averaging** | valid (arg) but creates a train/test mismatch: test is single-trial | MEBM uses it for phonemes [FT] | no | **skip** for training |
| **Span-length augmentation** (draw span lengths from the holdout histogram) | **valid, in-domain**: it is a sampling change, not a signal transform | 7gf.78 (ours, proposed); 7gf.58 lone-span priced ≤+0.006 | partly planned | see note under shortlist |

### What is possible WITHOUT raw FIF or SSS moments (arg)

- The SSS subspace can be **estimated from the data**: take the top-k eigenvectors U of the train sensor
  covariance, with k ≈ the measured rank of 75-93.
- That is enough to:
  - draw in-subspace noise (U z, with z scaled by the eigenvalues);
  - project ANY existing augmentation back onto the subspace (x ← U Uᵀ x_aug), which repairs `noise`, `gain`
    and `dropout`;
  - run BGMix background mixing, which needs no geometry at all.
- Without geometry we cannot do physically correct head rotation or translation, or true regional dropout.
- Regional dropout needs only 2-D or 3-D sensor positions. Options:
  - read them from one HF raw FIF file;
  - use MNE's Neuromag layout (`Vectorview-all`). That this layout applies to Triux channel naming is
    **unverified**.

---

## 4. Ranked shortlist (3 to try next)

The bar is ≥1.5 pt BAcc@10 vs the original baseline. The seed band is 0.598-0.619, so every arm needs 4 seeds
(~25 min) to be read.

### #1 — In-subspace noise / background mixing (replaces our iid `noise` and `gain`)

- **Mechanism.** Train-time robustness to the noise the test actually contains. This is the only augmentation
  family where we have a concrete reason our previous null was a weak instrument rather than a refutation: our
  iid noise is mostly off-subspace (§3).
- **Variants**, all needing no geometry:
  - (a) noise ~ N(0, λ·Σ_train);
  - (b) add a random non-word-locked train window, scaled by α ~ U(0, 0.5) (BGMix without label mixing);
  - (c) project the existing `noise`/`gain` output back onto the train PCA basis.
- **Price.** About 1 h to implement and test, plus 25 min for 4 seeds per variant. Variant (b) is the one to run
  first.
- **Plausible gain.** Local: about 0-1 pt. Our SNR problem is per-trial neural variability, and real-background
  mixing makes it harder, not easier. BGMix's big gains were at tiny n (SSVEP).
- **Odds of clearing 1.5 pt locally:** ~10%.
- **Upside.** It is the most holdout-shift-relevant option: new sessions and recording noise.

### #2 — Regional spatial dropout (replaces our iid `dropout`)

- **Mechanism.** Removing a contiguous sensor patch is the only dropout that removes information after SSS,
  because iid dropout is reconstructable from the neighbours. It forces distributed spatial codes.
- **Evidence.** The only measured MEG speech number in the frontier line: +1.7 pt Gwilliams, 0 Schoffelen
  (Défossez [FT]). Gwilliams is the closest to us (naturalistic story listening).
- **Price.** About 1 h, mostly getting sensor positions from one HF FIF or the Vectorview layout, plus 25 min for
  4 seeds.
- **Plausible gain.** 0-1.7 pt. Défossez's +1.7 was multi-subject with a spatial-attention front end; within one
  subject the spatial code is fixed, so less to gain.
- **Odds of clearing 1.5 pt:** ~15%. This is the best-evidenced of the three.

### #3 — Mixup (input-level or embedding-level, labels mixed for the classification/contrastive target)

- **Mechanism.** Smooths the decision boundary between the 50 words. MEG superposition is physically linear, so
  the input side is valid. Label semantics are the weak point.
- **Evidence.** Only generic physiological-signal evidence (Guo 2023, 16/18 wins [AB/SUM]). No MEG or speech
  ablation found.
- **Price.** About 30 min (a few lines in the loss path) plus 25 min for 4 seeds.
- **Plausible gain.** 0-1 pt. With 307k samples, regularisation by interpolation usually buys tenths of a point.
- **Odds of clearing 1.5 pt:** ~5-10%. It is ranked last on evidence but is cheapest to implement.

### Not shortlisted but higher expected value for the LB

- **Span-length / holdout-histogram sampling (7gf.78)** targets the measured ~5 pt local-vs-holdout gap
  directly. If that gap is real, this beats every signal transform above on wall-clock ÷ gain. But it is a
  sampling-distribution fix, not an augmentation, and it is already tracked.
- **SSS virtual head position** belongs to the Broad track only.

**Overall honesty note.**
- The combined expected local gain of #1+#2+#3 is perhaps 1-2 pt, which sits at the bar.
- The field's evidence says augmentation is a small lever at large n: Rommel large-n 1-2% relative;
  MEGConformer final null; Défossez, d'Ascoli and MEG-XL use none.
- Run these as cheap instruments between bigger data/architecture runs, not as the main bet.

---

## 5. Time-frequency inputs vs raw time domain

**Who uses what.**
- **Raw time domain (the frontier):**
  - Défossez 2023: 120 Hz, 3 s; mel spectrograms appear only as speech *targets* [FT].
  - d'Ascoli 2024/25: 0.1-40 Hz, 50 Hz [FT].
  - MEG-XL [FT].
  - PNPL 2026 baselines: raw 306×T [FT].
  - MEGConformer: raw 306 ch at 250 Hz [FT].
  - MEBM-Phoneme: 306×125 [FT].
  - No top LibriBrain 2025 entry that I read used spectrogram input.
- **Spectrogram input on LibriBrain:** arXiv 2604.04129 tried an STFT CNN (Nfft 25, hop 5). Its best test score
  was 25.90% (with InstanceNorm) vs 60.95% for the time-domain CNN with InstanceNorm. Without InstanceNorm it
  scored 15.91% [FT]. "The STFT CNN variants are also consistently weak" [FT].
- **Band contribution on LibriBrain** (MEGConformer, band-limited inputs [FT]):
  - Phoneme task: unfiltered 43.69%; Delta alone 28.64% (significant); Beta, Gamma and HGA not significant.
  - Speech detection: Delta 75.86%; HGA 47.29%, at or below chance.
  - The best single bands (Alpha, Delta) retain about 87% of the unfiltered score, and Gamma/HGA contribute
    negligibly.
  - In short, the signal is low-frequency and phase-locked.
- **LibriBrain100 descriptive analysis** (speech vs non-speech, not decoding [FT]): the robust signatures are
  evoked amplitude over temporal sensors, **low-frequency inter-trial phase coherence**, and delta-theta power.
- **EEG BCI literature:**
  - Schirrmeister et al. 2017 (https://arxiv.org/abs/1703.05051) [AB]: deep ConvNets on raw EEG match FBCSP
    (band-power features) on motor imagery and learn spectral-power features internally.
  - Rommel's tasks are spectral by nature, which is why phase-destroying augmentations survive there [FT].
  - Dash et al. 2020 (MEG, 5 imagined/spoken phrases) [AB, snippet only, URL not verified]: a CNN on wavelet
    scalograms beat an ANN on statistical features, 96% vs 93%. That compares time-frequency against
    hand-crafted statistics, not against a raw-time-domain CNN.
  - So it is no evidence against raw input.
- **Our closed instrument 7gf.80** (`.beads/issues.jsonl`) — features and classifier:
  - high-gamma log-envelope in the bands 30-45, 55-95 and 105-120 Hz, taken from the 250 Hz h5;
  - band-covariance whitening to the top 80 components, then Butterworth (order 4), |Hilbert|, log, and a per-run
    z-score;
  - 8 bins of 0.1 s over 0-0.8 s, giving 1920 dims, classified by shrinkage LDA.
- **7gf.80 — results** (n=2512):
  - High band alone scored 0.2315 macro (chance 0.20), vs 0.3247 for the matched <25 Hz arm.
  - Concatenating the two cost 2.1 pt; fusing with the context head cost 1.1 pt.
  - Conclusion: no usable word-locked information above 25 Hz. This agrees with MEGConformer's Gamma/HGA null.

**Argument (arg), not a finding.**
- At 50 Hz the CNN's first temporal convolution is already a learnable FIR filterbank over 0-25 Hz.
- An STFT or complex-wavelet front end supplies a *fixed* filterbank (complex Morlet = amplitude + phase) that
  the network could learn anyway. Its only potential addition is the |·| nonlinearity, which the network can
  also compute.
- Envelope or band-power features discard exactly the phase-locked component that carries the evoked word
  response.
- Expected gain from switching the input representation: about 0 pt, with documented regressions on this very
  dataset (2604.04129).

**What 7gf.80 does NOT rule out.** *Induced* (non-phase-locked) power changes below 25 Hz, such as theta/alpha/beta
envelopes, as an auxiliary channel next to the raw input. 7gf.80 tested only above 25 Hz. A cheap way to test this
is the same LDA instrument on 4-8, 8-13 and 13-25 Hz log-envelopes, about 15 min. Its odds of a ≥1.5 pt fused gain
are low (<10%), because a deep CNN with ReLUs can already represent band energy. It is worth running only as an
instrument that closes the time-frequency axis entirely.
