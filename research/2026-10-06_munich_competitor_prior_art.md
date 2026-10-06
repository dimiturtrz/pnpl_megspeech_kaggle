# Prior art a Munich-area leader could build on: MEG word decoding (2026-10-06)

**Question.** The Deep public LB leader ("Munich Logic Lab") scores 0.765 BAcc@10 against our 0.639. What
published methods could produce a lead that size? Which of them have we not yet run?

**Scope.**
- Public sources only, found by web search on 2026-10-06.
- Covers groups and methods only. No team members are identified or profiled.
- Mechanisms are read from abstracts and paper text. Numbers are as each paper reports them, not re-measured by us.
- The "our status" column was audited read-only against:
  - `.beads/issues.jsonl`
  - `interpretations/deep/*.md` and `interpretations/broad/*.md`
  - `research/2026-10-04_meg_augmentation.md`
  - `docs/PLAN.md`

**Headline: no Munich group has published MEG word or speech decoding.** Searches covered TUM, LMU, Helmholtz
Munich, MCML and MPI, and found no such paper. MEGFormer had been attributed to Munich by one search snippet; the
paper itself is from Skoltech, Sharjah, AIRI and MSU. The leader's method is therefore unknown. The sections below
cover what any strong team could take from the open literature.

## 1. Competition paper: PNPL 2026, arXiv 2609.03231

<https://arxiv.org/abs/2609.03231>. Organised by Oxford PNPL with Maastricht, Montréal/Mila and Google DeepMind.

**Data.**
- Subject 0 has ~80 h: the Sherlock canon, TIMIT, MOCHA-TIMIT and 30 Moth stories.
- Subjects 1-32 have 40, 20 or 10 min each (12, 10 and 10 subjects). Subjects 33-40 are zero-shot.
- Vocabulary: 50 high-frequency words, plus Moses-50 as the secondary vocabulary.

**Baselines.**

| Track | Baseline | BAcc@10 | Other |
|---|---|---|---|
| Deep | d'Ascoli 2025 | 73.23 | BAcc@1 25.60; Moses 83.60 |
| Broad | MEG-XL | 42.96 | |

**Explicit hint:** "Joint training on Subject 0 and Subjects 1–32 will improve generalisation on the latter
subjects." External data and pretrained models are allowed.

## 2. LibriBrain100 dataset paper: arXiv 2608.25204

Setup: MEG-XL fine-tuned on a 0.2-0.6 s word window.

| Training data | Subject 0 (Sherlock test) | Subjects 1-32 |
|---|---|---|
| With subject 0 and subjects 1-32 | ~0.65 | ~0.50 |
| Without the other group | marginally better than with | ~0.35 |

- Subject 0 sits at ~0.65 BAcc@10. Multi-subject training helped subject 0 on TIMIT and MOCHA, but not on
  Sherlock.
- For subjects 1-32, adding subject 0 is worth about +15 pt.
- Keeping only 25% of the data gave no significant drop, so cross-subject transfer carries the Broad score.

## 3. MEG-XL: arXiv 2602.02494 (ICML 2026, Oxford)

Code: <https://github.com/neural-processing-lab/MEG-XL>

**Mechanism.**
- Tokenizer: BioCodec RVQ, 6 levels, vocabulary 256, per channel, ×12 compression.
- Encoder: 8-layer, 512-wide criss-cross (time × sensor) transformer with sensor Fourier embeddings.
- Pretraining: masked-token prediction (3 s blocks, 40% masked) over a 150 s context, on ~300 h from 800
  subjects (MOUS, Cam-CAN, SMN4Lang).
- Fine-tune: 50 word-aligned 3 s windows, pooled over the word interval, then an MLP to T5 with D-SigLIP.

**Numbers.**
- At 13% of the data: 57.3% vs ~20% for d'Ascoli.
- At 100%: 63.0% vs ~65%.
- Gains from context diminish beyond ~100 s.
- Its edge is data efficiency. That matters for Broad, not for an 80 h Deep model.

## 4. d'Ascoli et al. 2025: arXiv 2412.17829, Nat Commun (Meta)

Code: <https://github.com/neural-processing-lab/dascoli-word-decoding>

**Mechanism.**
- Preprocessing: 0.1-40 Hz, 50 Hz, RobustScaler, clamp ±5, baseline 0-0.5 s.
- Window: 3 s (−0.5 to +2.5 s around the onset).
- Word encoder: spatial attention, subject layer, conv, attention pooling.
- Context: a 16-layer, 16-head, 1024-wide rotary transformer across the sentence.
- Target and loss: T5-large middle layer with D-SigLIP (top-10 18.2 vs 15.7 for CLIP, 250 words).
- Recipe: AdamW 1e-4, batch 64, cosine over 50 epochs.

**Numbers.** Averaging 8 repetitions of the same word roughly doubles accuracy, reaching ~80% top-10. This is the
largest single effect in the literature.

## 5. Brain2Qwerty v2: arXiv 2608.18114 (Meta)

Task: typing, 9 subjects × 10 h, decoded as sequences.

**Mechanism.**
- Preprocessing: 0.5-45 Hz, 100 Hz, RobustScaler.
- Augmentation: ±400 ms crop jitter, drift noise, time and channel masking, time-stretch 0.8-1.2.
- Encoder: Fourier-feature channel merge, dilated conv, 4-layer Conformer, CTC plus an auxiliary head.
- Word aligner: SigLIP with hard DTW.
- LLM stage: Qwen3-4B with LoRA r128. Per-subject adapters are combined by **model soup**. Uses 10% modality
  dropout, label smoothing 0.02 and beam 16.

**Numbers.**
- WER: 0.39 with the LLM, 0.55 encoder-only, 0.43 with an n-gram LM.
- CER improves log-linearly with hours and has not saturated.
- An automated research loop found +16-20% relative.

## 6. MEGConformer: arXiv 2512.01443 (LibriBrain 2025 phoneme winner, F1 65.83)

**Mechanism and ablations.**

| Mechanism | Effect |
|---|---|
| Conformer (7 layers, 12 heads, kernel 127), 0.5 s phoneme window | Base architecture |
| Speech detection: 2.5 s window vs 0.5 s | +10.8 |
| Per-window instance norm vs batch/layer norm | >200% relative on holdout |
| Dynamic grouping (average 100 same-label samples into one training example) | 38.39 → 44.29 |
| Inverse-sqrt class weights | 40.92 → 44.29 (+3.4) |
| Stride 60 | +2.4 |
| MEGAugment | Null |
| 5-seed majority vote | 55.09 → 65.82 on holdout |

Recipe: AdamW 1e-4, wd 0.05, batch 256.

## 7. MEGState: arXiv 2512.17978 (Academia Sinica / Keio, 2025 phoneme track)

**Mechanism.**
- Multi-resolution convolutions (kernels fs/2 … fs/16) plus a sensor-wise S5 SSM.
- **Class-prototype averaging** (100 samples), with mixup at α=0.5 between two classes' prototypes.

**Numbers.**
- Test F1: 41.11 vs 34.82 for the baseline.
- LB: 55.74 single model, 68.41 for a 5-model ensemble.

## 8. Other 2025-2026 papers

**MEG-to-MEG transfer (Basque Center / EHU), arXiv 2602.18253.**
- MEGConformer pretrained on 50 h of LibriBrain, then fine-tuned on 5 min × 18 subjects.
- Z-scores per subject and task, per sensor and time point. **Uses test-set statistics on the test set**
  (transductive).
- RollAugment; soft targets.
- Gains: +1-4% in-task, +3-6% cross-task.

**Bypassing reconstruction (Xiao, Wang, Cheng), arXiv 2605.13099.** 1st place on the 2025 extended speech-detection
track, F1 0.962.
- Mechanism: contrastive retrieval of the MEG against a LibriVox audio database. Detection then runs on the
  retrieved audio.
- In effect the stimulus is identified from a public corpus.
- **Rules flag:** external data is legal, but the final test uses "entirely different" stimuli. Retrieving a
  public source text would be an oracle, not a decoder.

**Robustness under distribution shift (Taiwan), arXiv 2604.04129.**
- Instance norm: +13.48 pt.
- An augmented CNN collapsed to 3.60 on holdout.
- Conclusion: preprocessing matters more than architecture.

**MEGFormer (MICCAI 2024; Skoltech/Sharjah/AIRI/MSU, not Munich).** Code: <https://github.com/maryjis/MEGformer>
- Mechanism: Défossez CNN plus transformer, CLIP to wav2vec2-xlsr-53, 3 s at 120 Hz, clamp 20 SD, subject layer,
  spatial attention to 270 channels.

| Dataset | Configuration | Top-10 |
|---|---|---|
| Gwilliams | Base | 69.66 |
| Gwilliams | Depth 6 + dropout 0.2 | 77.78 |
| Gwilliams | + sound-aligned non-overlapping segments (+50% segments) | 83.99 |
| Schoffelen | Base → best | 67.89 → 70.49 |

- Accuracy rises monotonically with segment length across 3, 4 and 7 s.

**Shorter notes.**
- **Keyword spotting (Oxford), arXiv 2510.21038:** focal loss and top-k pooling. Log-linear data scaling.
- **LM integration (Maghsoudi, Shamma, Ganapathy), arXiv 2609.31997:**
  - Mechanism: contrastive acoustic plus linguistic targets, decoded with a neurally constrained LM beam.
  - Finding: the LM helps imagined speech far more than listened speech.
- **Interpretable decoding (Ossadtchi group), arXiv 2608.01481:**
  - Mechanism: a spherical-harmonics spatial layer.
  - Finding: retrieval relies on silence, intensity, vowels and onsets, i.e. acoustics, not semantics.
- **MEG foundation-model roadmap, arXiv 2609.04461:**
  - Within-subject scaling beats between-subject scaling.
  - Fixed tokenizers perform about as well as learned ones.
- **Not relevant to classification:** MindVoice (Fudan, arXiv 2605.31173, reconstruction) and the Csaky thesis
  (Oxford, arXiv 2510.24733, mostly inner-speech negatives).

## 9. Munich neuroengineering landscape

Munich trains and funds neuroengineering, so a strong student or lab team from there is plausible. None of its
groups has a public MEG word-decoding method. The main programmes:

- **TUM Elite MSc Neuroengineering (MSNE).** Run by the Dept. of ECE, with faculty from LMU, MPI Neurobiology /
  Biological Intelligence and Helmholtz Munich. Currently not admitting.
  - <https://www.tum.de/en/studies/degree-programs/detail/neuroengineering-master-of-science-msc>
  - <https://www.elitenetzwerk.bayern.de/en/home/funding-programs/elite-graduate-programs/overview-of-elite-graduate-programs/neuroengineering>
- **TUM CoC Neuro-Engineering:** <https://www.ei.tum.de/en/research/coc-neuro-engineering/>
- **TUM ICS neuroengineering** (BCI and exoskeletons): <https://www.ce.cit.tum.de/en/ics/research/areas/neuroengineering/>
- **Others:** LMU's Graduate School of Systemic Neurosciences, BCCN Munich and MCML (ML foundations).

The team name ("Logic Lab") hints at a symbolic or reasoning angle, such as LM or constraint decoding. That is
speculation, not evidence.

## 10. Ranked: mechanisms that could move word BAcc@10, with our status

**How "plausible size" was estimated.** Each value is the source's effect, discounted for transfer to our setup.
Our setup is 50 words, ~307k labelled sub-0 words, a seed band of ~±0.012 and a 1.5 pt bar. Data scaling is
known to give ~+3 pt per doubling (7gf.104).

**Evidence grades.**
- **A:** replicated, or a clean ablation.
- **B:** one paper's ablation.
- **C:** indirect.

**Ordering.** Rows 1-9 are mechanisms we have not run, or have not run validly. Rows 10 onward are ones we tried.
**New** marks a mechanism that is not in our current model's list.

| # | Mechanism (source) | Plausible size on our Deep | Evidence | Our status |
|---|---|---|---|---|
| 1 | **Test-time averaging over repeats of the same stimulus.** Holdout sentences are identical across all 40 subjects, so pool the posteriors across subjects (d'Ascoli 8-rep ×2). **New** | +3 to +8 pt; the only lever priced at ~+0.07 (7gf.65). Rules check needed: it reads other subjects' Kaggle test MEG. | A (d'Ascoli) | Audited, never run (7gf.15 in_progress; algebra done) |
| 2 | **Training-time dynamic grouping / class-prototype averaging** of same-word trials (MEGConformer 38.4→44.3; MEGState). **New** | +0 to +3 pt. Raises SNR per example; our 50-word set has many repeats per class. | B, phoneme task | Untried. Same-word averaging was skipped for training (aug doc §shortlist). BGMix 7gf.112 is the nearest relative (template + residual), not the same thing. |
| 3 | **Prototype mixup** between classes (MEGState α=0.5). **New** | +0 to +1.5 pt | B, single ablation | Untried (aug doc shortlist #3) |
| 4 | **Causal-LLM span beam / shallow fusion** over the 50 slots (Brain2Qwerty 0.43→0.39 WER over n-gram; 2609.31997) | 0 to +2 pt. Our classes are function words with OOV neighbours, so priced <10% odds of +1.5 (lvj audit). | B (sequence tasks), C for us | In progress (megspeech-lvj open; untracked `evaluation/studies/lm*.py`). BERT MLM fusion measured harmful (7gf.60.1). Pretrained-LM prior was gated out (7gf.6). |
| 5 | **MEG-XL fine-tune** (2602.02494; LibriBrain100 ~0.65 on sub-0) as a diverse ensemble member | 0 to +1.5 pt as a member. Single model ≈ ours. | B | Invalidly run: LoRA diverged at ep1, recipe failure (7gf.5.3); frozen reads 0.36 (7gf.5.x) |
| 6 | **Longer / pre-onset raw window** (3 s; MEGFormer monotone 3→7 s; MEGConformer 2.5 s detection) | 0 to +1.5 pt. The span transformer already sees neighbours. | B | Partly tried: 2 s null (lb_gap doc), lead 0.1 s null (7gf.38). 3 s untried (7gf.115 open). |
| 7 | **Time-stretch 0.8-1.2 / large crop jitter** (Brain2Qwerty). **New** | ≤ +1 pt; every MEG-input aug we ran was null or negative. | C | Stretch untried. Jitter ±40 ms −5.4 pt (7gf.18). |
| 8 | **Focal loss / top-k pooling** (2510.21038). **New** | ≤ +1 pt | C | Untried. Attention pool tried: 0.615 (7gf.45). |
| 9 | **Per-subject LLM-adapter model soup** (Brain2Qwerty) | ~0 on Deep, as one subject | C | Not applicable to Deep. Seed weight soup built (7gf.20). |
| 10 | Joint sub-0 + subjects 1-32 training (paper hint; LibriBrain100 +15 pt on **Broad**) | Deep ~0; Broad large | A for Broad | Deep tried: 0.602/0.609, null (7gf.15.1). Broad: 33-subject models exist (4te.8 broad-nat); Broad epic megspeech-dcn open. |
| 11 | Per-window instance norm (MEGConformer; 2604.04129) | — | A on holdout shift | Tried: local null 0.6154 (7gf.25.14); ensemble LB 0.613 < 0.637 (4te.10) |
| 12 | Inverse-sqrt class weights (MEGConformer +3.4) | — | B | Tried: null 0.603 (7gf.23); qr2 balanced arms open |
| 13 | SigLIP/D-SigLIP to T5; CLIP to wav2vec2 (d'Ascoli, MEGFormer, MEG-XL) | — | A in the field | Tried: null 0.613 (7gf.37), 0.607 (7gf.25.3), wav2vec2 null (7gf.54). Autopsy open (megspeech-99y). |
| 14 | Spatial attention + subject layer (Défossez/d'Ascoli/MEGFormer) | — | A | Tried: 0.6472 vs 0.6453 (7gf.55) |
| 15 | Conformer encoder (MEGConformer; Brain2Qwerty) | — | B | Tried: 0.594 (qi7.3), 0.595 sentence (4te.7) |
| 16 | Deep rotary 16L/1024 context transformer (d'Ascoli) | — | B | Tried: 0.6234 (7gf.51.1); faithful replica 0.6408 (7gf.51) |
| 17 | Time/channel masking (Brain2Qwerty) | — | C | Tried: amask5 val null, local −2.5 (megspeech-blq); cho null |
| 18 | Multi-resolution conv / SSM (MEGState) | — | B | Tried: k357/seg4 null (7gf.26); S4D 0.614 (7gf.27) |
| 19 | Transductive per-subject z-score with test statistics (2602.18253) | — | B | Tried: rescale no-op (7gf.14); AdaBN −0.064 (7gf.61.1); EA hurts Deep (7gf.61) |
| 20 | Seed ensembles / majority vote (MEGConformer, MEGState) | — | A | Done: geometric-mean ensembles; saturates past 16 seeds (selection_bias doc) |
| 21 | Masked SSL pretraining on own MEG (MEG-XL style) | — | B | Tried: 0.6377 vs 0.6361 (7gf.109.4 in_progress); external MASC null (7gf.57) |
| 22 | Stimulus retrieval from public corpora (2605.13099) | Possibly huge, but an oracle | B | Untried, **and should stay untried**: it violates the spirit of a decoding competition, and the final test uses different stimuli. |

## Reading

**Where a 12.6 pt lead could come from.** Only row 1 has a mechanism and a published effect of that size.

- **Row 1, cross-subject test-time pooling.** Averaging 8 repeats roughly doubles accuracy (d'Ascoli). The Kaggle
  holdout gives the same sentences to 40 subjects, so a team that pools across Broad-track subjects gets that
  effect for free.
  - It hinges on a rules reading: may a Deep prediction use the Kaggle test MEG of other subjects?
  - Ask the organisers, or read the Kaggle rules thread, before building it.
- **Row 2, dynamic grouping.** This is the one cheap untried training mechanism with a clean ablation behind it
  (+5.9 abs).

**What we can stop chasing.** Everything the field reports as its architecture or objective (rows 11-21), we have
already run, and each sat inside our seed band.

**Two things not to assume.**
- No source here explains our IOI local-vs-LB gap (7gf.105).
- No source reports a Deep-track number on the Kaggle holdout, so 0.765 has no published comparator.
