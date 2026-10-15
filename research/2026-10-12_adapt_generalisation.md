# Few-shot per-subject adaptation that generalises: levers for the Broad adapt (2026-10-12)

**Question.** Our adapt fine-tune (stem → subject 0 + one ses-12 session each for subjects 1-32, encoder at 0.1x lr,
40 epochs) memorises. At ep80 local test drops .055 while subject-0 val holds. A frozen encoder costs .04-.06. Which
published methods make a few-shot per-subject adapt generalise, and which of them do we not have yet?

**Scope.**
- Web reading only, about 15 sources. Numbers are as each source reports them; no compute was run.
- This is a delta on the 10-04, 10-08, 10-09 and 10-11 docs; what they cover is not repeated.
- An "adapt run" is one ep40 adapt from an existing stem. I did not measure its wall-clock.

**Already built (not re-proposed):**
- L2-SP pull toward the stem (`megspeech/training/regularise/weights/pull.py`, 7gf.147.13, in flight).
- Subject-matrix anchor (`regularise/anchor.py`, worse, 7gf.35.2).
- Eval shrink toward subject 0's matrix (`SubjectLayer.shrink`, +.019 LB, 7gf.35.3).
- Mean / source-mean unseen fallback (`model/input/subjects.py` `Unseen`, 7gf.147.1).
- One-scalar `encoder_lr` / freeze (`training/transfer.py`, `model/motion.py`).
- EMA (`optim.ema`) and seed ensembles (`model/ensemble.py`, 7gf.20).
- BGMix (`regularise/bgmix.py`), regional dropout (`regional.py`) and same-word averaging (`average.py`).
- EA (`data/cache/alignment.py`, +.038, 7gf.56). AdaBN measured −.064.
- Adapt length dose (7gf.147.12): ep20 scored −.009 public.

**The ruler problem comes before every lever (13i).**
- Broad local test is ses-11 of the same book, reader and domain as the ses-12 adapt set. Public uses different
  stimuli.
- `encoder_lr` 1.0 scored +.02 local but −.008 public.
- So any adapt regulariser chosen on local test is biased toward exactly the specialisation we are trying to stop.
- Every row below is therefore priced against **public**, or against the unseen-stimulus instrument (the open "Broad
  gap: unseen-stimulus test for adapt-stage changes" issue), never against local test alone.

## Ranked levers

| # | Lever | Mechanism | Evidence (number, setting) | Have it? | Plausible on Broad public | Cost |
|---|---|---|---|---|---|---|
| 1 | **WiSE-FT: interpolate weights between stem and adapt** | Fine-tuning moves the features of the fine-tune distribution and leaves others behind. A linear interpolation θ = (1−α)·stem + α·adapt keeps most of the in-domain gain and recovers the stem's off-domain behaviour. Eval-only: sweep α ∈ {.25, .5, .75}. Only parameters present in both models are interpolated; the subject matrices are the shrink's job. | [Wortsman et al. 2022](https://arxiv.org/abs/2109.01903), CLIP on ImageNet: **+4 to 6 pp OOD** over prior robust fine-tuning, +1.6 pp ID, and +0.8 to 3.3 pp ID over standard fine-tuning on 7 transfer sets. | **No.** EMA averages over time, not stem↔adapt. Pull is the train-time cousin, and WiSE-FT composes with it. | 0 to +1.5. It attacks exactly the stimulus-specialisation axis of 13i. | ~1 h of code (state-dict lerp + eval). 0 GPU-h training. 1-2 public submissions to read α, since local cannot. |
| 2 | **Shrink per subject by data volume (empirical-Bayes keep)** | The shrink uses one `keep` for every subject. A James-Stein-style keep_s = n_s/(n_s+k) gives data-poor subjects less of their own matrix. | Ours (7gf.147.1): the mean matrix beats each subject's own matrix by **+.046 for subjects 23-32 (~300 words)** and +.008 for 13-22, but scores −.009 for 1-12 (~1300 words). The 7gf.35.3 shrink transferred to LB at ~0.95. | **Partly.** Shrink and mean fallback exist; keep is not per subject. | +0.5 to +1.5. Most of the +.046 is on a third of the subjects. Matrices are less stimulus-specific than the encoder, so local test is a fairer ruler here. | ~1 h (k from one CPU sweep on the ruler). 0 training. |
| 3 | **Surgical / partial tuning: subject layer + first encoder block (or BitFit: biases + norms) at full lr, rest frozen** | Subject shift is an *input-level* shift. Tuning only the first block fits it without letting the deep layers memorise trial or word content. This is the middle ground between full encoder at 0.1x (memorises) and frozen (−.04 to .06). | [Lee et al. ICLR 2023](https://arxiv.org/abs/2210.11466): first-block-only tuning **82.8 vs 79.9** full FT on CIFAR-C (input shift). [BitFit, Ben Zaken et al. 2022](https://arxiv.org/abs/2106.10199): "competitive with (and sometimes better than)" full FT at small-to-medium data. [d'Ascoli et al. 2024/25](https://arxiv.org/abs/2412.17829): accuracy rises with log data *per subject*, so a 1-session subject supports few parameters. | **No.** `encoder_lr` is one scalar, and `fused_lr` is per encoder, not per block. | 0 to +1.5. Our frozen result shows the encoder must move; this tests *where* it must move. | ~2-3 h code (per-block lr groups in `transfer.py`) + 2 adapt runs (first-block; biases+norms). |
| 4 | **LP-FT: fit subject matrices and heads with the encoder frozen, then unfreeze (with pull)** | When a randomly or poorly initialised head trains jointly with the encoder, its early large gradients distort the pretrained features. Fitting the new linear parts first leaves only small corrections for the encoder. | [Kumar et al. ICLR 2022](https://arxiv.org/abs/2202.10054): across 10 shift datasets, LP-FT is **+10% OOD, +1% ID** over full FT. | **Buildable from flags.** Run 1 with `encoder_lr 0` (exists, measured as an end state only), then run 2 `warm` from run 1 with `encoder_lr .1` + `pull`. Never run as a 2-stage recipe. | 0 to +1. Size depends on how far the adapt subjects' initial matrices are from fitted. Unknown. | 2 chained adapt runs, no code. |
| 5 | **Subject-specific low-rank corrections inside shared layers (SuLoRA)** | W = W_shared + A_s·B_s on conv/linear layers, a per-subject rank-r delta. It gives subjects more than a sensor matrix while capping their capacity to memorise. | [Klein et al. 2025, arXiv 2510.08059](https://arxiv.org/html/2510.08059): **MEG speech perception top-1 41.94 vs 40.60** with half the parameters. BCI-IV-2a 64.72 vs 55.00 subject-agnostic and 59.36 per-subject models. Code is **MIT**: [timonkl/SubjectConditionedLayer](https://github.com/timonkl/SubjectConditionedLayer). [Biderman et al. 2024](https://arxiv.org/abs/2405.09673): LoRA "learns less, forgets less". | **No.** LoRA was removed as dead code with the MEG-XL e2e path. The subject layer is the sensor-space-only version. | 0 to +1. Its MEG gain is +1.3 top-1 on a different task, and its zero-shot value for subjects 33-39 is ~0. Supersedes the 10-11 #4 price with an MEG number. | ~0.5 day code (MIT drop-in, read for the recipe) + 1 stem/adapt run. |
| 6 | **Same-word cross-subject mixup in subject space** | Mix two subjects' windows of the *same word* after the subject layer, label unchanged. The mixed sample spans the space between subjects, so the encoder cannot key on subject identity. | [Inter-subject category-specific augmentation, TNSRE 2023](https://doi.org/10.1109/tnsre.2023.3300961) [abstract]. [mixEEG 2025](https://arxiv.org/abs/2504.07987) [abstract]. EEG emotion / motor imagery; no MEG word number found. | **Near.** BGMix swaps template and background within a key; this crosses subjects in the latent space instead. | 0 to +0.5. | ~3 h code + 1 run. |
| 7 | Seed soup of adapts from one stem | Adapts that start from one stem are likely linearly connected, so a uniform soup ≈ the ensemble at 1x inference. | [Wortsman et al. ICML 2022](https://arxiv.org/abs/2203.05482): greedy soup 81.03 vs best single 80.38; uniform soup 79.97 (lower). | Ensembles yes (7gf.20). Soup of adapts no. | ~0 over our ensemble. It only matters if inference time caps the member count. | 1 h. |


**Not ranked, each ~0 for us:**
- **Per-subject early stopping.** The only per-subject labels are the adapt session itself, which is the same
  stimulus, so selecting on them would fit the stimulus (13i).
- **Layer-wise lr decay** ([ULMFiT](https://arxiv.org/abs/1801.06146)). It lowers early-layer lr, the opposite of #3
  for an input-level shift.
- **POYO "unit identification"** ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/8ca113d122584f12a6727341aaf58887-Paper-Conference.pdf)).
  It freezes the model and learns only the new session's embeddings. That is our frozen-encoder arm, measured
  −.04 to .06.

**Blocked by our constraints:**
- TTA ([2509.24700](https://arxiv.org/abs/2509.24700)).
- Picking a zero-shot subject's matrix by its test-run covariance. That fits on test MEG beyond whitening; it is an
  owner call.
- Riemannian Procrustes stretch/rotate ([Rodrigues 2019](https://arxiv.org/abs/1806.03573)), which needs target
  labels. Plain re-centring is legal, and we already have it as EA.

## Reading

- **Run #1 and #2 first.** They cost no training and target the two failures we measured: the encoder specialising
  to the stimulus (13i) and the data-poor subject matrices (7gf.147.1). #1 can only be read on public: spend at most
  two submissions.
- **#3 is the structural question: where does the encoder need to move?** 0.1x on every layer memorises, and frozen
  loses .04-.06. #4 needs no code and composes with #3 and with pull.
- **#5 is the only lever with an MEG-speech number.** That number (+1.3 top-1) is below our bar, so treat it as
  directional.
- **No lever here helps subjects 33-39 (zero-shot)** beyond the mean fallback we already have.

## Not yet built

1. Stem↔adapt weight interpolation at eval (WiSE-FT, α sweep).
2. Per-subject shrink keep from word count, keep_s = n_s/(n_s+k).
3. Per-block encoder learning rates / surgical freeze (first block, or biases+norms only).
4. Two-stage LP-FT adapt recipe (flags exist; the recipe was never run).
5. Subject-specific low-rank deltas in shared encoder layers (SuLoRA).
6. Same-word cross-subject latent mixup.
7. Soup of adapt seeds (low value).
8. An unseen-stimulus ruler for adapt-stage changes. It is the instrument every row above needs (13i, issue open).

## Sources

All are linked inline. Two more were read for context:
- [Défossez 2023](https://www.nature.com/articles/s42256-023-00714-5): removing the subject layer costs −17 pt top-10.
- [Csaky 2023](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26500): MEG subject embeddings; group models beat
  per-subject models on low-accuracy subjects. Numbers were not extracted (captcha).
