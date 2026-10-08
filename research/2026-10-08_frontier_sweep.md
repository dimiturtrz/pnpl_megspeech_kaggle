# Frontier sweep: published MEG word-decoding levers not yet covered (2026-10-08)

**Question.** What does the 2025-2026 frontier do that this repo has not tried? Which of those could plausibly
move public BAcc@10 by >= 1.5 pt absolute (Deep .668, Broad .601)?

**Scope.**
- Web search and reading only, 2026-10-08. About 25 sources opened. No compute was run.
- Numbers are as each paper reports them; we did not re-measure any.
- This is a **delta** on two earlier docs:
  - [2026-10-06_munich_competitor_prior_art.md](2026-10-06_munich_competitor_prior_art.md) covers PNPL 2026, LibriBrain100,
    MEG-XL, d'Ascoli 2025, Brain2Qwerty v2, MEGConformer, MEGState, 2602.18253, 2605.13099 and 2604.04129. Its §10
    ranks 22 levers with our status.
  - [2026-10-04_meg_augmentation.md](2026-10-04_meg_augmentation.md) covers augmentation.
  - [2026-10-07_target_encoders.md](2026-10-07_target_encoders.md) covers target encoders.
  - None of that is repeated here.
- "Our status" was checked read-only against `.beads/issues.jsonl`.

**Constraints that close levers (from beads, not from the papers):**
- **Test-time cross-subject pooling is dead.** The organiser announcement says cross-track mapping is leakage
  (Kaggle topic 741919, recorded in 7gf.15).
- **Owner decision 2026-10-08: no fitting on unlabelled test MEG.** This rules out TTA, test whitening and
  transductive adaptation, though the decision is kept open.
- **The LM prior and the GPT-2 targets still await explicit owner sign-off** (memory note in beads).

**Headline.**
- No 2026 competitor write-up or code is public. GitHub and arXiv searches for PNPL-2026 solutions return only the
  organisers' paper, and Kaggle discussion is not indexed.
- The new frontier material since 10-06 is thin. Most of it is either already in our tree or blocked by the
  constraints above.
- Four levers are new and legal:
  - selective pooling (#2)
  - intermediate-depth readout (#3)
  - OSI-licence specifics for the LM rescoring already in flight (#1)
  - SRM-style closed-form subject alignment for Broad (#5)
- Only #1 has a mechanism sized for the 10-13 pt gap to the leaders, and its odds stay low.

## Ranked levers

| # | Lever | Mechanism | Evidence (number, dataset) | Expected on our task | Cost | Licence | Our status |
|---|---|---|---|---|---|---|---|
| 1 | **Causal-LM span rescoring, recipe specifics** | Beam over the sentence's 50-word slots. Score = MEG logprob + λ·LM logprob, with **λ=1.5 and beam 5**. OOV positions are handled by **predictive in-filling**: an XGBoost OOV detector (AUROC 0.88), then the LLM fills the gaps, or the LLM is prompted with `<UNK>` / top-5 candidates. | Jayalath, Landau, Parker Jones, [arXiv 2505.13446](https://arxiv.org/abs/2505.13446). LibriBrain BLEU 0.21 → 0.25 with beam+fill. Uses Llama-3.2-1B. Shown on brain-to-text, not on top-10 classification. | 0 to +3 pt, both tracks equally. 7gf.114 infers that the leaders' gap is mostly subject-agnostic: their Deep → Broad drop is 3.6 pt against our 6.9. That is consistent with an LM-type prior but not proof of one. The 10-06 doc priced it at <10% odds of +1.5, because our classes are function words with OOV neighbours. In-filling is the missing piece for exactly that case. | Already coded (untracked `evaluation/studies/lm*.py`, megspeech-lvj). Adding in-filling costs ~0.5 day. CPU or small GPU. | **Llama 3.2 Community licence is not OSI.** Use Qwen3-0.6B/1.7B (Apache-2.0), SmolLM2 (Apache-2.0), Pythia (Apache-2.0), OLMo-2 (Apache-2.0) or GPT-2 (MIT). | lvj open. Gated on owner sign-off. BERT MLM fusion measured harmful (7gf.60.1). |
| 2 | **Selective pooling: curate the pretraining mix by standalone decodability** | Pool only the datasets that decode well on their own. Jointly training with a high-quality dataset lifts the weak ones. Pooling everything indiscriminately does not. | [2505.13446](https://arxiv.org/abs/2505.13446): standalone decodability correlates with pooling gain at **r = .95** (p = .048, n = 4 datasets). Gwilliams ~6.3% → ~12.7% when pooled with LibriBrain + Armeni. The roadmap [2609.04461](https://arxiv.org/abs/2609.04461) restates this as "composition matters more than raw scale". | **Deep ~0.** Subject 0 is the strong set; weak sets cannot lift it. **Broad 0 to +2 pt:** dropping low-decodability sources (Kymata? Appleseed? MEG-SCANS, just added in u2m.3) could stop them diluting the pooled trunk. The n = 4 correlation is weak evidence. | One leave-one-source-out sweep over the pooled sources, about 1 run each, scored on the Broad ruler. | n/a | MASC on Broad tried: single +.027, absorbed by the ensemble (7gf.57). No per-source ablation in beads. |
| 3 | **Intermediate-depth ("pyramid") readout with a bounded residual on a frozen contextual decoder** | Freeze a CNN + sentence-transformer word decoder, the same family as ours. Read a temporal pyramid from intermediate GLU stages 2 and 4, fuse it with semantic queries, and add one residual correction to the logits, norm-bounded at < 2c with c ∈ {0.15, 0.25} chosen on val. Not iterative at test time. | HDND, [arXiv 2609.24095](https://arxiv.org/abs/2609.24095), 250 words, balanced top-10. Armeni MEG 47.19%, Gwilliams 18.01%. The pyramid is worth ~2-3 pp, and the shuffled-pyramid control loses 1-3 pp. Beats the contextual baseline in 9/9 conditions. | +0.5 to +2 pt. Their baseline is our architecture class. The bounded residual on a frozen base is cheap and cannot do much harm. | Fits the 6ow peer protocol ("frozen base + residual-boosted peers") directly: one head plus cached intermediate features, hours not days. | No code released. | **New.** No intermediate-layer readout is in beads. |
| 4 | **Smooth top-k loss, aligned to BAcc@10** | Train on a surrogate of top-10 recall per class (Berrada 2018) rather than top-1 CE. | No new MEG evidence. The metric definition (per-class top-10, averaged) is confirmed in [2505.13446](https://arxiv.org/html/2505.13446) and [2609.03231](https://arxiv.org/abs/2609.03231). | 0 to +1.5 pt. | 1 run. | n/a | 7gf.73 open. |
| 5 | **Closed-form shared-space subject alignment (SRM / per-subject affine into a shared latent) for Broad** | Stimulus-locked alignment: fit each subject's linear map into a shared space from responses to the same stimulus, then apply the pretrained decoder unchanged. Intracortical BCI uses the same per-subject affine idea and trains only the affine for a new subject. | ECoG: [2607.19394](https://arxiv.org/abs/2607.19394) narrows the source → held-out-subject drop (no MEG numbers). Intracortical: [bioRxiv 2026.02.27.708564](https://www.biorxiv.org/content/10.64898/2026.02.27.708564v1.full), PER 19.7 → 16.1 from joint training, matching within-subject. | Broad 0 to +2 pt, and only if subjects 1-32 heard stimuli that subject 0 also heard. Verify that first. Our learned subject layer is the gradient-fit version of the same thing. Closed-form SRM pays off at 10-minute subjects, where backprop overfits the map. | ~1 day. CPU closed form plus a fine-tune. | n/a | Idea noted (ct5 / 7gf.15 notes "same stimulus … hyperalignment"). Not run. |
| 6 | Sentence-consistent evidence pooling (Group Context Bias); audit of structural shortcuts | (a) Pool evidence that is consistent across a sentence's windows. (b) **Audit warning:** signal-blind Gaussian noise reached 66.3% R@1 under variable-length windows, and fell to chance once windows were fixed and splits held stimulus identity apart. | [2605.24524](https://arxiv.org/abs/2605.24524): GCB lifts Gwilliams to 52% and MOUS to 29%. 95.7% of top-1 errors pick the wrong *sentence*. | (a) ~0: our span transformer already does this. (b) Instrument only. Our shuffled-MEG context reads .54 of the .63 val (7gf.114). Check which part comes from IOI / position metadata, and whether that part transfers to the final test's new stimuli. | (b) one eval-only ablation. | n/a | Span transformer and IOI (7gf.105) done. |
| 7 | Semantic bottleneck (sentence-level semantic embedding target) | Map the whole sentence's MEG through dilated conv, then a temporal transformer, then mean-pool to a sentence semantic embedding (ADA). No word alignment needed. | [2609.10296](https://arxiv.org/abs/2609.10296), LibriBrain Sherlock: BERTScore 0.830 vs BrainECHO 0.828. The d'Ascoli-style word classifier wins on lexical metrics (WER .871). | ~0 for word top-10. The evidence is a semantic metric only, and ADA is a proprietary API (not OSI). | — | ADA not OSI. | Sentence aux targets covered in the target-encoders doc. |
| 8 | MEBM-Phoneme: multi-scale dilated conv + 3 BrainMagic encoders + conv attention; session-aware trial averaging | — | [2603.02254](https://www.alphaxiv.org/abs/2603.02254), LibriBrain phoneme macro-F1 60.95 (local val). Removing BrainMagic costs −6.52; multi-scale −1.20; conv attention −1.35; weighted CE −0.98. | ~0. Every component maps to something we ran and found null: multi-scale 7gf.26, averaging (closed arm), class weights 7gf.23. | — | No code. | Covered. |
| 9 | Brain foundation models: BrainOmni (656 h MEG + 1,997 h EEG), Brain-OF, MEG-Mamba | SSL pretraining on pooled MEG/EEG. | MEG-XL ([2602.02494](https://arxiv.org/abs/2602.02494)) reports beating "brain foundation models" on word decoding. MEG-Mamba ([2610.00746](https://arxiv.org/abs/2610.00746), Oxford OHBA): Cam-CAN resting state, source-parcellated, evaluated generatively only, no decoding. | ~0. Our own SSL arms came out null (7gf.109.4, 7gf.57). | High. | MEG-Mamba is CC-BY-4.0. BrainOmni licence unverified. | Not pursued. Low value. |
| 10 | Source-free test-time adaptation (multi-scale + self-distillation + TTA) | — | [2509.24700](https://arxiv.org/abs/2509.24700) (sEEG, DU-IN word benchmark, SOTA claimed). | **Blocked** by the owner's no-test-fitting decision. Our transductive normalisation arms hurt Deep anyway: AdaBN −0.064, EA (7gf.61). | — | — | Blocked. |

## Reading

- **What the leaders might be doing.** The literature offers no published mechanism of the size of the 10-13 pt
  gap other than the two already ruled out: test-time cross-subject averaging (dead) and stimulus retrieval (an
  oracle). The leaders lose little on unseen subjects (7gf.114, an inference). That points to a subject-agnostic
  prior, so #1 is still the only lever whose mechanism scales to the gap. Its prior is still low, because the scored
  words are high-frequency function words. In-filling over OOV neighbours is the untested part.
- **Cheapest new shot:** #3, the intermediate-depth bounded residual. Its baseline is our model class, it slots into
  6ow, and its reported effect (2-3 pp) is near the 1.5 pt bar.
- **Broad-specific:** #2 and #5 act on the data and alignment axis. Data and alignment usually outsize knob
  changes. Both need a cheap check first: per-source standalone scores (#2) and stimulus overlap with subject 0 (#5).
- **Licence watch:** Llama-family and ADA are out. The OSI LMs listed in #1 are drop-in replacements.

## Sources

- <https://arxiv.org/abs/2505.13446> · <https://arxiv.org/html/2505.13446>
- <https://arxiv.org/abs/2609.24095>
- <https://arxiv.org/abs/2605.24524>
- <https://arxiv.org/abs/2609.10296>
- <https://arxiv.org/abs/2609.04461>
- <https://arxiv.org/abs/2610.00746>
- <https://www.alphaxiv.org/abs/2603.02254>
- <https://arxiv.org/abs/2607.19394>
- <https://www.biorxiv.org/content/10.64898/2026.02.27.708564v1.full>
- <https://arxiv.org/abs/2509.24700>
- <https://arxiv.org/abs/2608.01481> (MASC retrieval: accuracy rises monotonically with window, 14% at 1.5 s to 62% at 5 s. Already priced as 7gf.115.)
- <https://arxiv.org/abs/2609.27650> (survey; no new methods)
