# Levers that raise the MEG-derived share of the Deep score (2026-10-12)

**Question.** In our best Deep model, stimulus timing (word inter-onset intervals, via `OnsetEncoding`) carries
about .31 bacc10 above chance, and the MEG encoder about .10. Local test: intact .694, MEG-blind .590, timing zeroed
.338. Public: intact .671, timing zeroed .306. See `interpretations/deep/2026-10-12_representation_audit.md` §5.
Which published or frontier methods (2023-2026) raised the MEG share in a similar setting, and which of them fit our
rules?

**Scope.**
- Web reading plus our docs and beads. No compute was run.
- This is a delta on `2026-10-06_munich_competitor_prior_art.md`, `2026-10-08_frontier_sweep.md`,
  `2026-10-09_external_meg_speech.md`, `2026-10-11_frontier_delta.md`, `2026-10-11_dascoli_port_diff.md` and
  `2026-10-12_adapt_generalisation.md`. What they cover is cited, not repeated.
- Numbers are as each source reports them. [unverified] marks a claim I did not read in the primary text.
- Written 22:45 Sofia, 12 Oct. The deadline is 15 Oct AoE, about 3 days away. Any lever over about 1 day of
  wall-clock is priced out.
- Wall-clock is in "Deep runs", one fine-tune on the RTX 5090. Its length is not measured here; the bead history
  suggests a few hours.

## 1. New primary source: the organisers' lab names our shortcut

[Jayalath & Parker Jones, "Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text", arXiv 2609.40359](https://arxiv.org/abs/2609.40359)
(Oxford PNPL, submitted 30 Sep 2026). Code: [neural-processing-lab/SimpleB2T](https://github.com/neural-processing-lab/SimpleB2T).
The code is MIT, except `simpleb2t/vendor/`, which is CC BY-NC 4.0.

**The finding (LibriBrain100 subject 0, balanced top-1 over 50 words):**
- d'Ascoli's joint sentence model scores **22.3%** on real MEG and **22.0%** on a synthetic signal with no brain
  information. A timing-only input scores 22.9%.
- Abstract: "Neighbouring windows partially overlap, implicitly revealing the interval between words... the method
  reaches 22.0% balanced accuracy on synthetic signals containing no brain information, compared with 22.3% on real
  brain recordings."
- With windows processed independently, real MEG scores 9.5% and synthetic 5.8%.
- The shortcut reproduces on Armeni and Le Petit Prince. It is absent on fixed-timing datasets.

**What this means for us:**
- Our MEG-blind .590 is the same effect, measured with top-10 on our own model. It is not a bug in our harness.
- The shortcut is **legal here**. The Kaggle test ships `word_onset_s` (`docs/competition/overview.md`), and the
  rules do not address stimulus timing.
- Cross-track use of `word_onset_s` / `shuffle_seed` is disqualifying (`docs/competition/announcements.md`).
  Within-track timing is not.

**What the paper does NOT give us:** a method that raises the MEG share *on a single 1 s window*.
- SimpleB2T: process each window independently; small CNN, temporal pool, residual MLP (about 20M parameters);
  3 s window at 50 Hz with baseline over the first 0.5 s; T5-large layer-12 targets; D-SigLIP loss.
- Its large gains come from two places:
  - aggregating k distinct responses to the same word: WER 65.6 at k=1 vs 36.6 at k=5, on a 92-word clinical
    benchmark;
  - Qwen3-8B rescoring.
- **Neither is open to us.** Test windows carry no label to group by. Grouping by MEG similarity fits on test MEG,
  which our own rule forbids (owner 2026-10-08, decision open; rules.md:35 leaves test-time adaptation "Not addressed").
- The 3 s window is impossible: the test releases `meg (B, 306, 250)`, i.e. 1 s (`2026-10-11_dascoli_port_diff.md`).

**Published MEG-only word level.** About 9.5% top-1 out of 50, isolated windows, 66 h of subject 0. No source
found reports a method that lifts isolated single-trial MEG word decoding by ≥1.5 pt at fixed data. The gains in the
literature come from (a) more same-subject data, (b) repeat averaging and (c) a language prior.

## 2. Einstein check: axes already null in-house

None of these is re-proposed:

| Axis | Bead | Result |
|---|---|---|
| Remove timing from the representation (adversary, span cut) | 7gf.98, 7gf.99 | .6382 vs .6445; GRL collapses or gives .6127 |
| Product of experts with a frozen timing model | 7gf.100 | 3-seed .6383, null |
| Contrastive MEG↔stimulus (CLIP-style) | 7gf.109.3 | +.006, noise; kept as a blend member |
| Audio / text teacher targets (wav2vec2, HuBERT, Whisper, SigLIP-t5) | 7gf.54, 7gf.141.6/.8, 7gf.37 | null or parity. Teacher-probe ceiling .20-.26 is below the MEG embedding's .307 (`2026-10-07_target_encoders.md`) |
| SSL pretrain | 7gf.109.4 | probe at random-init level |
| Foundation models (MEG-XL etc.) | — | null; NeuralBench "only marginally" (`2026-10-11_frontier_delta.md`) |
| Shift TTA / AdaBN / EA on Deep | 7gf.167, 7gf.61 | null / −.064 / hurts |
| Per-run centring / standardisation, per-subject levels | 9rp, c6e8717 | +.009 directional; levels built |
| Longer window | w15 memory | more MEG reliance, public −.042; 1 s at test anyway |
| Linear readout of MEG beyond timing | signal ceiling doc | +.006 bits; there is no linear lever |

## 3. Ranked levers (legal for Deep, not yet run)

Each lever is priced as wall-clock ÷ expected gain. Gains are bacc10 on public, absolute.

| # | Lever | Mechanism | Evidence | Data | Rules | 5090 wall-clock | Plausible gain | Odds ≥1.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Isolated-window MEG expert + late fusion** | Train our CNN encoder with **no span context and no onset input** (SimpleB2T recipe at 1 s). Then fuse its log-probs with the intact model, log-linear, weight picked on VAL. The joint model under-trains the encoder because timing explains the easy variance. A model that cannot see timing must fit MEG, and its errors decorrelate from the context model's. | 2609.40359: isolation makes repeat-aggregation and LLM priors "substantially" more effective. Single-model gain from isolation is reported only *against the shortcut* (9.5 isolated vs 22.3 joint), not as fused. 7gf.100 PoE used a frozen *timing* model, the opposite expert, so this is not the same axis. Our pool → mix step is +.149, so a pool-trained expert sits near .45-.55 alone [inference]. | sub-0 train, as now | Legal: no test fitting; MIT recipe, re-implemented in our code | ~2 h code (an `OnsetEncoding` off / no-span flag may already exist; check `model/` before building) + 1 Deep run + CPU fusion sweep | 0 to +1.5 | ~20% |
| 2 | **LM fusion over the isolated expert** | Our realistic LM beliefs captured −.012 on the joint model (`2026-10-12_information_budget.md`). One explanation: timing already encodes word length / frequency, which is what an n-gram LM also knows. 2609.40359 finds the LM prior only pays once the timing shortcut is removed. So: apply LM rescoring to expert #1, or to its fused output. | 2609.40359 (Qwen3-8B on SimpleB2T: "substantially" better; numbers are on WER, not bacc10). LM λ=1.5 in 2505.13446 (`2026-10-08_frontier_sweep.md`). | needs #1; an OSI LM | Legal if LM licence is OSI (rule 6c); the in-tree `lmfusion.py` / `lmscore.py` are work in progress (untracked) | 0 extra training; CPU eval after #1 | 0 to +1 | ~15% |
| 3 | **Timing (modality) dropout at train** | Zero the onset features for p≈.1-.3 of spans during training, keep them at test. The encoder must carry the label on those spans. This is NOT the 7gf.98/99 adversary: those removed timing from the representation at test as well. | [Brain2Qwerty v2, 2608.18114](https://arxiv.org/abs/2608.18114): 10% modality dropout (`2026-10-06_munich_competitor_prior_art.md`); its isolated effect is not reported [unverified]. | as now | Legal | ~1 h code + 1-2 Deep runs (p = .1, .3) | 0 to +1 | ~10% |
| 4 | **Training-run adversary on the pool (ALIGN mechanism, source runs only)** | Run identity is 3.01 of the pool's 3.72 bits and 14.8% of its variance (audit §1-2). A GRL run classifier on *training* runs only pushes the pool toward run-invariant word features. | [ALIGN, arXiv 2603.18299](https://arxiv.org/abs/2603.18299): intracortical ALS, T12 WER 60.01 → 46.50. It **trains on unlabelled target sessions**, which our own rule forbids (not the organisers; rules.md:35). The source-only variant is untested in the literature. Risks: our timing GRL collapsed (7gf.99); run-centring was only +.009. | as now | Source-only: legal. Target-session ALIGN: blocked by our rule. | ~2-3 h code + 2 Deep runs (adv weight .01, .05) | 0 to +1 | ~10% |
| 5 | **Train-time same-word grouping / averaging (re-test)** | Average 2-4 same-word train windows into one sample, raising SNR on the training target. | [MEGConformer 2512.01443](https://arxiv.org/abs/2512.01443): dynamic grouping 38.4 → 44.3 on phonemes. d'Ascoli 2412.17829: 8-rep averaging doubles accuracy, *at test*. Our `regularise/average.py` exists; bead 5e6 is open for a re-test. avg2p50 showed +3.5 sd more MEG reliance (efd). | as now | Legal at train; test-time grouping is blocked | 1 Deep run (flags exist) | 0 to +1 | ~10% |
| 6 | **DSS / xDAWN spatial denoiser fit on train repeats** | Find spatial filters that maximise across-repeat reproducibility of the same word, fit on train, and apply as a fixed linear front end at test. | [GED tutorial, Cohen 2021](https://arxiv.org/abs/2104.12356); [DSS, Särelä & Valpola 2005](https://www.researchgate.net/publication/220320724_Denoising_Source_Separation). No published deep-decoding gain on speech MEG found. Our linear ceiling says linear MEG beyond timing is +.006 bits, so a linear filter is unlikely to unlock much. | as now | Legal (fixed filter, not fit on test) | ~3 h code (bead 3ib) + 1 Deep run | 0 to +.5 | ~5% |

**Priced out by the deadline** (each >1 day, or needs new data):
- External same-paradigm MEG pretraining: LPP ds007523 (CC0) and SMN4Lang (CC0, on disk); see
  `2026-10-09_external_meg_speech.md`. Data volume is the one axis that clearly moves MEG bits: +.10 on Sherlock1
  from 11× data (signal ceiling doc). But those are other subjects, scanners and languages, and the conversion plus
  pretrain cost exceeds the remaining wall-clock.
- SuLoRA / per-subject deltas (`2026-10-12_adapt_generalisation.md` #5) are a Broad lever, not a Deep MEG-share one.

**Blocked by the rules:**
- TTA on test MEG ([2509.24700](https://arxiv.org/abs/2509.24700)).
- ALIGN with target sessions.
- Test-time same-word aggregation (SimpleB2T k>1; d'Ascoli 8-rep).
- Cross-subject test pooling (7gf.15).
- EA whitening: legal only on Broad (per-run EA, already +.038 there, 7gf.56). It hurt on Deep.

## 4. Reading

- **No lever has a published number at ≥1.5 pt on the MEG share under our constraints.** The only big published
  levers are test-time repeat aggregation and a language prior on top of a shortcut-free model. The first is blocked;
  the second has failed for us so far on the joint model.
- **#1 + #2 is the one structural bet.** It is the only path the new paper suggests that can reach the bar:
  - the expert gives decorrelated MEG evidence;
  - fusion recovers the timing information;
  - the LM may finally add once timing no longer duplicates it.
  Cost: about 1 run plus CPU fusion. Expected about +.5 to +1, with roughly 1-in-5 odds of clearing 1.5. Best ratio
  on the list.
- **#3-#5 are knob-sized.** Run them only in GPU gaps. Report any gain as directional.
- **Ceiling arithmetic.** MEG-specific signal is about .10 bacc10 / .55 bits. Even doubling its usable share is
  about +.05 only if fusion is lossless. The realistic gain from any one lever above is a fraction of that.

## Sources

Inline above. Also re-read:
- [LibriBrain100, 2608.25204](https://arxiv.org/abs/2608.25204)
- [PNPL 2026 competition, 2609.03231](https://arxiv.org/abs/2609.03231)
- [Défossez et al. 2023](https://www.nature.com/articles/s42256-023-00714-5): subject layer and CLIP loss; covered in
  the 10-06 doc.
- [d'Ascoli et al. 2024/25, 2412.17829](https://arxiv.org/abs/2412.17829)
