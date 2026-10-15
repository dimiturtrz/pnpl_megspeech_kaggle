# Where we really are (audit, 2026-10-07)
> Superseded by [2026-10-13](2026-10-13_where_we_are.md).

Sources: `.beads/issues.jsonl` (189 closed issues), 496 commits, every `interpretations/` doc, and the per-epoch
telemetry of the current recipe. Three read-only axis audits (supervision, input, sentence/eval) were cross-checked
against bd. Current ruler: kcut (Kaggle-length cut). Seed floor of ema-cdrop25: val .6136 sd .0066, local test
.6548 sd .019. The bar is ~.015.

## The score, honestly

| | Deep public | Broad public |
|---|---|---|
| Ours, best | .6638 (ens-cd8lf16, 24 models) | .5879 (R1) |
| Ours, best single | .6477 (ema-cdrop25-s1) | .5633 |
| Leader | ~.765 | — |

The trajectory:

- **Oct 1–2, the large steps:**
  - prior .20
  - CNN .22
  - span context .31
  - sentence spans .569, the largest single jump (+.18)
  - 4-seed ensemble .637
- **Oct 3–7, five days of wobble around .60–.66:**
  - The net gain is +2.7 pt.
  - The EMA + context-dropout .25 recipe and ensembles produced all of it.
  - No single-model mechanism contributed any of it.

The leader gap is ~10 pt. Their Deep−Broad gap is 3.6 pt, against our 6.9 (7gf.114). They generalise across
subjects far better, which points to more data or pretraining rather than a better head.

## What the model actually is

Per-epoch telemetry for ema-cd25-xsp10sem05-ft-s1:

| | Epoch 9 | Epoch 39 |
|---|---|---|
| fit (training spans) | .76 | .856 |
| val | .616 | .643 |
| val with MEG shuffled | .52 | .53 |
| neural share (val − shuffled) | .10 | .12 |

- **About 80% of the score is the context/text prior.** MEG adds ~.11.
- **The word head alone is weak.** On MOCHA it scores .29, against .875 for the context head.
- **The fit–val gap grows to .21 after epoch 24**, while val stays flat.
- **The gap is mostly MEG memorisation** (7gf.128, `studies.gap`, 10:18). Fit and val each scored real,
  MEG-shuffled and order-shuffled:

  | Model | Real fit / val (gap) | MEG-shuffled gap | Order-shuffled gap |
  |---|---|---|---|
  | ema-cdrop25-s1 | .776 / .603 (.174) | .031 | .058 |
  | xsp10sem05-ft-s1 | .776 / .619 (.157) | .005 | .068 |

  - With MEG removed, the text prior alone generalises: the fit-val gap collapses to .03 or below.
  - The MEG share is .22–.24 on training spans against .075–.086 on val. The encoder reads trial-specific MEG
    of words it trained on.
  - Caveat: the fit-side MEG shuffle draws windows from all sources (the val shuffle stays within a domain), so the
    MEG share on fit may be somewhat overstated.
  - The levers this points to are on the MEG side: SNR and trial-level regularisation, such as trial averaging
    (5e6 arm 3, now running), rather than more text-side masking.
- **Front-end arms that lower the MEG share score worse.** The set-pool d25 arm has a neural share of .049.

## The ledger: ~110 ideas

| Outcome | Count | Notes |
|---|---|---|
| Cleared .015 on the current ruler, single model | **0** | — |
| Kept in the recipe | sentence spans, span transformer, EMA .9998, context dropout .25, tau .5, EA (Broad), BGMix (Broad) | Most predate Oct 3 |
| Ensemble-level gains | label-free members (+.007 public Deep), KD students (+.009, old ruler) | ~1 pt is the bound: seed-to-seed correlation is .97 |
| Null or killed on ONE run | ~45 | — |
| Null or killed on an OLD ruler or recipe | ~40 | Pre-EMA, sentence cut (overstates by 3.6–5.4 pt), or the n=2512 val that also selected tau and epoch. On that ruler, local vs public Spearman was −0.29 (h1u.13), so those nulls AND those gains are both unranked. |
| Harness-suspected verdicts | ~12 | fp16-NaN augments, patience-8 truncation, unnormalised KD teacher, type-level POS tags, unwhitened teachers (99y: only 11–24 predictable dims; its wav2vec2 row read misplaced targets, fixed 10-15: 27 family-significant dims, teacher-probe .94 not .22 — megspeech-e1s), phoneme accuracy never logged, MEG-XL LoRA divergence, the set-pool crash (fixed in 539ec2c) |
| Proposed, never run | ~35 | See below |

### By axis

- **POS:** every arm ran on the old ruler with type-level tags (7gf.93/94/95–97, cn2 +.011 directional). An
  in-context POS target has never been run on cd25.
- **Audio:**
  - wav2vec2 speech aux: null.
  - Anchors: killed by the 16-d bottleneck.
  - Label-free speech stage 1: at parity, and the source of ensemble diversity.
  - Dense acoustic aux (7gf.125.1) and the stage-split acoustic expert (7gf.75): never run.
- **Text:**
  - semantic, lexical, GPT-2 in-context: null as aux terms, at parity as stage 1.
  - LM fusion: bigram, BERT and oracle all killed. The causal LM + beam code is written but was never run (lvj).
- **Sentence / context:**
  - Spans are the one big win.
  - IOI timing's gain was an artefact of the ruler (7gf.105).
  - Word mask: the earlier result was invalid (patience bug) and a rerun is pending. WordMask hides the word, which
    trains the text side further.
  - Neighbour heads: noise.
- **Channel / sensor:**
  - Triplet/site tokens, sensorwise filters, site attention, pyramid, set-pool and SSS subspace are all null or
    worse.
  - Physics says this axis is closed: post-SSS rank is ~76, mixing is quasi-static, imaginary coherency sits at the
    floor.
- **Normalisation:** LW whitening amplifies the tail ~200×; noise whitening is null. EA is the default on Broad
  and dead on Deep.
- **Augmentation:** specaug40 (MEGConformer band-stop at chance .4, already run) is null across 2 seeds; noise,
  amask, spatial dropout, drop-path and dropout .35 are all null.
- **Encoders:**
  - conv = conformer (tie); capacity sweeps are flat.
  - MEG-XL frozen on word windows scores .36, while it was pretrained on 150 s contexts.
  - MEG-XL LoRA diverged at epoch 1 and was never retried.

## The one strong measurement nobody acted on

**7gf.104 data scaling: ~+3 pt per doubling of training data, not saturated.** It is the largest measured effect
in the project, and it matches the leader's cross-subject advantage. We cannot get more labelled subject-0 data.
What remains are routes that BORROW data:

1. **MEG-XL fine-tuned at long context** (30–60 s spans). That puts ~150 s-context multi-subject pretraining in
   our encoder. The cost is ~3–8 h per run plus 3–5 recipe tries. Odds of clearing the bar: ~20%.
2. **Multi-subject SSL (ct5, approved).** Limited because the subject 1–32 MEG on disk is only ses-11/12. Those
   sessions are forbidden for LABELLED use. Using them unlabelled for SSL needs the owner's reading of the rule.
3. **Same-word trial averaging in training** (5e6 arm 3, MEGConformer +6 F1). It raises SNR per example without
   new data. It has never been run, and it is cheap.

## Ranked plan (wall-clock ÷ expected gain)

| # | Run | Cost | Odds of clearing 1.5 pt | Why |
|---|---|---|---|---|
| 1 | 7gf.128 gap locator | 20 min eval | instrument | Decides whether the .21 gap is text or MEG, and so which side to attack |
| 2 | Same-word trial averaging (training) | ~2 h code + 2 h run | ~15% | Data-lever proxy; never tried |
| 3 | MEG-XL long-context frozen smoke run, then fine-tune | 1 h, then about a day | ~20% | Borrowed data; context mismatch with our word windows explains the .36 |
| 4 | Distil cd8lf16 into one model (feature-level, per-teacher heads) | ~3 h | ~15% | A single model at ensemble level frees the ensemble for a second step |
| 5 | Teacher fix 99y: reduced-rank targets or soft labels | ~3 h | ~10% | Every text/audio null used unwhitened targets |

Dropped as knob space: further front-end variants, augmentation doses, dropout levels, head
weights.

## Discrepancies still open

- E0/E1: local .694/.681 against public .620/.592, a 7–9 pt drop where the recipe usually drops ~4.
- Single-seed public scatter is ~6 pt against ~1 pt locally (bgmix s0/s1).
- Refit is negative on both tracks, with no cause isolated.
- Rescore gap .0066 (4qa).
