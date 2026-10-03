# megspeech — PLAN (source of truth)

This is the canonical plan. Read it at session start before proposing work. Substance lives here;
ordered execution status lives in `ROADMAP.md`.

## Goal

Enter **PNPL 2026 / LibriBrain100** (closes **2026-10-15 AoE**): classify heard words from non-invasive MEG.
Two tracks, one repo — **Deep** builds the base pipeline, **Broad** reuses its backbone.

| track | data | task |
|---|---|---|
| Deep | subj0, ~80 h MEG, 306 sensors | 50-word vocabulary classification |
| Broad | 32 subjects × ~40 min (subj0 allowed); calibration cut to 40 / 20 / 10 min | same vocabulary, cross-subject |

Rules that shape the work: any external data / pretrained model allowed; prize requires beating the
reference baseline; top 3 per track, a team wins at most once; winners provide checkpoints (maybe code);
holdout split into public + final partitions. Site: libribrain.com/editions/2026 · paper arXiv 2609.03231.

## Headline metrics

BAcc@10 (balanced top-10 accuracy over the 50 words, chance 20%); tiebreak BAcc@1; OVMI also reported.
Kaggle lists the metric as `Balanced-Acccuracy` (sic); `pnpl`'s `write_submission` docstring says "argmax",
which contradicts the FAQ's BAcc@10. Which one the live scorer computes is unverified, so we report both.

| track | reference | paper BAcc@10 (own test split) | holdout BAcc@10 (3rd-party rerun) | LB top (2026-10-01) | ours |
|---|---|---|---|---|---|
| Deep | d'Ascoli 2025 | 73.23% (BAcc@1 25.60%) | 46.7% | 76.5% (63 teams) | — |
| Broad | MEG-XL | 42.96% | 30.8% | 72.9% (28 teams) | — |

The paper numbers are NOT the holdout ruler. The holdout rerun is from gereonelvers/2026-pnpl-competition-baselines,
which has no license. It found both baselines need sentence context: d'Ascoli's CNN branch alone scores below
chance on isolated words. The leaderboard is far above the references, so a faithful replication buys
eligibility, not rank.

## Submission

- **Kaggle:** `pnpl-competition-2026-deep` / `-broad`; 10 submissions/day, team ≤ 10; closes **2026-10-16 12:00 UTC**
  (the site says 15 Oct AoE, about the same instant).
- **CSV:** `index`, 50 primary-vocabulary columns in `pnpl/competition/data/vocabulary.csv` order, then 50
  `moses_<word>` columns (secondary vocab, unscored per pnpl; 101 columns, 960 rows in the Kaggle example). That
  file spells `it’s` with a curly apostrophe; d'Ascoli's list uses `its`. Values must be finite and ≥ 0.
  Write it with `pnpl.competition.submission.write_submission`; never hand-roll it.
- **Rows:** deep has 960 = 868 sentence-word + 92 word. Broad has 37,439, covering subjects 1–39 (33–39 are
  zero-shot per the code; the site says 1–32). Canonical order: subjects ascending, sentence before word,
  then stored order — iterate `LibriBrainCompetitionHoldout` and the order matches by construction.
- **Scored rows:** the public/private split is not exposed. Third-party notes have word rows as Ignored.
  Rules: "final rankings are determined on independent holdout data recorded with stimuli that are entirely
  different" — the mechanism is unstated. Overfitting the public LB is therefore doubly pointless.
- **Prize:** must beat the reference baselines; the top 3 provide checkpoints and maybe training code.

## Data

All CC-BY-NC-4.0, on HuggingFace under `pnpl/`. Sizes measured from the HF tree API (2026-10-01):

| dataset | contents | size |
|---|---|---|
| `LibriBrain` Sherlock1–7 (subj0) | preprocessed `.h5` + events `.tsv` | **~60 GB** |
| same | raw `.fif` (BIDS) | ~484 GB — **not needed** |
| `LibriBrain-Competition-2026` | holdout `.npz` | 5.7 GB |
| `LibriBrain2` sub-0 | Sherlock8–9, TIMIT, MOCHATIMIT, TheMoth (= Podcasts) `.h5` + `.tsv` | **~36 GB** |
| `LibriBrain2` subj 1–32 | Sherlock1 ses-11/12 `.h5` (0.1–0.5 GB each) | ~20 GB |
| `LibriBrain2` | everything incl. raw `.fif` | 239 GB — never pulled whole |

Deep needs the `.h5` + `.tsv` only; `pnpl`'s download fetches nothing else (default
`preprocessing_str="bads+headpos+sss+notch+bp+ds"`). D: had 319 GB free at setup — the raw `.fif` would
not fit and is not used.

**Partitions** (`pnpl` 0.2.0 `datasets/libribrain100/manifest.py`):

| data | train | validation | test |
|---|---|---|---|
| sub-0 Sherlock1–9 | everything else | Sherlock1 ses-11 run2 | Sherlock1 ses-12 run2 |
| sub-0 TIMIT | all 14 sessions | – | – |
| sub-0 MOCHA | ses 1, 4 | ses 2 | ses 3 |
| sub-0 Podcasts | all others | 29 | 30 |
| subjects 1–32 | none | Sherlock1 ses-11 run1 | Sherlock1 ses-12 run1 |

The Deep training set is therefore Sherlock1–**9** plus TIMIT, MOCHA and Podcasts (HF dirs `MOCHATIMIT`,
`TheMoth`) — ~96 GB of sub-0 `.h5` across the two repos (measured 2026-10-01). Fetch: `python -m megspeech.data.raw.fetch`. For Broad, `validation` IS the SFT set: `partition="train"`
raises.

**`pnpl` loader pitfalls:**
- `LibriBrain100Word` builds its vocabulary from every word in the selected runs, sorted. It does no
  lowercasing, has no 50-word restriction, and maps an unknown word to label 0. Map onto the competition
  vocabulary ourselves.
- Standardisation is a z-score clipped at ±10, with stats pooled over the dataset's own runs. Each partition
  computes its own stats unless `channel_means`/`channel_stds` are passed in. Train stats must be carried
  to val/test/holdout.
- The default window is 0–0.5 s; the holdout gives 1 s.

**Preprocessing is fixed**, train and test alike (file names: `proc-bads+headpos+sss+notch+lfreq01+hfreq125+sfreq250`):
bad-channel marking → cHPI head-position → SSS/Maxwell with movement compensation → line-noise notch →
0.1–125 Hz band-pass → 250 Hz. No raw holdout exists, so anything we do must be a transform of THAT
signal applicable at test time (re-filtering inside the band, spatial projections, alignment,
normalization). Raw `.fif` is at most a pretraining source.

**Holdout format** (`.npz` headers read 2026-10-01; 80 files = 40 subjects × 2):

| file | arrays |
|---|---|
| `subjXX_holdout2_word.npz` | `meg` (92, 306, 250) f32 — 1 s windows, `tmin` 0, `tmax` 1; `sfreq` 250 |
| `subjXX_holdout01_sentence.npz` | `meg` (156, 306, 1030) f32 + `sentence_sample_mask`, `sentence_n_times`; `word_onsets_s` / `word_mask` (156, 9); `sentence_extra_end_s` 1.0 |

The sentence holdout carries every word's onset with its surrounding MEG — the only place cross-word
context is available at test time.

**Layout** (root `D:/data/neural`, shared with mindscape; `core/paths.py`): `raw/pnpl_libribrain/`,
`processed/pnpl_libribrain/`, `models/pnpl_libribrain/` (ours), `models/foreign/<name>/` (MEG-XL etc.).

## Structure

- `megspeech/` — the pipeline (data → model → eval → submission). `core/` — task-agnostic kernel.
- `comp/` — submission glue + competition facts (not `kaggle/`: shadows the installed package).
- `tests/` — unit / integration / e2e.

## Approach

SOTA gather → replicate → build past it.

1. **Replicate** the Deep reference on the official splits — the ruler every later number is read against.
   Recipe (github.com/neural-processing-lab/dascoli-word-decoding, CC-BY-NC-4.0):
   - Backbone: SimpleConvTimeAgg CNN — hidden 160, depth 5, ChannelMerger 306→270, subject layers,
     attention time-pooling, 1024-d output.
   - Context: a sentence Transformer on top — depth 16, 16 heads, dim 1024, rotary.
   - Objective: SigLIP against frozen t5-large word embeddings (layer 12/25, token-mean); prediction is
     cosine retrieval over the 50 words.
   - Input: 3 s window from onset; re-band-pass 0.1–40 Hz; 50 Hz; RobustScaler per channel; baseline
     0–0.5 s; clamp ±5.
   - Training: AdamW lr 1e-4, wd 0, cosine over 50 epochs, patience 10, batch 128. Early-stop on val
     BAcc@10 (macro, 50-word vocab).
   - Check: the holdout gives each word only 1 s plus sentence context (`sentence_extra_end_s` 1.0), not 3 s.
     How the 3 s window maps onto the sentence holdout is the first thing to read in their code.
2. **Levers** (ranked by plausible size): contrastive alignment to pretrained speech/word embeddings
   (Défossez 2023 recipe), MEG-XL fine-tune, sentence context on the sentence holdout (neighbouring
   words' MEG + a language prior over the 50-word vocabulary).
3. **Broad**: Deep backbone + per-subject alignment (Euclidean/Riemannian) or subject layer; report the
   minutes-vs-accuracy curve.

## Open questions

- GPU split with `rsna_knee_kaggle` (deadline 2026-10-22).
- Exact Broad derivative footprint in `LibriBrain2` (measure before pulling).
- Does the live scorer compute top-10 or argmax, does it score word rows, and how does it match `it’s`?
  The first submission (the example CSV, then a trivial prior) answers the format half.
- MEG-XL (MIT, 2 × ~284 MB checkpoints; wants a 40–80 GiB GPU): does it fine-tune within the 5090's 32 GB?
