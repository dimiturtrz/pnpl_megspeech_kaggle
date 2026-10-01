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

BAcc@10 (balanced top-10 accuracy, chance 20%); tiebreak BAcc@1; OVMI also reported.

| track | reference | BAcc@10 | BAcc@1 | ours |
|---|---|---|---|---|
| Deep | d'Ascoli 2025 | 73.23% | 25.60% | — |
| Broad | MEG-XL | 42.96% | — | — |

## Data

All CC-BY-NC-4.0, on HuggingFace under `pnpl/`. Sizes measured from the HF tree API (2026-10-01):

| dataset | contents | size |
|---|---|---|
| `LibriBrain` Sherlock1–7 (subj0) | preprocessed `.h5` + events `.tsv` | **~60 GB** |
| same | raw `.fif` (BIDS) | ~484 GB — **not needed** |
| `LibriBrain-Competition-2026` | holdout `.npz` | 5.7 GB |
| `LibriBrain2` | Broad subjects, raw + derivatives | 239 GB total — pull derivatives only |

Deep needs the `.h5` + `.tsv` only (what `pnpl`'s `LibriBrainWord` reads with
`preprocessing_str="bads+headpos+sss+notch+bp+ds"`). D: had 319 GB free at setup — the raw `.fif` would
not fit and is not used.

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
2. **Levers** (ranked by plausible size): contrastive alignment to pretrained speech/word embeddings
   (Défossez 2023 recipe), MEG-XL fine-tune, sentence context on the sentence holdout (neighbouring
   words' MEG + a language prior over the 50-word vocabulary).
3. **Broad**: Deep backbone + per-subject alignment (Euclidean/Riemannian) or subject layer; report the
   minutes-vs-accuracy curve.

## Open questions

- GPU split with `rsna_knee_kaggle` (deadline 2026-10-22).
- Exact Broad derivative footprint in `LibriBrain2` (measure before pulling).
- Which holdout subject ids are Deep (subj00?) vs Broad, and how word vs sentence holdouts are scored.
