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

## Structure

- `megspeech/` — the pipeline (data → model → eval → submission). `core/` — task-agnostic kernel.
- `comp/` — submission glue + competition facts (not `kaggle/`: shadows the installed package).
- `tests/` — unit / integration / e2e.

## Approach

SOTA gather → replicate → build past it.

1. **Replicate** the Deep reference on the official splits — the ruler every later number is read against.
2. **Levers** (ranked by plausible size): contrastive alignment to pretrained speech/word embeddings
   (Défossez 2023 recipe), MEG-XL fine-tune, longer context window around the word.
3. **Broad**: Deep backbone + per-subject alignment (Euclidean/Riemannian) or subject layer; report the
   minutes-vs-accuracy curve.

## Open questions

- GPU split with `rsna_knee_kaggle` (deadline 2026-10-22).
- Exact Broad derivative footprint in `LibriBrain2` (measure before pulling).
