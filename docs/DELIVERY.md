# Verification package

For the PNPL 2026 organisers: the checkpoints, configuration, dependencies and commands that reproduce our two
ranked submissions, one per track. Rules context: `competition/rules.md:27-30` asks top teams for "runable model
checkpoints".

## What is ranked

No submission was ticked on Kaggle, so each track scores its best-public one:

| track | submission | public | models | blend record |
|---|---|---|---|---|
| Deep | `d-ac05xfam16` | .69483 | 16 | `d-ac05xfam16/metrics.json` |
| Broad | `b-xc5k25b0ps` | .63097 | 29 | `zz-b-xc5k25b0ps/metrics.json` |

## Files

`megspeech_verification_checkpoints.zip` (1.3 GiB, shared by link with this repo), unzipped into `<data>/models/`:

```
pnpl_libribrain/
  d-ac05xfam16/        Deep blend: metrics.json (members + scoring) and the submitted submission.csv
  ema-cd25-…-ft-…/     Deep members, 16 dirs: model.pt, context.json, metrics.json
  zz-b-xc5k25b0ps/     Broad blend: metrics.json (members + scoring) and the submitted broad_submission.csv
  bg90-…/, bgmix05-…/, ema-bgmix05-…/   Broad members, 29 dirs: model.pt, context.json, metrics.json
  d-ac05xfam8/, zz-b-xc5k25b05ps/, xc5k25/   other picks, built from the same models (below)
```

Which model dir belongs to which track is the `members` list in each blend's `metrics.json`. The two lists are
disjoint. Every member's `metrics.json` `config` is its complete training configuration, and the blend's
`config.scoring` holds its inference flags. `hits.pt` and `probs.pt` are local-test outputs, which inference never
reads.

[`submissions.csv`](submissions.csv) lists all 259 Kaggle submissions: public score, model dir, models, and whether
the weights survive.

## Setup

```bash
git clone https://github.com/dimiturtrz/pnpl_megspeech_kaggle && cd pnpl_megspeech_kaggle
uv sync                                   # Python >=3.12; exact versions pinned in uv.lock (torch from the cu130 index)
cp paths.example.yaml paths.yaml          # set  data: <data>
uv run python -m megspeech.data.raw.fetch # public LibriBrain training data from HF (no token), ~120 GB
unzip megspeech_verification_checkpoints.zip -d <data>/models/
```

- **Test data:** holdout archives go in `<data>/raw/pnpl_libribrain/COMPETITION_HOLDOUT/subjNN_holdout{01_sentence,2_word}.npz`,
  in the layout `pnpl.competition.holdout.LibriBrainCompetitionHoldout` reads. New data in that layout runs
  unchanged. Clear `<data>/processed/` before swapping archives (re-caching is automatic, but a fresh cache is the
  clean option).
- **Why training data is needed at inference:** the softmax temperature is re-selected on the labelled local test
  (pnpl validation + test). One training h5 also gives the channel layout, and the training word table gives the
  prior. None of this depends on the test rows.

## Run

```bash
uv run python -m megspeech.submission --rebuild d-ac05xfam16    --out verify-deep  --device cpu   # Deep, ~10 min per 4 members on CPU
uv run python -m megspeech.submission --rebuild zz-b-xc5k25b0ps --out verify-broad --device cpu   # Broad, >1 h CPU
```

`--rebuild` reads members, scoring flags and track from the blend's record. Output goes to
`<data>/models/pnpl_libribrain/verify-*/` (`submission.csv` or `broad_submission.csv` plus `metrics.json`). Drop
`--device cpu` to use a GPU.

On our machine, a CPU rebuild of the released holdout matched what we submitted. Deep was bit-exact (960 rows, max
|Δp| 0). Broad agreed on 99.7% of top-1 picks, with mean |Δp| 4e-5 from float order.

## Other picks

| track | submission | how |
|---|---|---|
| Deep | `d-ac05xfam8` | `--rebuild d-ac05xfam8`: 8 of the 16 Deep models |
| Deep | `d-old4` | weights deleted in a 2026-10-20 disk cleanup. Not reproducible without retraining |
| Broad | `b-xc5k25b05ps` | `--rebuild zz-b-xc5k25b05ps`: same 29 models, Sinkhorn toward the prior^.05 |
| Broad | `b-xc5k25` | `--rebuild xc5k25`: same 29 models, no Sinkhorn (public .61238) |

## Training

`uv run python -m megspeech.training.cli.train --origin <member> --name <new>` re-runs a member from its recorded
config. The config names its warm start, a pretrained stem from `megspeech.training.cli.pretrain`. Stems are not in
the archive. The inference rebuild above is audited. An end-to-end retrain has not been re-run.

## External data and models

The rules do not address these (`competition/rules.md:35`), so we list them. Both ranked submissions train on
LibriBrain data only: `pnpl/LibriBrain`, `pnpl/LibriBrain2`, and the competition release. No other MEG dataset is
used, and nothing outside it is read at inference.

- **Deep:** the 8 pretraining stems (`ema-cd25-xsp10sem05-ac05x-s{1..8}`, recorded in each member's
  `config.optim.stem`) use three frozen, public targets during pretraining only:
  - `facebook/wav2vec2-large-xlsr-53` latents of the heard speech (weight 1.0);
  - `google-t5/t5-large` layer-12 embeddings of each heard word (weight .5);
  - the speech envelope (weight .5).

  The audio is the Sherlock chapter audio released with LibriBrain (`megspeech.data.raw.fetch`). The words and
  timings come from LibriBrain's event files. The 50-word fine-tune and inference use MEG only.
- **Broad:** no pretrained or external model. Training uses subject 0 plus the Broad subjects' released labelled
  data, the latter for per-subject adaptation.
- **Not in either ranked submission:** MEG-XL, GPT-2 language-model fusion, hubert and data2vec targets, and other
  MEG corpora (MEG-MASC and others). `fetch.py` can download these for studies; they are not needed to rebuild.

## Disclose

- **Broad `b-xc5k25b0ps` is transductive.** It runs 30 Sinkhorn steps per subject over the test rows' output
  probabilities, toward equal predicted mass per word (`megspeech/evaluation/metric/balance.py`). Its Kaggle
  description says "no Sinkhorn (toward 0)". That is wrong: toward 0 means a uniform target, not no balancing.
  `b-xc5k25` is the same models without it.
- Deep uses no test-time adaptation.
- No cross-track mapping is used (`competition/announcements.md`).
