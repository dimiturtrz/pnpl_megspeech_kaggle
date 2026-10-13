# Final-evaluation delivery

The organisers rank on held-out data "recorded with stimuli that are entirely different from anything released", and
ask top teams for "runable model checkpoints" (`competition/rules.md:27-30`). This page lists what to hand over for
each pick, and the command that turns checkpoints and a holdout directory into a submission. The rationale for the
picks is in megspeech-7gf.29.1. The rebuild audit is megspeech-7gf.29.1.1.

## Inputs

- **Checkpoints:** `<data>/models/pnpl_libribrain/<member>/` for each member below (`model.pt`, `metrics.json`).
- **Data root:** `paths.yaml` → `data:`. Holdout archives go in `raw/<pnpl HOLDOUT_SUBDIR>/subjNN_holdout{01_sentence,2_word}.npz`,
  in the layout of `pnpl.competition.holdout`, which the organisers' own loader reads.
- Inference also reads one training-run h5 to get the channel layout (`KaggleCache.layout`) and the training word
  table for the prior.
- Archives replaced after caching are re-cached (7gf.29.1.2). A fresh `processed/` is still the clean option.

## Picks

| track | pick | members | flags | rebuild vs submitted |
|---|---|---|---|---|
| Deep | d-ac05xfam | `ema-cd25-xsp10sem05-ac05x-s{k}-ft-s{k}`, k = 1..4 | none | bit-exact |
| Deep | d-old4 | `ema-cd25-xsp10sem05-s{k}-ft-s{k}`, k = 2..5 | none | bit-exact |
| Broad | b-xc5k25ps | 29, listed in `xc5k25ps/metrics.json` | `--fallback.unseen mean --fallback.keep 0.25 --balance.per_subject --balance.toward 0.25 --balance.steps ?` | pending |

## Command

```bash
uv run python -m megspeech.submission <members...> --out <dir> [--track broad] [flags] [--device cpu]
```

This writes `<dir>/submission.csv` (or `broad_submission.csv`) and `metrics.json`. Since 7gf.29.1.1, `metrics.json`
records the members together with the scoring flags.

## Organiser data

- **Format:** the test rows are read by `pnpl.competition.holdout.LibriBrainCompetitionHoldout`, the organisers'
  own loader. New data in that layout runs unchanged. Any other layout needs an adapter to it.
- **Training data is required too.** The command re-selects its temperature on the labelled local ruler, so the
  training data has to be present (`python -m megspeech.data.raw.fetch`, public). The ruler does not depend on the
  test rows, so new test data gets the same temperature.
- **Runtime on CPU:** about 10 min for a 4-member Deep blend. Broad's 29 members take over an hour.

## Disclose

- **Broad:** the Sinkhorn balancing is transductive. It runs per subject over the test rows' output probabilities.
- No cross-track mapping is used (`competition/announcements.md`).
