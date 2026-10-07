# Teacher RSA: which 50-word geometry does MEG already share (megspeech-7gf.141.1)

Scope: model `ema-cd25-syn05-s1`, CPU. MEG matrix = mean prob-after-tau softmax of each true word over the 50
candidates, read on the scored (official validation) unseen-span words (2545 words, all 50 words present), symmetrised.
Teacher matrix = cosine of type-level word vectors. RSA = Spearman over the 1225 off-diagonal upper-triangle pairs; CI =
percentile interval of 1000 seeded resamples of the 50 words (pairs of a word with itself dropped). One model, one seed:
the CIs cover word sampling only, not model-seed variance.

Command: `CUDA_VISIBLE_DEVICES= uv run --no-sync python -m megspeech.evaluation.studies.rsa --run ema-cd25-syn05-s1`
(EmbeddingGemma 2 vectors are written once with `--write-gemma` under transformers >= 5.19, which the project venv,
at 5.18.0, does not yet recognise; the venv was left untouched).

| teacher | dims | words | RSA (Spearman) | 95% CI (bootstrap over words) | min pairwise cosine distance |
|---|---|---|---|---|---|
| one-hot (null) | 50 | 50 | +0.000 | [+0.000, +0.000] | 1.000 |
| t5-large h12 | 1024 | 50 | +0.308 | [+0.177, +0.432] | 0.073 |
| wav2vec2-large-xlsr-53 | 1024 | 50 | +0.117 | [-0.032, +0.257] | 0.000 |
| whisper-medium-L8 | 1024 | 50 | +0.138 | [-0.009, +0.277] | 0.000 |
| whisper-medium-L12 | 1024 | 50 | +0.131 | [-0.011, +0.268] | 0.000 |
| whisper-medium-L16 | 1024 | 50 | +0.135 | [-0.007, +0.273] | 0.001 |
| whisper-medium-L20 | 1024 | 50 | +0.139 | [-0.001, +0.278] | 0.001 |
| whisper-medium-L24 | 1024 | 50 | +0.150 | [+0.014, +0.290] | 0.001 |
| embeddinggemma-2 768 | 768 | 50 | +0.197 | [+0.042, +0.358] | 0.023 |
| embeddinggemma-2 256 | 256 | 50 | +0.186 | [+0.031, +0.335] | 0.021 |
| embeddinggemma-2 128 | 128 | 50 | +0.170 | [+0.010, +0.323] | 0.014 |

Not run: HuBERT-ll60k and PanPhon (no cached vectors; other beads).

## Reading
- The one-hot null is 0 by construction: its off-diagonal is constant, so there is no rank to correlate. "Beats
  one-hot" here means a CI above 0.
- t5-large (+0.31, CI +0.18..+0.43) and EmbeddingGemma 2 at every width (+0.17..+0.20, lower bounds +0.01..+0.04) clear
  zero. Whisper L24 clears it barely (+0.014); wav2vec2 and whisper L8-L20 do not.
- Against t5, no teacher is above it. The Gemma 768 CI overlaps t5's; the point gap (0.11) is inside the CI widths,
  so "t5 >= Gemma" is a point estimate, not established.
- Inference (not tested): the speech type means have cosine distance ~0 between distinct words, i.e. one dominant common
  direction in the raw per-token states; the raw cosine is then nearly flat across words, which plausibly depresses their
  RSA. A centred/whitened variant was not run, so these rows say "raw type means", not "acoustic teachers".
- Inference (not tested): the MEG confusions track text/semantic similarity more than raw acoustic similarity, but
  confusions among short function words also carry length and timing; RSA cannot separate these.
- Directional only: all gaps are below the 1.5%-absolute bar's analogue for downstream accuracy; this is a ranking
  instrument, not a result about training gains. Matryoshka truncation changes RSA by <= 0.03, within noise.
