# Teacher RSA: which 50-word geometry does MEG already share (megspeech-7gf.141.1)

Scope: CPU. MEG matrix = mean tau-adjusted softmax of each true word over the 50 candidates, on the scored unseen-span
VAL words (2545 words, all 50 words present), symmetrised. Teacher matrix = cosine of type-level word vectors in two
forms: RAW, and CENTRED (mean over the 50 type vectors subtracted before the cosine). RSA = Spearman over the 1225
off-diagonal pairs; CI = percentile interval of 1000 seeded word resamples. CIs cover word sampling only, not model-seed variance.

Command: `CUDA_VISIBLE_DEVICES= uv run --no-sync python -m megspeech.evaluation.studies.rsa --run ema-cd25-syn05-s1 ema-cd25-s0r ema-cd25-s10`

EmbeddingGemma 2 vectors were written once with `--write-gemma` under a side-installed transformers 5.19 (the venv's
5.18.0 does not know the architecture). Not reproducible from pyproject yet.

## Circularity check
`ema-cd25-syn05-s1`: softmax CE (label smoothing 0.1) plus a syntactic-tag aux 0.5. Semantic, lexical, speech,
phonetic and teacher terms are all off, so nothing regressed onto t5 or any listed teacher; its geometry is not
t5-induced by training targets. A syntactic head could still shape geometry, so the table is also run on two plain-CE
runs with no auxiliary term (`ema-cd25-s0r`, `ema-cd25-s10`; many such runs exist). Chosen over a model-free template
geometry because the plain-CE runs already exist and cost seconds on CPU.

## ema-cd25-s0r (plain CE, no aux: circularity-free reference)
| teacher | dims | words | RSA raw | 95% CI raw | min cos dist raw | RSA centred | 95% CI centred | min cos dist centred |
|---|---|---|---|---|---|---|---|---|
| one-hot | 50 | 50 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | [+0.000, +0.000] | 1.020 |
| t5-large h12 | 1024 | 50 | +0.304 | [+0.176, +0.425] | 0.073 | +0.284 | [+0.174, +0.401] | 0.287 |
| wav2vec2-large-xlsr-53 | 1024 | 50 | +0.127 | [-0.018, +0.265] | 0.000 | +0.183 | [+0.057, +0.301] | 0.012 |
| whisper-medium-L8 | 1024 | 50 | +0.150 | [+0.004, +0.286] | 0.000 | +0.181 | [+0.058, +0.296] | 0.012 |
| whisper-medium-L12 | 1024 | 50 | +0.145 | [+0.002, +0.278] | 0.000 | +0.180 | [+0.057, +0.290] | 0.014 |
| whisper-medium-L16 | 1024 | 50 | +0.148 | [+0.005, +0.282] | 0.001 | +0.183 | [+0.061, +0.296] | 0.015 |
| whisper-medium-L20 | 1024 | 50 | +0.150 | [+0.007, +0.284] | 0.001 | +0.192 | [+0.067, +0.309] | 0.013 |
| whisper-medium-L24 | 1024 | 50 | +0.162 | [+0.021, +0.294] | 0.001 | +0.203 | [+0.076, +0.327] | 0.010 |
| embeddinggemma-2 768 | 768 | 50 | +0.184 | [+0.037, +0.329] | 0.023 | +0.219 | [+0.114, +0.336] | 0.180 |
| embeddinggemma-2 256 | 256 | 50 | +0.174 | [+0.028, +0.315] | 0.021 | +0.173 | [+0.063, +0.285] | 0.201 |
| embeddinggemma-2 128 | 128 | 50 | +0.157 | [+0.003, +0.306] | 0.014 | +0.154 | [+0.045, +0.271] | 0.210 |

## ema-cd25-syn05-s1 (syntactic aux 0.5)
| teacher | dims | words | RSA raw | 95% CI raw | min cos dist raw | RSA centred | 95% CI centred | min cos dist centred |
|---|---|---|---|---|---|---|---|---|
| one-hot | 50 | 50 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | [+0.000, +0.000] | 1.020 |
| t5-large h12 | 1024 | 50 | +0.308 | [+0.177, +0.432] | 0.073 | +0.283 | [+0.162, +0.401] | 0.287 |
| wav2vec2-large-xlsr-53 | 1024 | 50 | +0.117 | [-0.032, +0.257] | 0.000 | +0.177 | [+0.048, +0.299] | 0.012 |
| whisper-medium-L8 | 1024 | 50 | +0.138 | [-0.009, +0.277] | 0.000 | +0.175 | [+0.048, +0.294] | 0.012 |
| whisper-medium-L12 | 1024 | 50 | +0.131 | [-0.011, +0.268] | 0.000 | +0.174 | [+0.048, +0.290] | 0.014 |
| whisper-medium-L16 | 1024 | 50 | +0.135 | [-0.007, +0.273] | 0.001 | +0.178 | [+0.052, +0.292] | 0.015 |
| whisper-medium-L20 | 1024 | 50 | +0.139 | [-0.001, +0.278] | 0.001 | +0.189 | [+0.062, +0.310] | 0.013 |
| whisper-medium-L24 | 1024 | 50 | +0.150 | [+0.014, +0.290] | 0.001 | +0.200 | [+0.074, +0.327] | 0.010 |
| embeddinggemma-2 768 | 768 | 50 | +0.197 | [+0.042, +0.358] | 0.023 | +0.206 | [+0.098, +0.330] | 0.180 |
| embeddinggemma-2 256 | 256 | 50 | +0.186 | [+0.031, +0.335] | 0.021 | +0.160 | [+0.038, +0.279] | 0.201 |
| embeddinggemma-2 128 | 128 | 50 | +0.170 | [+0.010, +0.323] | 0.014 | +0.147 | [+0.039, +0.268] | 0.210 |

## ema-cd25-s10 (plain CE, second seed)
| teacher | dims | words | RSA raw | 95% CI raw | min cos dist raw | RSA centred | 95% CI centred | min cos dist centred |
|---|---|---|---|---|---|---|---|---|
| one-hot | 50 | 50 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | [+0.000, +0.000] | 1.020 |
| t5-large h12 | 1024 | 50 | +0.299 | [+0.166, +0.415] | 0.073 | +0.285 | [+0.169, +0.406] | 0.287 |
| wav2vec2-large-xlsr-53 | 1024 | 50 | +0.138 | [-0.000, +0.269] | 0.000 | +0.192 | [+0.069, +0.312] | 0.012 |
| whisper-medium-L8 | 1024 | 50 | +0.159 | [+0.015, +0.295] | 0.000 | +0.191 | [+0.067, +0.310] | 0.012 |
| whisper-medium-L12 | 1024 | 50 | +0.154 | [+0.012, +0.286] | 0.000 | +0.191 | [+0.069, +0.309] | 0.014 |
| whisper-medium-L16 | 1024 | 50 | +0.157 | [+0.018, +0.288] | 0.001 | +0.195 | [+0.073, +0.311] | 0.015 |
| whisper-medium-L20 | 1024 | 50 | +0.159 | [+0.022, +0.288] | 0.001 | +0.205 | [+0.081, +0.323] | 0.013 |
| whisper-medium-L24 | 1024 | 50 | +0.170 | [+0.036, +0.298] | 0.001 | +0.215 | [+0.092, +0.336] | 0.010 |
| embeddinggemma-2 768 | 768 | 50 | +0.189 | [+0.028, +0.338] | 0.023 | +0.216 | [+0.109, +0.334] | 0.180 |
| embeddinggemma-2 256 | 256 | 50 | +0.181 | [+0.024, +0.326] | 0.021 | +0.176 | [+0.064, +0.291] | 0.201 |
| embeddinggemma-2 128 | 128 | 50 | +0.162 | [+0.003, +0.307] | 0.014 | +0.159 | [+0.053, +0.274] | 0.210 |

One-hot is 0 by convention in both forms (constant off-diagonal).

## Reading
- t5 is not circular: +0.30 raw / +0.28 centred on all three models; teacher values differ across models by <= 0.03, below CI widths.
- Centring raises xlsr/whisper from +0.12..+0.17 raw (CIs touching 0 on syn05) to +0.17..+0.22, lower bounds +0.05..+0.09,
  clear of zero on all three. Inference (not tested separately): the removed common direction masked their structure.
  Min centred cosine distance ~0.01 for speech teachers vs 0.18-0.29 for t5/Gemma-768 (raw 0.000 vs 0.07/0.02).
- Centred, t5 (+0.28) has the top point estimate; whisper L24 (+0.20..+0.22) and Gemma 768 (+0.21..+0.22) follow, then
  other whispers/xlsr (+0.17..+0.20), Gemma 256/128 (+0.15..+0.18). t5's lead (~0.07) and all gaps among the rest are
  inside CI widths (~0.23): directional, not separable here.
- Centring helps Gemma 768 slightly, lowers 256/128 slightly (<= 0.03, noise-level).
- Every non-one-hot teacher has rho <= 0.3 (<= ~9% of rank variance). This ranks candidate geometries only; it says
  nothing about downstream accuracy.
- Not run: HuBERT-ll60k, PanPhon (no vectors).
