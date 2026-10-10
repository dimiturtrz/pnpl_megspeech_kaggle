# Deep: which teacher geometry matches what the MEG blend confuses (2026-10-10)

Setup: `python -m megspeech.evaluation.studies.rsablend m11wd` (eef38ae). The MEG matrix is the mean softmax of
m11wd's 36 members on official-val unseen-span words (n=2545, all 50 classes), symmetrised to a 50×50 confusion.
Each teacher gives a 50×50 cosine matrix from per-word means over subject-0 training words. Spearman rho over the
off-diagonal, 95% CI from 1000 bootstrap resamples over the 50 words. "Centred" removes the mean type vector first.

| teacher | rho raw | rho centred |
|---|---|---|
| word frequency (control) | +.055 [−.090, +.217] | — |
| t5-large h12 (text) | **+.305** [+.170, +.428] | **+.279** [+.168, +.400] |
| embeddinggemma-2 768 (text) | +.187 | +.215 |
| gpt2-large (text) | +.171 | +.212 |
| whisper-medium L24 | +.166 | +.210 |
| data2vec-audio-large | +.144 | +.205 |
| hubert-ll60k | +.131 | +.188 |
| wav2vec2-xlsr | +.132 | +.187 |

CIs of the non-t5 rows overlap each other (roughly ±.13). Frequency is about 0, so the match is not a frequency
artefact.

## Reading

- What the blend confuses lines up with t5's text geometry about twice as well as with any speech teacher. The
  speech teachers cannot be told apart.
- The public ranking runs the other way: speech-target families score higher on public than the masc-only blend
  (see [ruler vs public](2026-10-10_ruler_vs_public.md)). Inference, not tested: the speech targets help by
  separating words the MEG already handles in a text-like way, not by copying the speech geometry.
- An RSA match measures shared structure, not usable signal. It does not license a t5-target arm by itself.

## What this does not cover

PanPhon features, w2v-BERT 2.0, whisper-large-v3 and flan-t5 were not cached, so they were not run.
