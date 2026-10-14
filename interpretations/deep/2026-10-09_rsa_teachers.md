# Teacher RSA on the H/D/M members: no teacher beats xlsr; a teacher aux did not pull the geometry (megspeech-7gf.141.1)

Follows [2026-10-08_teacher_rsa.md](2026-10-08_teacher_rsa.md), which ran the same study on plain-CE and syn05 runs.
Study: `megspeech/evaluation/studies/teachers/rsa.py` (dev `e9e6545`). Issue: megspeech-7gf.141.1.

## Question
For each candidate aux teacher, does its geometry over the 50 target words match the geometry of our trained MEG word
head? The ranking is meant to gate which teachers get a training run.

## Method
- **MEG matrix.** The model's mean tau-adjusted softmax score for each true word over the 50 candidates, symmetrised.
  It is read on the 2545 scored unseen-span VAL words, and all 50 words are present.
- **Teacher matrix.** The cosine of type-level word vectors: each word's per-token targets averaged over the training
  words. It comes in two forms: RAW, and CENTRED (the mean of the 50 type vectors subtracted first).
- **RSA.** Spearman correlation over the 1225 off-diagonal pairs.
- **CI.** The 2.5-97.5 percentile interval of 1000 seeded word resamples.
- **Null.** A Mantel permutation: 1000 shuffles of the teacher's word labels with the MEG matrix held fixed. The p is
  one-sided, (1 + #null >= rho) / 1001, so its floor is 0.001.

What the MEG matrix is NOT:
- It is not a confusion-count matrix.
- It is not a 1 - cosine RDM of MEG embeddings. It is mean softmax mass, so Spearman over it equals Spearman over
  "1 - similarity" RDMs only up to rank.

What the teacher vectors and CIs are NOT:
- The teacher vectors are not contextual. Context is averaged away by the type mean.
- The CIs cover word sampling only, not model-seed variance.
- The raw speech type means are near-collinear (min pairwise cosine distance ~0.000-0.001), so their raw rho mostly
  ranks tiny residual differences. That is why the centred column exists.

Members, one per Deep ensemble kind:

| member | run | aux at stem training |
|---|---|---|
| H | `ema-cd25-xsp10sem05-hub-s1-ft-s1` | speech 1.0 on hubert-large-ll60k + semantic 0.5 |
| D | `ema-cd25-xsp10sem05-d2v-s1-ft-s1` | speech 1.0 on data2vec-audio-large + semantic 0.5 |
| M | `ema-cd25-masc27w-s1` | every aux weight 0 |

M is the clean reference. The full H/D/M table is in the 7gf.141.1 notes.

Command: `CUDA_VISIBLE_DEVICES=-1 MEGSPEECH_DATA=D:/data/neural uv run --no-sync python -m megspeech.evaluation.studies.teachers.rsa`

## M table
| teacher | rho raw [95% CI] | p raw | rho centred [95% CI] | p centred |
|---|---|---|---|---|
| one-hot | 0 (constant) | 1.000 | 0 | 1.000 |
| t5-large h12 | +0.309 [+0.176, +0.429] | 0.001 | +0.285 [+0.172, +0.406] | 0.001 |
| embeddinggemma-2 768 | +0.186 [+0.033, +0.335] | 0.002 | +0.223 [+0.111, +0.347] | 0.001 |
| embeddinggemma-2 256 | +0.174 [+0.025, +0.320] | 0.002 | +0.178 [+0.060, +0.299] | 0.001 |
| embeddinggemma-2 128 | +0.160 [-0.000, +0.312] | 0.002 | +0.164 [+0.052, +0.279] | 0.001 |
| gpt2-large (owner-gated target) | +0.171 [+0.032, +0.309] | 0.001 | +0.213 [+0.083, +0.338] | 0.001 |
| whisper-medium L8 | +0.151 [+0.003, +0.289] | 0.002 | +0.187 [+0.058, +0.307] | 0.001 |
| whisper-medium L12 | +0.145 [-0.000, +0.281] | 0.003 | +0.186 [+0.059, +0.304] | 0.001 |
| whisper-medium L16 | +0.149 [+0.007, +0.286] | 0.002 | +0.190 [+0.066, +0.304] | 0.001 |
| whisper-medium L20 | +0.151 [+0.014, +0.285] | 0.001 | +0.200 [+0.073, +0.315] | 0.001 |
| whisper-medium L24 | +0.163 [+0.026, +0.298] | 0.001 | +0.211 [+0.083, +0.334] | 0.001 |
| data2vec-audio-large | +0.141 [-0.003, +0.277] | 0.003 | +0.206 [+0.078, +0.323] | 0.001 |
| wav2vec2-large-xlsr-53 | +0.131 [-0.012, +0.263] | 0.005 | +0.189 [+0.064, +0.309] | 0.001 |
| hubert-large-ll60k | +0.128 [-0.016, +0.258] | 0.006 | +0.189 [+0.068, +0.307] | 0.001 |

## Reading
- **The pre-registered rule is not met.** The rule, applied to M only, was: a teacher is better if its rho is above
  xlsr's with non-overlapping CIs. No teacher meets it, raw or centred. t5 comes closest: its raw lower bound is
  +0.176, against xlsr's upper bound of +0.263.
- **Every non-one-hot geometry is above the permutation null.** Every teacher has p <= 0.006 on M (<= 0.014 on any
  member). Each shares a weak but real part of what the MEG head confuses.
- **Circularity: a teacher aux did not pull the geometry toward that teacher.** The H stem was trained with HuBERT as
  its speech target, yet H vs hubert is +0.121 raw / +0.182 centred, against M's +0.128 / +0.189. D vs data2vec is
  +0.138 / +0.203, against M's +0.141 / +0.206. The aux-trained member is no closer to its own teacher than the
  no-aux member is, and is not closer to its own teacher than to the other one (H vs data2vec +0.133 / +0.199;
  D vs hubert +0.125 / +0.185). Either the speech aux shapes the encoder without reaching the 50-word head geometry, or the type
  mean erases what it shaped. This study cannot tell the two apart.
- **Among speech teachers,** whisper-medium L24 has the highest centred rho on M (+0.211 vs xlsr's +0.189). That is
  directional only, since the CIs overlap almost entirely. It is the pick for the next aux try on other grounds:
  apache-2.0, cached, and no member kind trained on it yet. The seeds are queued as bq67 (7gf.141.8).

## On the 36-member blend (2026-10-10)
`python -m megspeech.evaluation.studies.teachers.blend m11wd` (eef38ae) uses the mean softmax of all 36 m11wd members, with
teacher vectors fitted on subject-0 training words only. Every rho moves by at most .006 from the M table: t5 +.305 /
+.279, xlsr +.132 / +.187, whisper L24 +.166 / +.210, word frequency (a log-count gap, as a control) +.055
[−.090, +.217]. Averaging 36 members of mixed teachers leaves the geometry where a single no-aux member has it, which
agrees with the circularity reading above. The frequency control rules out a frequency artefact behind t5's lead.

## What this does NOT rule out
- **A real gap between teachers.** The CIs are marginal per teacher over the same word resamples, so they are strongly
  correlated. A paired bootstrap of rho differences (for example t5 minus xlsr) was not run, and it could separate
  teachers that the marginal CIs cannot.
- **A decoding gain.** RSA geometry is not decoding gain. A teacher whose 50-word geometry matches MEG may add nothing
  to top-10 accuracy, and one that matches poorly may still help as an aux, by shaping the encoder on the many words
  outside the 50 or through its contextual per-token states. Only a training run measures that.
- **Seed variance.** One member per kind was measured, so the CIs carry no model-seed variance.
