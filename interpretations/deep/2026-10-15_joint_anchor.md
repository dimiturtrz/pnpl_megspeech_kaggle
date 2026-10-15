# Joint anchor target: MEG-reachable teacher directions as the stem's speech target

megspeech-7gf.109.7 · code `megspeech/evaluation/studies/teachers/anchor.py` · 2026-10-15

## Idea

The stem's contrastive targets are 1024-d teacher spaces (wav2vec2 audio latent, t5 word embedding), most of whose
directions MEG cannot predict. Ridge-CCA between a trained encoder's word embeddings and each teacher, fit on
official-train words only, keeps each teacher's 32 strongest canonical target axes, scaled by their canonical
correlation; the blocks are concatenated (64 d) and a teacher missing for a word (wav2vec2 has audio for 99 of 156
runs) is zero-filled. The stem trains its speech term on that instead of wav2vec2.

## Harness bug found on the way (megspeech-e1s)

`Words.embed` carried the reader's within-sentence `position`; per-run target files index by the word's place in
its run. The first anchor build placed 8,213 of 765,256 rows, and the 99y alignment study had paired MEG words with
other words' wav2vec2 latents. Training was never affected (the corpus keys run-level positions).

Fixed reach, held-out train runs, encoder `sent40-ioi-ema9998-s0` (trained on wav2vec2 — home advantage):

| teacher | sig comps per-comp / family | val corr top-1 / 5 / 20 | teacher-probe BAcc@10 |
|---|---|---|---|
| wav2vec2-xlsr-53 (bugged) | 68 / 15 | .776 / .142 / .020 | .221 |
| wav2vec2-xlsr-53 | 95 / 27 | .761 / .486 / .111 | .938 |
| hubert-large-ll60k | 78 / 23 | .743 / .341 / .068 | .993 |
| data2vec-audio-large | 84 / 21 | .711 / .353 / .069 | .991 |
| whisper-medium L16 | 71 / 17 | .724 / .302 / .006 | .977 |
| whisper-medium L24 | 76 / 16 | .730 / .330 / .047 | .993 |
| gpt2-large | 76 / 18 | .634 / .269 / .032 | 1.000 |

## Result — anchor replacing wav2vec2 and t5 (`semantic 0`), seeds 1–2, matched to base `ac05x-s{k}-ft-s{k}`

| seed | VAL best (anchor / base) | local test τ .5 (anchor / base) |
|---|---|---|
| 1 | .6506 / .6387 | .6874 / .7088 |
| 2 | .6496 / .6407 | .6844 / .7131 |

- VAL +1.0, but per-domain it swaps sign between seeds (seed 2 MOCHA −9, Sherlock +2): noise-level.
- Local test −2.5, both seeds below all four base seeds (.709–.717, SD .004), on both Sherlock1 and TheMoth.
- The fine-tune converges faster (+3.5–4 pt VAL at epochs 5–9); the lead closes by epoch ~30.
- As ensemble members: fam8 .7192, fam8 + both anchor fine-tunes .7192, fam8 + two base fine-tunes .7157. Some
  diversity, no gain.

## What it does and does not rule out

The arm changed two things: it swapped wav2vec2 for the anchor AND dropped the full 1024-d t5 term (the anchor keeps
32 t5 directions). Also, the anchor is the s1 encoder's own CCA space, so the stem is pulled toward what an already
trained model reads — a self-distillation that may narrow the representation new stimuli need. Canonical
correlations were in-sample (the encoder trained on these words), so the ρ-weighting barely discriminates for
wav2vec2. Split arm (anchor + `semantic .5`): see below.

## Split arm — anchor as the speech term, t5 kept (`semantic .5`), seeds 1–2

| seed | VAL best (ancsem / base) | local test τ .5 (ancsem / anchor-only / base) |
|---|---|---|
| 1 | .6403 / .6387 | .6995 / .6874 / .7088 |
| 2 | .6490 / .6407 | .6920 / .6844 / .7131 |

Restoring the full t5 term recovers about half the loss (−1.5 vs −2.5 pt); both seeds are still below all four base
seeds, again on Sherlock1 and TheMoth alike. The part that remains belongs to replacing wav2vec2 with its
encoder-reachable 32 axes: VAL cannot see it (+0.5–0.8), the stimuli the encoder never read can.

Reading: an anchor fit on a trained encoder's own CCA space compresses the speech target to what one model already
decodes, and that narrowing costs transfer. The gpt2-large third block (`anchor-cca3`) was gated on this arm showing a
local-test gain; it did not, so it was not run. Not ruled out: an anchor added *beside* the full wav2vec2 target, or
one fit on held-out-run canonical correlations (out-of-sample ρ) — neither has a mechanism that predicts ≥1.5 pt, so
both stay unfiled.

## Beside arm — base recipe kept, anchor added as a second speech term (`reachable .5`), seeds 1–2

megspeech-7gf.170. Pre-registered bar: 2-seed mean local test ≥ .726 (base .711 + 1.5).

| seed | VAL best (ancbes / base) | local test τ .5 (ancbes / base) |
|---|---|---|
| 1 | .6488 / .6387 | .7067 / .7088 |
| 2 | .6527 / .6407 | .6876 / .7131 |

Mean local test .6972, −1.4 against base; seed 2 loses on TheMoth (.638). As a ninth fam8 member ancbes-s1 lowers
the blend .7192 → .7136.

Across all three arms (replace, split, beside) the anchor raises VAL by 0.5–1.2 and lowers the local test by 0.2–2.6.
Keeping the full wav2vec2 target does not protect transfer: any pull toward one trained encoder's CCA axes helps on
stimuli that encoder read and costs on stimuli it did not. Closed; the out-of-sample-ρ variant shares the mechanism
and is not filed.
