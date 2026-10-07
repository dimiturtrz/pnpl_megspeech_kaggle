# Single-model levers on the dropout-0.25 recipe: what moved, what didn't (2026-10-06)

Ruler: `ioi --grouping kaggle --partition test` — pnpl official validation (n 1285 Sherlock1 + 1212 TheMoth)
and official test (1347 + 775), macro BAcc@10. Every arm is `--origin sent40-ioi-ema9998-s0 --context.dropout
0.25`, 40 epochs, one seed unless marked. Seed floor of that recipe (ema-cdrop25 s0-s7): **val .6136 sd .0066,
test .6548 sd .019**. Matched control for seed-1 arms: ema-cdrop25-s1 **.6029 / .6590**. Issues:
megspeech-7gf.122 (partitions, ensembles), 7gf.125 (front end), 7gf.126 (vocab weight, anchors), 7gf.127
(neighbour heads, label-free encoders).

## Headline

Nothing cleared the 1.5-point bar on a single model today. Fourteen mechanism arms across the front end, the
objective and the schedule all land inside one seed sd of the floor. Two structural facts came out of it, and
they matter more than any one arm:

1. **Val does not rank seeds.** Spearman(val, test) over the 8 cd25 seeds ≈ −0.07. The top 4 seeds by val
   ensemble **.017 worse** on test than the bottom 4. With an sd of .0066 on val, any one-seed delta under
   ~.013 is not a ranking.
2. **Objectives that read the 50 labels converge to the same model.** The true-class log-prob correlation
   with a matched seed is .97 for every 50-way-label variant, the same as seed vs seed. Only the audio-supervised
   terms decorrelate, and only to ~.955–.96 (see Diversity below).

## Arms (kcut val / test)

| Axis | Arm | Val / test | Read |
|---|---|---|---|
| Front end | site3w3d3o16 (Neuromag triplet tokens, flatten + fuse) | .6157 / .6456 | null |
| | site3w3d3o2 (compress to 2 per site) | .6024 / .6527 | null; log-val +1 pt did not hold |
| | site3o16 + site attention | .5988 / .6245 | worse than its control (−.017 val) |
| | sw8d1r (per-sensor shared filter bank, depth 1) | .6082 / .6317 | null; training-log VAL .656 did not transfer |
| | pyr24 / pyr24g1 (temporal pyramid) | .5941 / .6196, .5974 / .6422 | worse; early overfit (best epoch 10–17) |
| | specaug 20 / 40 / 60 | .5948 / .6334 … .6016 / .6301 | null to worse |
| Objective | anchaud16 / anchtxt16 (frozen 16-d PCA anchors) | killed at .46–.50 | 16-d bottleneck kills it, for audio and text alike |
| | anchaud16tr (trainable 16-d) | .5940 / .6379 | −3 sd val; the bottleneck still costs ~.02 |
| | anchaudfull (frozen full-dim wav2vec2 anchors) | .6168 / .6550 | neutral |
| | sp05vw8 (speech aux .5, in-vocab rows ×8) | .6087 / .6577 | neutral |
| | sp05 / sp10 (speech aux .5 / 1.0) | .6074 / .6564, .6016 / .6565 | neutral |
| | sp10anch (speech 1.0 + audio anchor head) | .6108 / .6517 | neutral |
| | nbr03 (next/prev-word heads .3), s1–s4 | val mean .6033, test .6720/.6894/.6643/.6798 | paired vs matched seeds: test +.009, val −.003 = noise |
| | nbr10 (heads 1.0) | .6029 / .6452 | heavier heads hurt test −.014 |
| | nbr03 + sp05vw8 | .5938 / .6577 | no stacking |
| | nbr03 + drop-path .1 / + anchfull / dropout .35 | .6098 / .6573, .6035 / .6515, .6171 / .6502 | all inside scatter |
| Schedule | nbr03, 60 epochs | .6001 / .6384 | VAL kept rising (.629) while official fell: longer overfits; 40 stays |
| Inference | neighbour-head fusion α .25/.5/1 | val .568/.530/.448 | hurts; the weak heads (~.33 alone) dilute the context head |

## Why the front-end line closed

The post-SSS input has rank ~76 of 306 per run (7gf.68 found it first, rank 74), and the subspace rotates
between runs. A 1x1 spatial mix first is therefore the physically right first operation: quasi-static mixing
with no lag. Imaginary coherency sits at the estimator floor, and the real part is 4–7× larger. A channel-first
(time-first) front end only buys weight sharing and equivariance. It buys no lag modelling, and the arms agree:
triplets, per-sensor and site attention are all null or worse.

The owner's reframing still stands and is untested: triplets as *tokens* feeding token-level site dropout and
attention pooling, which in turn feed masked-site SSL and Broad multi-subject. The pool arm crashes at first
backward ('(*bias): last dimension must be contiguous'). A CPU fix is pending.

## Why the objective line (with labels) closed

Next/prev heads were meant as anticipation, but the 1 s post-onset window spans 3–5 words, so the heads partly
read the neighbour's own evoked response. Their predictions correlate .977 with the matched seed. They teach the
encoder nothing the context head did not already teach.

Anchors fail by bottleneck rather than by source: frozen-16 costs −.08 for audio and text alike, and full
dimension is neutral. Not refuted: anchors as an *auxiliary target* rather than as the head, and neighbour
targets outside the 50 (lexical next-word).

## Diversity and ensembles

From `diversity.py` (Kaggle cut, val):

| Pair | Correlation |
|---|---|
| Seed vs seed | .9735 |
| Same-scheme knobs | .959 |
| Different schemes (all models, 18:01) | .923 |

Against the off-seed controls s0/s2: nbr03 .972/.968 (= a seed); sp05vw8 .960/.954 (real but small).

Best val pairs: s1+sw8d1r .628 and s0+sp05vw8 .623, against seed pairs at .613 (+.01, directional). The k-curve
over cd25 seeds plateaus by k = 4 (test k1 .659 → k4 .666).

The ambiguity decomposition (Krogh & Vedelsby 1995) bounds this: the ensemble gain equals the mean member
disagreement, so at ρ ≈ .97 it is about one point. The gap to the public leader is ~11 points (.765 vs .657).
**The ensemble is a ~1-pt lever; the member has to improve.**

## Public leaderboard: what local predicts

| Deep submission | Local test | Public |
|---|---|---|
| ens-cd25-8 | .6961 | **.6572** (best Deep) |
| ens-cd25-8lf6 (8 seeds + 6 label-free) | .7021 | .6547 |
| ema-cdrop25-s1 | .7013 | .6477 |
| ema-cdrop25-rfall-s1 (refit on all) | — | .6165 |
| ema-cdrop25-s7 | .6649 | .6438 |
| ens-E0 (instrument) | .6940 | .6203 |
| bgmix05-s0 / s1 | .6168 / .6264 | .5274 / .5891 |

The s1/s7 local gap of .036 shrinks to .004 on public: per-seed local-test spread mostly does not transfer. Two
bgmix seeds 0.01 apart locally sit 0.06 apart on public, consistent with the public SE ≈ .023 for one score
plus seed variance.

Broad best: R1 aligned .5879, then ens-cd25-8lf6 aligned .5873, then ens-bgmix05-s0to5 .5837, then ens-cd25-8
aligned .5714. On Broad the label-free members add +.016 public over ens-cd25-8. On Deep they move it by -.003.
Both are within one public SE.

Refit on all data (refit truncates the schedule to the selected epoch) dropped public on both tracks:

- Deep: rfall-s1 .6165 vs s1 .6477.
- Broad: bg90-cd25-rfall-s0 .5410 vs bg90-cd25-s0 .5592.

The cause is not isolated. The refit's compressed schedule and seed-scale scatter both fit.

## Label-free encoders (7gf.127): parity, not a gain

Owner hypothesis (7gf.127): every 50-label objective pulls the encoder into one basin. The proposed test is to
train the encoder from label-free terms alone, then probe it.

- **Stage 1:** `--optim.heads 0`, last epoch, with speech 1.0 plus phonetic / semantic / lexical, or a GPT-2
  in-context text target.
- **Stage 2:** `--optim.stem`, frozen (`encoder_lr 0`) or fine-tuned at 0.1.

Pre-registration:

- Single model .55–.59 val.
- Correlation to seeds .85–.92.
- Worth keeping only if a pair with a seed beats seed pairs by **> .015 val**.

If it fails, check the harness first. The suspects:

- Does stage 1 converge at all without labels?
- Did the stem actually load?
- Does the frozen probe have enough head capacity?

Only after those check out does the failure count against the idea.

**Outcome.** Stage 1 converged, the stem loaded, and the frozen probe reached seed level. Single models landed at
**parity** with the supervised seeds (controls val .603–.613, test .659–.686), well above the .55–.59
pre-registration.

| Stage-1 target | Stage 2 | Val | Test |
|---|---|---|---|
| speech + semantic | frz / lr10 / ft (s1) | .6128 / .6287 / .6189 | .6727 / .6620 / .6791 |
| speech + semantic | lr10 s0 / s2 | .6099 / .6156 | .6586 / .6704 |
| speech + phonetic | frz / lr10 / ft (s1) | .5986 / .6123 / .6197 | .6755 / .6651 / .6611 |
| GPT-2 text, stem A | frz s0 / s1 / s2 | .6317 / .6347 / .6262 | .6774 / .6735 / .6531 |
| GPT-2 text, stem B | frz s2 / s0 | .6077 / .6282 | .6656 / .6731 |

The GPT-2 val gain over the controls (~+.02) came from stem A. A second stage-1 seed (stem B) does not reproduce it.
Labels add roughly nothing to the encoder, but label-free is not better.

Diversity is the asset:

- Correlation to seeds is .935–.948, against .9735 between seeds.
- The best seed + label-free pair reaches .621 val, against a seed-pair mean of .6132. That misses the > .015 gate.
- A greedy-6 ensemble scores val .6306 and test .6759.

GPT-2-target models stay unsubmitted until the owner clears the pretrained-LM prior.

## What this does not rule out

- Bigger data (more subjects/sessions under the rules).
- A fine-tuned MEG-XL (7gf.5.3 diverged on recipe).
- Token-based front end plus site dropout plus SSL.
- Dense acoustic auxiliary on site tokens (7gf.125.1).
- Anchors as an auxiliary target.
- Lexical next-word targets.

These are the levers with a plausible size above 1.5 points. Every knob tried today was not one of them.
