# AXES — every lever this project has tried, its verdict, and where it is written down

## Why this file exists

This file is an INDEX, not a second home for numbers. A row says what was tried, the best delta against
its own control, and the recorded mechanism, then points to the bead or interpretation that holds the
evidence. **Re-open rule:** a CLOSED or SATURATED axis is re-opened only by answering the mechanism in
its WHY cell (a new input, a fixed harness, an untested form). Quoting a number again does not re-open it.
"no why recorded" means the source gives no mechanism. Do not invent one; record it when you find it.

## Rulers — never compare numbers across rulers

| tag | ruler | notes |
|---|---|---|
| **K** | kcut val/test: `ioi --grouping kaggle`, val = carved 10%, test = pnpl official val+test. Since c731c56 each word is bounded at its piece's end + 1 s (before: sentence end, which leaked MEG to windows > 1 s) | the current ruler; ctrl ema-cdrop25 s0-s2 mean .6105/.6717 on the piece-bounded version. Older numbers below are pre-c731c56. Control **ema-cdrop25-s1 .6029/.6590**. Seed floor over 8 cd25 seeds: val .6136 sd .0066, test .6548 sd .019, Spearman(val,test) ≈ −0.07 |
| **LT** | local test, sentence cut (before kcut) | inflated by long spans; IOI's gain lives here (7gf.105) |
| **OV** | old official val, sentence-span novel macro n=2512 (before 2026-10-05) | local-vs-public Spearman −0.14..−0.29, so its nulls and gains are UNRANKED. Bases: sent40-ioi-s0 0.6205, sent40-ioi-ema9998-s0 0.6453, base 0.6445 |
| **GAP** | old n=1479 ruler | obsolete |
| **BR** | Broad local ruler n=25322 | Broad track only |
| **PUB** | Kaggle public LB, ~434 rows | SE ≈ .023 for one score; a paired same-row delta is tighter |

A result needs **≥ .015 absolute** on its ruler. Under the bar the verdict is DIRECTIONAL at most, and a
single seed on K has to be read against the floor sd above.

## Status vocabulary

| verdict | meaning |
|---|---|
| **KEPT** | in the recipe |
| **CLOSED-null** | tried, no effect beyond noise; the WHY says which use is closed |
| **DIRECTIONAL** | sign is consistent but under the bar; keep if free, never plan around it |
| **OPEN** | not yet measured cleanly (missing metrics, harness bug, pending seeds) |
| **UNEXPLAINED** | measured effect with no mechanism recorded |
| **SATURATED** | knob family with ≥3 null variants; re-open only via a new mechanism |

---

## A. Architecture / encoder

**The capacity lever is signal-bound:** epistemic uncertainty is ~0.056 of ~2.6 nats (7gf.116, sou). Size
knobs are SATURATED.

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| context width 512 / 384 | 4te.1, 7gf.46 | 0.557 (−0.061 vs s7) / 0.6269 vs 0.6205 (OV) | CLOSED-null | capacity overfits | 4te.1, 7gf.46 |
| hidden 512 | 4te.6 | 0.606 (OV) | CLOSED-null | no why recorded | 4te.6 |
| depth 16L / latent 4 | 7gf.51.1, 7gf.67 | 0.6234 / 0.624 (OV) | CLOSED-null | overfits | 7gf.51.1, 7gf.67 |
| **width/depth/capacity: SATURATED** | above + sou | — | SATURATED | loss is signal-bound (epistemic ~2%) | sou, 7gf.116 |
| conformer mixing / encoder | 4te.7, qi7.3, 5e6 convconf | 0.595 / 0.594/0.586 (OV); convconf .641 peak = band (LT-era val) | CLOSED-null | spans too short; on cd25 it ties at peak, then overfits (gap .29 @e37) | 5e6, interpretations/deep/2026-10-05_meg_encoders.md |
| SSM | 7gf.27 | 0.614 (OV) | CLOSED-null | no why recorded | 7gf.27 |
| multiscale seg4/k357 | 7gf.26 | 0.607/0.608 (OV) | CLOSED-null | no why recorded | 7gf.26 |
| attention pool | 7gf.45 | 0.6146 vs 0.6205 (OV) | CLOSED-null | no why recorded | 7gf.45 |
| merger | 7gf.55 | +0.013 no-EMA; 0.6472 vs 0.6453 with EMA (OV) | CLOSED-null | gain is variance reduction, which EMA already supplies | 7gf.55 |
| GLU | 7gf.29.7 | 0.6338 (OV) | CLOSED-null | no why recorded | 7gf.29.7 |
| final LayerNorm | h1u.6 | 4-seed ens 0.6361 vs 0.6492 (OV) | CLOSED-null | no why recorded | h1u.6 |
| d'Ascoli faithful replication | 7gf.51 | 0.6408 (OV); own-metric 0.695 ≈ 0.641 ours | CLOSED (instrument) | their metric reads ~+0.05 over ours | 7gf.51, 7gf.114 |
| port composed | 7gf.29.6 | 0.620 (OV) | CLOSED-null | no why recorded | 7gf.29.6 |
| joint experts | 7gf.90 | +0.6 pt (OV) | DIRECTIONAL | no why recorded | 7gf.90 |
| MEG-XL frozen | 7gf.5.x, 7gf.62 | 0.358/0.278/0.366 (OV) | CLOSED-null (frozen use) | frozen foundation features carry no word content here | meg_encoders.md |
| MEG-XL LoRA / megxlsp9 | 7gf.5.x, megxlsp9 | LoRA diverged; megxlsp9 K metrics missing | OPEN | recipe failure, not a refutation | 7gf.5.x |
| channel-first sensorwise (sw16, sw8d1/sw8d1r) | 7gf.125 | sw16 killed at 23.5 min/epoch; sw8d1r .6082/.6317 (K) | CLOSED-null (this use) | sw16: tiny-channel convs GPU-inefficient. sw8d1r: val lift did not transfer to test. sw8d1 metrics missing | 7gf.125 |
| site set-pool front end (pyr24, site3*) | 7gf.125 | best site3w3d3o16 .6157/.6456; d0m .6083/.6358; d25m .5908/.6254; pyr24 .5941/.6196 (K) | CLOSED-null (this use) | set-pool halves the neural share (.049 vs ~.10-.12), so weight moves onto context; site dropout adds cost. site3o16p4d50: crash, never run | 7gf.125 |
| feature-level fusion of frozen stems | 7gf.131 fus3f / fus3ft | .5961/.6640, .6036/.6699 vs 2-member logit ens .6164/.6742 (K) | CLOSED-null (frozen, fused_lr 0) | each branch is used (silencing one costs −.04..−.07 val), but frozen features plus a new head under-train vs each stem's own tuned head. Untested: fine-tuned branches, learned logit weights | 7gf.131 |
| trainable-encoder capacity sweep | sou | not run | OPEN (pre-reg null) | instrument only | sou |

## B. Front end / input geometry / denoising / normalisation

**Closed by physics:** post-SSS rank is ~76, mixing is quasi-static, and imaginary coherency sits at the
floor. A fixed linear sensor mix adds nothing the net cannot learn (x30). Session normalisation hurts Deep
in every form tried.

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| SSS subspace projection | 7gf.68, 7gf.68.1 | lossless; k32/k57 −2.7/−1.3 pt (OV) | CLOSED | data lives in ~76 dims; truncating below that discards signal | 7gf.68 |
| sample rate 125 Hz / decimation | qi7.1, 4te.5 | 0.607 in band (OV); 12-25 Hz carries ~0.006 | CLOSED-null | the band above 25 Hz carries ~nothing | qi7.1, 4te.5 |
| high-gamma band (LDA / envelope) | 7gf.80, 7gf.69, x30 | 0.2315 vs 0.3247 low band; concatenation −2.1 pt (OV) | CLOSED-null (top-power projection); x30 OPEN | top-power high-band components are muscle/noise. xDAWN-on-envelope (word-locked SNR) untested | 7gf.80, x30 |
| ICA artefact removal | 7gf.52 | +0.010, CI −0.017..+0.037 (OV) | CLOSED-null | artefacts do not limit the score | 7gf.52 |
| Ledoit-Wolf whitening | 7gf.70 | 0.6251 vs 0.6453 (OV) | CLOSED-null | LW amplifies tail dims | 7gf.70 |
| noise whitening | 7gf.120 | −.004/+.001/−.002 (bal val / bal test / off test) | CLOSED-null | no why recorded | 7gf.120 |
| per-window instance norm | 4te.10, 7gf.25.14 | LT 0.639 = champion ens; PUB 0.613 < 0.637 | CLOSED-null | local gain did not transfer; no why recorded | 7gf.25.14 |
| reference preprocessing | 7gf.25.12 | 0.612 (OV) | CLOSED-null | no why recorded | 7gf.25.12 |
| holdout-fitted scaler | 7gf.25.4 | PUB 0.505 vs 0.567 | CLOSED-null | no why recorded | 7gf.25.4 |
| test-time per-session rescale | 7gf.14 | no-op | CLOSED-null | per-recording robust scaling already does it | 7gf.14 |
| FIR edge handling | 7gf.59 | null | CLOSED-null | no why recorded | 7gf.59 |
| Euclidean alignment (Deep, train or eval) | 7gf.56, 7gf.61 | train −0.021; eval 0.644→0.588 (OV) | CLOSED-null (Deep) | within-subject session drift is small; whitening destroys the covariance sub-0's model reads | interpretations/broad/2026-10-03_euclidean_alignment.md |
| AdaBN | 7gf.61.1 | −0.064 single / −0.055 ens24 | CLOSED-null | same as EA: session shift is not the Deep bottleneck | 7gf.61.1 |
| transductive pseudo-labels on holdout | 7gf.61 (c) | not run | OPEN (rules-gated, ~10% odds) | — | 7gf.61 |
| MASC 50 Hz notch bug | 7gf.133 | not run | OPEN (harness suspect for every MASC null) | converter notches 60/120 Hz but MASC mains is 50 Hz | 7gf.133 |

## C. Objective / heads / auxiliary targets

**7gf.31 epic: the loss/head axis is exhausted on this backbone.** 50-label objectives correlate .97 with a
matched seed; audio-supervised terms reach .955-.96, label-free stems .935-.948 (single_model_levers).
Pretrained teachers expose only ~11-24 predictable canonical components out of 1024 target dims, and every
teacher probe ceiling (0.20-0.26) sits below the MEG encoder's own 0.307 (99y).

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| SigLIP / contrastive | 4te.3, 7gf.37 | 0.508 GAP / 0.613 (OV) | CLOSED-null | target dims far exceed the predictable components (99y) | 7gf.37, 99y |
| semantic (t5) aux / OOV / sem80 | 7gf.25.3, 7gf.31.1 | 0.607 / 0.6141 (OV) | CLOSED-null | same as above | 99y |
| semctx tap | 7gf.51.2 | +0.017 no-EMA; 0.6495 with EMA (OV) | CLOSED-null | variance reduction that EMA already supplies | 7gf.51.2 |
| lexical aux / pretrain→ft | 7gf.40, 7gf.41 | 0.602 / 0.609 (OV) | CLOSED-null | teacher probe ceiling < MEG's own (99y) | 99y |
| phoneme aux | 7gf.25.7, qi7.2 | 0.617 / 0.607 (OV) | CLOSED-null | no why recorded (phoneme accuracy never logged) | qi7.2 |
| wav2vec2 speech teacher | 7gf.54 | null at w 1.0/0.3/centred (OV) | CLOSED-null | Sherlock-only audio bends the encoder toward one narrator | 7gf.54 |
| GPT-2 dense target | 7gf.101 | 0.6383 vs 0.6445 (OV) | CLOSED-null | top components are position/timing directions, not word content (99y, caveat) | 7gf.101, 99y |
| reduced-rank teacher targets (k 16-64) | 99y | not run | OPEN | 99y's canonical profile predicts k 16-64 | 99y |
| speech aux on cd25 (sp05, sp10, sp05vw8) | 7gf.126 | sp05 .6074/.6564; sp10 .6016/.6565, .6134/.6504; sp05vw8 .6087/.6577 (K) | CLOSED-null | inside the seed floor; no why recorded | 7gf.126 |
| **aux-weight / vocab_weight knobs: SATURATED** | sp05, sp10×2, sp05vw8, sp10anch | all inside the floor (K) | SATURATED | 50-label objectives move the model ~.97-correlated | 7gf.126, single_model_levers.md |
| frozen-anchor cosine head (audio/text) | 7gf.126 | frozen16 −.08; trainable16 .5940/.6379; full .6168/.6550 (K); anchtxt16 killed | CLOSED-null (head form) | a 16-d PCA bottleneck kills it; at full dim it is neutral. Untested: anchors as an aux target | 7gf.126 |
| anchors + speech (sp05anch, sp10anch, sptxt10) | 7gf.126 | sp05anch .6286/.6330; sp10anch .6108/.6517; sptxt10 .6297/.6628 (K) | CLOSED-null | val and test disagree in sign; no why recorded | 7gf.126 |
| label-free SSL stems, frozen / ft (xsp10ph05, xsp10sem05, xall120, xlex05sp10) | 7gf.109.x | range val .5986-.6306, test .6217-.6985 (K); means at parity | CLOSED-null (parity) | features are .935-.948 correlated with seed; parity per seed | single_model_levers.md |
| GPT-2 text stem (xtxt10) | 7gf.109.x, 7gf.127 notes | stem A frz .6317/.6774, .6347/.6735, .6262/.6531; B .6282/.6731, .6077/.6656; C .6241/.6604, .6290/.6420 (K) | DIRECTIONAL | +~.015-.02 val over controls but test does not follow; stem A's gain did not reproduce in B | 7gf.127 notes |
| masked / JEPA SSL pretrain | 7gf.109.4 | ssl20 .6377 vs .6361 val, LT .7046 vs .7013 | CLOSED-null (v1); v2 killed | v1: less late overfit only. v2 probe .198 is below random-init .208, so its features are not word-discriminative | 7gf.109.4 |
| multi-subject SSL (MASC cross-subject) | ct5 | not run | OPEN | subjects 1-32 share no legal stimulus with subj 0; feasible within MASC only, for Broad | ct5 |
| next / prev neighbour heads (nbr03, nbr10) | 7gf.127 | nbr03 s1-s4 val .5941-.6108, test .6643-.6894; nbr10 .6029/.6452 (K) | DIRECTIONAL (test only) | a 1 s window spans 3-5 words, so the heads read the neighbour's evoked response (corr .977). Val stays inside the floor | 7gf.127 |
| window 1.5 s (w15) | 7gf, w15-s1/s2 | +.009 val / +.006 test (K, piece-bounded) | CLOSED this use | the old +.018/+.037 was MEG past the piece end; bounded, it sits under the bar | memory w15-not-null |
| spec-cut training (1..9-word pieces) | bpz, ema-cd25-spec-s1 | .6217/.6644 (K) vs ctrl .6105/.6717 | NULL (s2 pending) | last-in-piece BAcc .44-.45 unchanged vs ctrl: the edge deficit is information, not a sentence-final prior | bpz |
| neighbour-head logit fusion at test | 7gf.127 | α .25/.5/1 → val .568/.530/.448 | CLOSED-null | it hurts; no why recorded beyond corr .977 | 7gf.127 |
| nbr03 variants (anchfull, dp10, e60, sp05vw8, cd35) | 7gf.127 | .6035/.6515, .6098/.6573, .6001/.6384, .5938/.6577, .6171/.6502 (K) | SATURATED | e60 overfits (40 epochs stays); the rest are inside the floor | 7gf.127 |
| dense acoustic envelope aux (ac03) | 7gf.125.1 | .6146/.6681 = +.012/+.009 (K) | DIRECTIONAL | mechanism-fired not verified: no VAL envelope file, so r is unmeasured | 7gf.125.1 |
| POS factor / in-context POS | 7gf.95/96/97, cn2 | +0.009 then noise; +.011 (OV) | DIRECTIONAL; cd25 rerun OPEN | no why recorded | cn2, where_we_are |
| context mask | 7gf.129 | not run | OPEN (P3, ~5%) | the word head already gives the encoder a MEG-only gradient | 7gf.129 |
| class weights / narrator down-weight / repeat_power | 7gf.23, 7gf.47, 7gf.13 | 0.603; −0.021/−0.043; spread 0.03 (OV) | CLOSED-null | narrator down-weight leaves fewer distinct windows, so it overfits | 7gf.47 |
| read-speech oversample | 7gf.11 | 0.446 vs 0.525 (GAP) | CLOSED-null | memorises TIMIT | 7gf.11 |
| adversarial timing removal / advctx / ptime PoE | 7gf.98, 7gf.99, 7gf.100 | .6382; collapsed/.6127; +0.010 (0.6545) (OV) | CLOSED-null; PoE DIRECTIONAL | no why recorded | 7gf.98-100 |
| fp32 CE | h1u.16 | −0.008 (OV) | CLOSED-null | no why recorded | h1u.16 |
| peer residual boosting | 6ow | +0.007 vs base, +0.001 vs MLP control (OV) | CLOSED-null | recalibration explains it | 6ow |

## D. Data / subjects / partitions

**Data scaling is the one lever measured as not saturated:** ~+3 pt per doubling (7gf.104). Subject-0
labelled data is fixed by the rules: subjects 1-32 hold only ses-11/12; Broad may train on them (5f3, decided 2026-10-07), Deep keeps them out (our choice). Flag `--data.adapt 11|12` (496ce22).

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| more subject-0 data (scaling curve) | 7gf.104 | ~+3 pt / doubling (bal val/test, off test) | OPEN (lever) | not saturated; no legal new subject-0 data | 7gf.104 |
| MASC multi-subject (27 subj) | 7gf.57, masc27w | every Deep use null-to-negative; masc27w .6252/.6606 (K) | CLOSED-null (Deep) | earlier masc27ft wins were best-epoch bias; 7gf.133 notch bug is an open harness suspect | 7gf.57, selection_bias.md |
| subj-33 curriculum | 7gf.15.1 | 0.602/0.609 vs 0.619 (OV) | CLOSED-null | matches LibriBrain Fig2 | 7gf.15.1 |
| Broad warm-start fine-tune | 4te.8 | never beats epoch 0 | CLOSED-null (high lr) | OneCycle re-heat; low lr untested | 4te.8 |
| Sherlock-only training | 4te.4 | 0.30 vs 0.51 (GAP) | CLOSED-null | no why recorded | 4te.4 |
| external data | 7gf.44 | priced ~0 | CLOSED | no why recorded | 7gf.44 |
| test-time cross-subject pooling | 7gf.15 | — | BARRED | organiser: cross-track mapping = disqualification (docs/competition/announcements.md) | 7gf.15 |
| k-fold cross-fit | 7gf.85 | −3.2 pt | UNEXPLAINED | no why recorded | 7gf.85 |
| bagging | 7gf.29.2 | null | CLOSED-null | 20% less data costs more than the diversity gains | 7gf.29.2 |
| refit on val (+1-2% data) | 7gf.25.2, 7gf.65, rfall-s1 | ~+0.003; rfall-s1 PUB .6165 vs .6477 | UNEXPLAINED | refit is negative on both tracks; cause not isolated | 7gf.65, where_we_are |
| span cut for training (spec / Kaggle-stat) | bpz | spec-s1 .6217/.6635 vs .6029/.6590 (K); -hl +2.6 | DIRECTIONAL (single seed) | trains on the test's short-piece axis; val and test disagree in size | bpz |
| sampler (visibility / priority power) | 7gf.124 | BR v05 .5845 vs v05p05 .5497 / pm05 .5439 | CLOSED-null (priority); visibility DIRECTIONAL | priority power hurts Broad with both signs | 7gf.124 |
| memorisation locator | 7gf.111, 7gf.128 | fit-val gap is mostly MEG memorisation | instrument | — | 7gf.128 |
| run reproducibility | 7gf.127 (s0r, s3r) | s0r .6185/.6686 vs .6127/.6705; s3r .6034/.6865 vs .6209/.6631 (K) | CLOSED (no drift) | run-to-run nondeterminism, inside the seed sd | 7gf.127 |

## E. Decode / context prior / timing

**About 80% of the score is the context/text prior; MEG adds about .11.** External-LM channels are priced
at <10% odds, because the 50 classes are function words and the test neighbours are mostly OOV (lvj).

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| sentence spans (context head) | 7gf.7, 7gf.25.9 | the big win; PUB .569 | KEPT | in-span word order is the dominant signal | 7gf.7 |
| context dropout .25 | cd25 | +.017 val, +.029 LT paired (bg90) | KEPT | regularises the context head | 7gf.112, where_we_are |
| order clock vs onset clock | 4te.2, 4te.9 | tie; 0.609 vs 0.605 (OV) | KEPT (order) | holdout-matched clock | 4te.2 |
| IOI timing input | 7gf.42, 7gf.105, 7gf.110 | LT +3.6; K test .6590 vs .6575 (+0.15), val −0.3 | KEPT (harmless), not a public lever | the gain lives in long 7-9 word spans and >0.92 s gaps that Kaggle spans lack (faulty-TEST axis); jitter training removes it, since it is sub-20 ms timing | 7gf.105, 7gf.110 |
| timing shortcut (MEG zeroed) | 7gf.71 | 0.521 vs 0.6445 (OV) | instrument | context alone carries most of the score | 7gf.71 |
| LM prior fusion / LM oracle | 7gf.6, 7gf.53 | top-1 0.146 < 0.5 gate | CLOSED-null | 50 classes are function words; neighbours are OOV | 7gf.53 |
| BERT MLM fusion | 7gf.60, 7gf.60.1 | monotone harmful | CLOSED-null | same external-text channel | 7gf.60.1 |
| causal LM + beam / oracle stage-1 bound | lvj, 7gf.61, lmbound.py | m11 VAL 2-fold, GPT-2-large pmi, true in-vocab neighbours: backoff +.044 [+.022,+.068], +2 right words +.091; drop +.011 / +.040; full-text oracle +.170 / +.254; local test same sign | OPEN (P2) — oracle bound clears +.03; realistic (predicted-neighbour) version next | bound uses TRUE neighbour words: an upper bound, not a deliverable | lvj |
| Sinkhorn balancing | 7gf.29.3 | +0.008 (OV); PUB +0.002 | DIRECTIONAL | no why recorded | 7gf.29.3 |
| per-member logit adjustment | 7gf.72 | — | CLOSED | vacuous by algebra | 7gf.72 |
| TTA shifts | 7gf.39 | −0.004 (OV) | CLOSED-null | onsets precise to < 20 ms | 7gf.39 |
| lone rows → word head | 7gf.58 | fix landed | KEPT (PUB blind) | — | 7gf.58 |
| holdout-length ruler | 7gf.78, 7gf.79 | −4.5..−5.4 pt sentence→kaggle | instrument (became K) | span-length axis | 7gf.78 |

## F. Ensembling / distillation

**Ensemble bound:** seed-to-seed correlation is .97, so ensembling is worth ~1 pt (Krogh-Vedelsby). Three
members reach the plateau (7gf.122).

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| EMA .9998 | 7gf.48 (7gf.19 pre-IOI null) | single +0.020 mean over 4 seeds; ens null (OV) | KEPT | variance reduction inside one model | 7gf.48 |
| seed ensemble | 7gf.20, ens-cd25-8 | PUB 0.637 → .6572 | KEPT | ~1 pt bound at corr .97 | 7gf.20 |
| pool growth / kind mixing | 7gf.122 | cd8lf16 LT-kcut .6471/.7182, PUB .66383 (best); all37 .6453/.7030; k3 test mean .6972, k6 .7021 | SATURATED | 3 models reach the plateau; mixing kinds vs seeds-only +.003 = noise; .7182 is a favourable draw | 7gf.122 |
| seed selection / greedy-on-val | 7gf.29.4, 7gf.122 | noise; greedy worse than uniform | CLOSED-null | selection fits val noise | 7gf.29.4 |
| mixed sent80 / 80-epoch members | 7gf.29.5, 7gf.49 | +0.001 / +0.007 (OV) | CLOSED-null | no why recorded | 7gf.49 |
| born-again KD | 7gf.63, 7gf.64, 7gf.66, 7gf.83 | singles +0.010, ens +0.0011; kd-s0 PUB 0.6050 vs 0.6178 | SATURATED | students copy teacher errors; unnormalised teacher | 7gf.66 |
| cross-fitted KD (kdt4) | 7gf.130 | kdt4-f0 K metrics missing | OPEN | — | 7gf.130 |
| frozen-stem fusion vs logit ensemble | 7gf.131 | see A | CLOSED-null | — | 7gf.131 |
| label-free members on Broad vs Deep | 7gf.122 | +.016 PUB on Broad, −.003 on Deep | DIRECTIONAL (Broad) | no why recorded | where_we_are |

## G. Regularisers / augmentation / training knobs

**Composition test:** combo3 = syn15 + specaug40 + dp10, each directionally above the control on K test
(+.012/+.015/+.014); combo3 lands at .6133/.6542. **The knobs do not add.** The MEG-augment family is
SATURATED.

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| specaug (20/40/60, deep9-specaug40) | 5e6 | specaug20 .5948/.6334; specaug40-s2 .6075/.6497; specaug60 .6016/.6301; deep9-sa40 .6136/.6511 (K); s3/s4 missing | SATURATED | no why recorded | 5e6 |
| synthetic mixing syn05 / syn15 | 5e6 | syn05 .6141/.6851, .6106/.6640; syn15 .6126/.6706 (K) | DIRECTIONAL (test) | no why recorded | 5e6 |
| combo3 composition | 5e6 | .6133/.6542 (K) | CLOSED-null | the knobs are not independent; no deeper why recorded | 5e6 |
| window / dropout grid w02d00, w02d02, w04d04 | 5e6 / 7gf.117 | .5568/.5800, .5543/.5993, .5568/.5911 (K) | CLOSED-null | the source does not decode the names; ≈ −.05 on both splits | einstein-check memory |
| w15 | 7gf.117 (likely the wm15 rerun without patience; naming not confirmed) | .6206/.6955 = +.018/+.037 vs ctrl; +.007/+.041 vs floor mean (pre-c731c56 K) | SUPERSEDED | the gain was MEG past the piece end; see the piece-bounded w15 row | 7gf.117 |
| word mask (wm15/wm30, pat8) | 7gf.117 | truncated by patience 8 | OPEN | harness bug, not a refutation | 7gf.117 |
| wd10, win03 | 5e6 | .6056/.6653, .6125/.6563 (K) | CLOSED-null | no why recorded | 5e6 |
| amask / noise / spatial dropout / aspat / adrop | blq, 7gf.91 | amask5 LT −.025; spatial dropout −1.6 pt (OV) | SATURATED (blq never validly run) | no why recorded | blq |
| drop-path .1 / dropout .35 | 5e6 dp10, cd35 | LT .6850 vs .7013 (−0.016); cd35-nbr03 .6171/.6502 (K) | CLOSED-null | no why recorded | 5e6 |
| trial averaging in training (avg2/4/8) | 5e6 arm 3 | avg2p50 .6049/.6372; avg4p50 .5909/.6448; avg8p100 .448/.477 (K) | CLOSED-null (this recipe) | memorisation is not cured by cleaner trials; unjittered partners are a harness suspect. Untested: averaging as an SSL view | 5e6 |
| onset jitter (data aug) | 7gf.18, 7gf.110 | j1 0.617, j2 −0.054 (OV) | CLOSED-null | onsets are precise to < 20 ms | 7gf.18 |
| light aug / stack | cho, qi7.4 | 0.602 / 0.608 (OV) | CLOSED-null | no why recorded | cho, qi7.4 |
| reference optimiser / batch 256 | 7gf.32.1, 7gf.36 | −0.045; no gain (OV) | CLOSED-null | no why recorded | 7gf.32.1 |
| window 2 s / lead 0.1 s | 7gf.25.13, 7gf.38 | 0.606 / 0.609 vs 0.617 (OV) | CLOSED-null | no why recorded | 7gf.25.13 |
| **window/lead/batch/optimiser knobs: SATURATED** | above | — | SATURATED | — | — |
| schedule / training length (decouple heads) | 7gf.117 | context head best ep16/40 | OPEN | patience 8 was a harness bug | 7gf.117 |
| early-stop patience 8 | uye, 7gf.117 | invalidated regulariser arms | CLOSED (bug) | the val dip at ep 13-21 precedes the LR-tail climb | 7gf.117 |
| best-epoch selection | selection_bias | wins were best-epoch bias | instrument | selection is seed-draw luck | interpretations/deep/2026-10-04_selection_bias.md |

## H. Broad track only

| axis | runs / beads | best Δ (ruler) | verdict | WHY | pointer |
|---|---|---|---|---|---|
| eval-time EA / shrink keep 0.5 | 7gf.56, 7gf.35.1 | +0.038 / +0.037 local; PUB +0.008/+0.009 | KEPT | cross-subject covariance shift | euclidean_alignment.md |
| train-time anchor | 7gf.35.2 | worse | CLOSED-null | no why recorded | 7gf.35.2 |
| BGMix p .5 | 7gf.112, 7gf.92 | Broad PUB +1.1/+0.7, local +1.9/+3.1; Deep PUB −1.9/−2.8 | KEPT (Broad only) | trades subject-0 fit for cross-subject generalisation; p1.0 collapses (no clean trials) | 7gf.112 |
| BGMix p .75 / 12 seeds / source-keyed | 7gf.112, 7gf.118 | .5967 vs .6141 (BR); 12 seeds no gain; mixed sign | CLOSED-null | — | 7gf.112, 7gf.118 |
| aligned/unaligned mix, seed ens | 7gf.29.8, 7gf.35.3/4 | ≤ +0.001; 0.5596/0.5661 | DIRECTIONAL | — | 7gf.35.4 |
| per-subject adaptation (subj 1-32 ses-11/12) | 7gf.132 | bg90-adapt12-elr{0.1,1.0} running 2026-10-07 | OPEN | — | 7gf.132 |
| attribution (`studies.attribution`) | efd | run on 11 Deep models, 400 val words | INSTRUMENT | one shared readout; set-pool front ends use ~40% less MEG; more MEG reliance ≠ more correct | efd |
