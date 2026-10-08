# Attribution and embedding geometry of the word encoder (2026-10-08)

Scope: CPU, one seed, no confidence intervals. Models: Deep `sent40-masc27-s0` (1.0 s window) and the Broad pooled
stem `ema-bgmix05-all-w15-cx0-s0` (1.5 s window). Unit: the **word head** alone (encoder -> linear head, no span
transformer), on subject 0's local test (MOCHATIMIT ses-3, Sherlock1 ses-12 run-2, TheMoth ses-30), n=2681
labelled words. bal10 = macro recall@10; chance is 0.20. Occlusion sets the signal to zero, which is out of
distribution; read it as an upper bound on reliance, not as a clean removal. The Broad adapt model
(`...-adapt12-mask-ep40`) was **not run** because the compute cap was hit. The scripts are one-off
(`study2.py` in the job scratch dir) and are not committed.

## Q1 — where and when the word head reads

Windows start at onset (lead 0). **Pre-onset time is never seen.** The median IOI to the next word is 0.14 s,
and the next onset falls inside the window for 96.5% (1.0 s) and 99.9% (1.5 s) of words. So every window
carries 5–8 following words.

| Deep, 1.0 s | Δbal10 vs base 0.281 |
|---|---|
| zero one 100 ms bin, 0–0.4 s | −0.015 to −0.019 each |
| zero one bin, 0.8–1.0 s | ≈ −0.005 |
| keep 0–0.5 s only | −0.032 |
| keep 0.5–1.0 s only | −0.049 |
| truncate at 0.8 s | −0.013 |

Integrated-gradients (IG) time share peaks at 0.2–0.4 s (0.15 per 100 ms) and falls to 0.05 at 0.9–1.0 s.

| Stem, 1.5 s | Δbal10 vs base 0.244 |
|---|---|
| keep 0–1.0 s (drop the 1.0–1.5 s tail) | **−0.011** (Δlogp −0.045) |
| keep 1.0–1.5 s only | −0.042 |
| zero one bin, 0–0.5 s | −0.009 to −0.014 |
| zero one bin, 0.5–1.5 s | −0.004 to −0.007 |

Reading: the tail beyond 1.0 s is worth about 0.01 bal10 to the stem's word head. That is directional and
below the 1.5-point bar. The tail mainly contains the next words' onset responses. The stem's IG time share is
U-shaped: 0.14 at 0–0.1 s and 0.12 at 1.4–1.5 s, against 0.035 in the middle. Occlusion does not show the
same edge mass. **Inference:** the IG edge peaks are an artifact of the zero baseline plus conv padding, not
real reliance on the edges.

**Sensors (Deep).**

- Grads carry the signal. Dropping the 204 grads costs −0.071, which equals keeping only the mags. Dropping
  the 102 mags costs only −0.012.
- IG share is 0.79 grads / 0.21 mags. Per sensor, a grad gets 1.19× the uniform share and a mag 0.62×.
- Temporal regions dominate. Dropping left temporal costs −0.035, right temporal −0.025, and any other region
  between −0.001 and −0.012.
- Keeping only one temporal region (39 sensors) costs −0.043 (left) or −0.057 (right). Keeping only any other
  region costs about −0.08.
- Top IG sensors are left temporal: MEG0212/0213/0242/0243/0342, plus 1322.

**Sensors (stem).** The stem relies on sensors more diffusely.

- Dropping the mags costs −0.023 and dropping the grads −0.040.
- Right temporal, right parietal and the vertex matter most (−0.021 to −0.027 each when dropped). IG share is
  highest per sensor in right occipital and right parietal.
- Top IG sensors are right posterior: MEG2223/2233/1922/2322/2422.

**Inference:** pooling subjects moves the encoder away from subject 0's left-temporal sensors and toward
sensors that agree across heads.

## Q2 — what the embedding encodes (Deep; stem in brackets)

Probes use 5-fold GroupKFold by sentence and are fit **in-domain on the local test**, so they are not
comparable to a model trained elsewhere.

| probe | bal10 |
|---|---|
| word head (trained) | 0.281 [0.244] |
| logreg on E | 0.315 [0.326] |
| kNN25 cosine on E | 0.313 [0.324] |
| **timing only** (dur, IOI prev/next, position, t_end, sent_len; logreg on an expanded basis) | **0.437** |
| duration only | 0.324 |
| position only | 0.232 |
| E + timing | 0.389 [0.402] |
| E with timing regressed out | 0.250 [0.252] |

- **The timing leak is bigger than the MEG signal on this probe.** Timing alone (0.437) beats the embedding
  (0.315), and adding timing to E gives +0.074 [+0.076].
- Timing explains 21% [24%] of E's variance linearly. Removing it costs −0.066 bal10.
- E itself predicts these timing variables, as R²: sent_len 0.47, ioi_prev 0.36, t_end 0.30, position 0.24,
  ioi_next 0.09 [0.37], dur 0.04. For the stem, the window's valid length gives R² 0.93: zero-padding past
  sentence end is directly visible.
- An earlier run had shown timing-only bal10 ≈ 1.0. That was a bug in the probe metric: ranks counted only
  strictly greater probabilities, so ties were treated as hits. The number is corrected above. The stem's
  timing-only numbers come from the pre-fix code and are **void**.
- **Inference:** much of what the word head "decodes" is word duration and rhythm (function words are short
  and come at predictable positions), which the encoder recovers from the evoked responses.

**Clustering (eta², with permuted-label null).**

- Run: 0.226 (null 0.001). Label: 0.078 (null 0.020). Position bin: 0.085 (null 0.005).
- Run can be decoded from E at balanced accuracy 0.986, and at 0.917 after subtracting the word centroids
  (chance 0.33). Run is confounded with source.
- **Embeddings cluster by recording session about 3× more strongly than by word.**

**Neighbour words.** On rows where both neighbours are labelled (n=773):

- The previous word's label decodes from E at 0.361 [0.332]. The word's own label decodes at 0.282 [0.306] on
  the same rows. The next word's label decodes at 0.285 [0.300].
- **Inference:** the encoder reads the response to the preceding word, which lingers in the first ~0.3 s,
  about as well as the current word. That is context information in the word slot.

## Q3 — Broad stem geometry across subjects

Reference: subject 0's Sherlock1 ses-11 run-2. Each subject k's ses-11 run-1 has the same text. 1308 paired
words for subjects 1–12, about 670 for 13–22, about 315 for 23–32. **Caveat:** the stem's training pool may
include these ses-11 runs, so these are not held-out numbers. The adapt model, which held out ses-11, was not
run.

Medians over subjects 1–32:

- Word-head bal10 0.233 (range 0.207–0.281), against 0.252 for subject 0 on the same text.
- Domain AUC vs subject 0 is **1.00 for every subject**, and the mean shift is 0.82 of the within-subject
  spread.
- Paired-word linear CKA 0.165 (shuffled 0.028). Paired centred cosine 0.094. A same-label word at a different
  instance gives 0.01.
- Class-centroid cosine is 0.23 for the same word and −0.004 for different words. A nearest-centroid match
  across subjects gets 0.14.
- Held-out CCA's first canonical correlation is about 0.75 for subjects 1–22, and falls to 0.0–0.6 for 23–32,
  where n is about 300.

Kaggle Broad test archives use approximate windows rebuilt from the archive arrays (labels unknown):

- Domain AUC is 1.00 for subjects 1, 8, 16, 24, 32 and 33–39.
- Pseudo-label centroid cosine is 0.17–0.47 for most subjects, but **0.056 for subject 32 and 0.007 for
  subject 33**.
- Word-head max-p is about 0.15 everywhere, so the head is no less confident on subject 33 than elsewhere.

**Inference:** a shared word axis exists, since same-word centroids align at about 0.23 cosine. It sits on top
of a subject offset that separates subjects perfectly. Subject 33 looks least aligned to subject 0's word
geometry.

## Levers (inferences, sized from the probes above)

1. **Explicit timing features into the word or context head** (duration, IOIs, position, t_end). The probe
   gain is +0.07 bal10 over the embedding alone. The sentence transformer already gets onset-position and
   log-IOI inputs, so the realised gain is likely smaller. The next check is to fit the same timing probe on
   the full model's context logits. This is the only lever here plausibly above 1.5 points.
2. **Per-subject linear alignment** for Broad: re-centring per subject, plus Procrustes/CCA on shared-text
   runs or pseudo-labels. The mean shift is 0.8 of the spread and centroid cosine only 0.23. Size unknown;
   test on held-out ses-11 with the adapt model.
3. **1.5 s vs 1.0 s window:** the 1.0–1.5 s tail is worth about 0.01 bal10 to the stem's word head. That is
   below the bar, so the window is not a lever to plan around.
4. **Sensor subsets:** mags and non-temporal regions are near-redundant for Deep (−0.012 and ≤ −0.012).
   Dropping them saves compute but gains no accuracy. Not a score lever.
5. **Session/run invariance:** run is 3× stronger than word in E. A per-run normalisation or adversarial head
   may help transfer to the final test's unseen stimuli. Size unknown.
