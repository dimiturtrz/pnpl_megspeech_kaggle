# Broad ruler: member selection does not transfer across text; family decisions do (2026-10-09)

Can the Broad ruler (subjects 1–32 on Sherlock1 ses-11, unseen spans, macro bal10) choose a blend that is still
better on text it has never seen? Everything here is CPU scoring of saved models, with no retraining.

- Blends average log-softmax, the same geometric mean `Ensemble` uses.
- Scoring is aligned and fixed at tau 0.5, which every arm selected.
- One split, no confidence intervals. Tracked in megspeech-7gf.147. Scratch scripts are not committed.

**Validation.** The 29-model all-in blend reproduces the logged `xc5w6all6` ruler score, .6329, exactly.

## Clean models for a cross-text test

Subjects 1–32 also have a ses-12 run with different text. Every `adapt12` model trained on subjects 1–32's
ses-12, so for them ses-12 is training data. Only the 11 stems are clean on both sessions: `bgmix05-s0..5`,
`ema-bgmix05-s0/s1` and `ema-bgmix05-w15-s0..2`. Each trained on subject 0 only, with no adapt, and subject 0's
ses-12 text is the local test. The cross-text tests therefore cover stems only.

## 1. Member-level greedy selection: text fit

The subject halves are seeded and stratified by word-count tier: A = 1, 3, 4, 5, 6, 8, 16, 17, 18, 20, 21, 23,
24, 26, 28, 30; B = the rest. Selection is greedy forward with replacement, starting from the best single model.

**Split by subjects only** (30-model pool, both halves on the ses-11 text). Selection looks real:

| fit → score | all-in | selected | Δ |
|---|---|---|---|
| A → B | .6323 | .6475 | +.015 |
| B → A | .6338 | .6446 | +.011 |

**Split by subjects and text** (stems only; fit on ses-12, score on ses-11; pre-registered at ≥ +.005 in both
directions):

| fit → score | all-in stems | selected | Δ |
|---|---|---|---|
| ses12/A → ses11/B | .6221 | .6197 | −.0023 |
| ses12/B → ses11/A | .6335 | .6179 | −.0157 |
| ses11/A → ses12/B (not pre-registered) | .6975 | .6918 | −.0057 |
| ses11/B → ses12/A (not pre-registered) | .6851 | .6963 | +.0113 |

The pre-registered test fails. The subject-split gain came from fitting the ses-11 text, which every subject
hears. This matches the earlier `xc5w6-lw` precedent: per-label weights fit on the ruler gained +.0138 there and
lost .0005 on public.

## 2. Family-level decisions: these transfer

All-in blends of the families bgmix (6), ema (2) and w15 (3), and of every union of them:

| blend | ses-11 | ses-12 |
|---|---|---|
| bgmix | .6146 | .6782 |
| ema | .5994 | .6721 |
| w15 | .6109 | .6754 |
| bgmix+ema | .6181 | .6847 |
| bgmix+w15 | .6285 | .6903 |
| ema+w15 | .6114 | .6786 |
| all three | .6278 | .6913 |

- Spearman correlation across the two sessions is 0.929 (p .0025).
- The sign of a pairwise difference agrees in 19 of 21 pairs. Among pairs with |Δ| ≥ .005 on ses-11 it agrees
  in 16 of 16, which passes the ≥ 80% pre-registration. The two flips are pairs within .0037.
- **The 0.7× shrink.** Measured differences repeat with the same sign but smaller on new text. Adding w15 to
  bgmix+ema gives +.0097 on ses-11 and +.0066 on ses-12. Inference: read a ruler margin at about 0.7×, and treat
  about .005 as the resolution floor.

## 3. Family weighting: no gain

Weighting schemes compared against member-equal all-in (ses-11 / ses-12):

| scheme | ses-11 | ses-12 | Δ |
|---|---|---|---|
| member-equal (current) | .6278 | .6913 | — |
| family-equal (1/3 per family) | .6207 | .6950 | −.0071 / +.0037 |
| √n family weight | .6246 | .6926 | −.0032 / +.0013 |

Pre-registered: adopt a scheme only if it gains ≥ .005 on both sessions. Both fail, and their signs flip
across text. Keep member-equal.

## Reading

The Broad ruler can rank coarse blend decisions: adding or dropping a whole family. It cannot rank member picks
or reweightings that sit below its ~.005 resolution. Read a family-add margin at about 0.7× when comparing it
to a bar. These tests used stems only; adapt families can't be tested across text, because no Broad session is
untrained for them on both texts.
