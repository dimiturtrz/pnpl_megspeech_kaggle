# Classical BCI paradigm features as extra inputs

megspeech-508 · `python -m megspeech.evaluation.studies experts.paradigms` · peer booster (megspeech-6ow) on peerbase-s0
with kf5t out-of-fold rows · 2026-10-16

## Question

Do the signals of the BCI paradigms in Vourvopoulos & Perdikis ch. 3 add learning signal when given to the model as
extra feature vectors? The cheapest honest instrument is the peer booster: the frozen base's log-probabilities plus a
feature group, against the same booster on the base alone. Pre-registered keep: fold cross-entropy below the control's
and a 3-seed official mean ≥ control + .015.

## Groups

Read from the 32 leading spatial components of the out-of-fold windows (1 s, 50 Hz):

| group | paradigm (ch. 3) | features |
|---|---|---|
| band | SMR ERD/ERS 3.2, passive 3.6, affective 3.7, error theta 3.8 | log power, delta / theta / alpha / beta, 128 |
| means | P300 3.3 (Eq. 3.2), evoked 3.4 | 100 ms window means, 320 |
| covariance | Riemannian (3.2, 3.6) | log-Euclidean tangent of 12 components, 78 |
| level | SCP / CNV 3.5 | uncentred window mean, 32 (the model's input is centred, so this is new to it) |

Not testable here: high gamma (3.9) is above the 50 Hz input's Nyquist; megspeech-7gf.80 measured it on 250 Hz data
alone (.2315, chance .20) and on top of the low band (−1.1 pt). Surprisal (N400 / ErrP) needs the word's own identity.
SSVEP / cVEP need a periodic or coded stimulus.

## Result (official macro bacc@10, chance .20)

| arm | fold CE | official, 3 seeds | mean |
|---|---|---|---|
| base-only control | 2.369 | .650 .648 .645 | .648 |
| + band | 2.382 | .637 .648 .646 | .644 |
| + means | 2.387 | .647 .645 .651 | .648 |
| + covariance | 2.379 | .639 .638 .651 | .643 |
| + level | 2.374 | .647 .641 .635 | .641 |
| + all | 2.392 | .647 .646 .641 | .645 |
| band alone | 3.397 | .213 .212 .216 | .213 |
| means alone | 3.334 | .285 .287 .290 | .287 |
| covariance alone | 3.392 | .211 .216 .215 | .214 |
| level alone | 3.388 | .223 .228 .247 | .233 |
| all alone | 3.328 | .270 .268 .276 | .272 |

## Reading

No group adds to the base: every arm's fold CE is worse than the control's, and every official mean sits at or below it.
The harness works: the same features alone decode above chance, and the ordering is the one the physiology predicts.
The phase-locked window means carry the most (.287), the induced measures (band power, covariance) barely clear chance
(.21), and the slow level carries a little (.23).

What the CNN reads from a centred 50 Hz window already contains everything these features decode, at more than twice
their strength (.64 against .29). Induced power and covariance inside a 1 s word window hold almost no word identity.
Ch. 3's own framing explains this. ERD/ERS, workload, affect and error theta are slow, state-level or task-level
signals measured over 2–10 s windows, and they track predictability rather than which word was heard. The uncentred
level, the one input the model never sees, decodes at .23 and adds nothing either.

Not ruled out:
- Features from per-sensor channels rather than 32 components.
- Supervised spatial filters (xDAWN per class, CSP).
- Longer, sentence-level windows for the state paradigms.
- A newer base than peerbase-s0.

None of these is priced near the bar: the strongest group alone is less than half the base.
