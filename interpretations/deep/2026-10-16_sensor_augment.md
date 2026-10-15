# Sensor augmentation read on the local test

megspeech-7gf.172 · recipe `ema-cd25-xsp10sem05-ac05x`, `--augment.dropout .2 --augment.gain .1` in stem and fine-tune ·
2026-10-16

## Question

Every earlier input augmentation (jitter 7gf.18, mask + noise cho, stretch d6r.7) was judged on VAL alone, and Broad
sensor augmentation lost public .019 (d6r.21). The anchor arms showed VAL and local test can move in opposite
directions; input regularisation is the mirror case — it could cost VAL and still help transfer to stimuli no model
heard. Pre-registered bar: 2-seed mean local test ≥ .726 (base .711 + 1.5) and VAL ≥ .635.

## Result (sentence cut, τ .5)

| seed | VAL (aug / base) | local test (aug / base) | TheMoth local test (aug) |
|---|---|---|---|
| 1 | .6351 / .6387 | .6906 / .7088 | .632 |
| 2 | .6288 / .6407 | .6813 / .7131 | .627 |

Mean local test .686, −2.5 against base and −1.1 against the redrawn-seed mean (~.697, megspeech-7gf.171). The
augmentation landed: stem SpeechTerm at epoch 4 is 5.71 against 5.53 without it. Fine-tune VAL tracks base for
seed 1 and trails ~1.5 pt for seed 2 from epoch 4 on.

## Reading

Sensor dropout and per-sensor gain cost both VAL and local test, on both sources. The gap between heard and unheard
stimuli is not a sensor-side invariance the model lacks; it is stimulus-side, so noise at the sensors only removes
signal. This agrees with Broad's public loss. Not ruled out: augmentation in the fine-tune only, or a milder dose —
neither has a mechanism that predicts +1.5 pt, so neither is filed.
