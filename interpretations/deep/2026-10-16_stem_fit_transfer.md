# Stem fit against transfer: capacity and stem length

megspeech-sou (h512), megspeech-toa (20-epoch stem) · recipe `ema-cd25-xsp10sem05-ac05x` stem + fine-tune · 2026-10-16

## Question

Several arms that fit the stem harder raised VAL and lowered the local test: the 80-epoch stem (megspeech-d6r.5,
−1.6 local test on both seeds) and the anchor arms (megspeech-7gf.109.7 / 170, −0.2 to −2.6). Two tests of that
direction: more capacity (conv hidden 256 → 512; 384 was flat, megspeech-7gf.46) should help if the model is capacity-bound; a 20-epoch stem (full
schedule, not a truncation) should help if stem fit to the heard stimuli is what costs transfer. Pre-registered bar for
each: 2-seed mean local test ≥ .726 (base .711 + 1.5).

## Capacity: h512 (sentence cut, τ .5)

| seed | VAL (h512 / base) | local test (h512 / base) |
|---|---|---|
| 1 | .6459 / .6387 | .7086 / .7088 |
| 2 | .6416 / .6407 | .6858 / .7131 |

Mean local test .697: −1.4 against base and equal to the redrawn-seed mean (~.697, megspeech-7gf.171). VAL +0.4 is
inside its rerun SD (~.005). Seed 1 led VAL by ~2.5 pt at fine-tune epochs 5–8, but the lead did not survive to the
end or reach the local test.

The model is signal-bound, not capacity-bound: extra width fits faster and lands where the base lands.

## Stem length: 20 epochs

| seed | VAL (stem20 / base) | local test (stem20 / base) |
|---|---|---|
| 1 | .6444 / .6387 | .6874 / .7088 |
| 2 | .6398 / .6407 | .7066 / .7131 |

Mean local test .697: again equal to the redrawn-seed mean, −1.4 against the favourable base draw. The stem took
12 min instead of 25, and seed 2 led base fine-tune VAL by ~1.5 pt at epochs 14–18 before converging to it.

## Reading

Neither direction moves the local test: wider (h512) and less-fit (stem20) both land at the redrawn-seed mean, and
only the 80-epoch stem fell below it. Stem fit between 20 and 40 epochs is flat; the transfer gap is not set by how
hard the stem fits. With capacity and stem length at the redrawn-seed mean, sensor augmentation below it
(2026-10-16_sensor_augment.md) and ensemble size
(2026-10-16_stem_seeds.md) saturated, the recipe is at its signal ceiling for single-subject Deep.
Base s1/s2 (.711) were a favourable draw; comparing new arms against them overstates every loss by ~1.4 pt.
