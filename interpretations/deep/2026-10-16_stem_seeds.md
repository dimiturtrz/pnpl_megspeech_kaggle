# Eight more pretrain seeds: does stem diversity grow the Deep ensemble?

megspeech-7gf.171 · recipe `ema-cd25-xsp10sem05-ac05x` stem + fine-tune, seeds 9–16 · 2026-10-16

## Question

fam8 (stems s1–s8, one fine-tune each) and fam16 (the same 8 stems, two fine-tune seeds each) score the same local
test (.7192 / .7204). A second fine-tune of a stem adds little diversity; a new stem might. Pre-registered bar for the
16-stem blend: local test ≥ .7154 and VAL ≥ .6467 to be built, public ≥ fam16 .69483 to replace a pick.

## Single members (sentence cut, τ .5)

| seed | VAL | local test |
|---|---|---|
| s9 | .6437 | .6866 |
| s10 | .6541 | .6811 |
| s11 | .6438 | .7060 |
| s12 | .6424 | .7005 |
| s13 | .6479 | .6988 |
| s14 | .6439 | .6940 |
| s15 | .6331 | .6943 |
| s16 | .6422 | .7210 |

New members sit ~1.5 pt below base s1–s4 (.709–.717) on local test with VAL in range. Not code drift: base-era code
(02c82fa) retrained at the same seeds gives the same spread (megspeech-7jg; same-seed reruns span local test
.679–.713, SD ~.013). Base s1–s8 were a favourable draw.

## Blends

| blend | members | VAL | local test |
|---|---|---|---|
| fam8 | s1–s8 | .6497 | .7192 |
| fam16 | s1–s8 × 2 ft | .6474 | .7204 |
| fam10stem | s1–s10 | .6476 | .7174 |
| fam16stem | s1–s16, one ft each | .6487 | .7140 |
| fam24 | fam16stem + 8 second fine-tunes | .6444 | .7170 |

Neither clears the bar: fam16stem passes VAL, misses local test by 0.1 pt; fam24 passes local test, misses VAL. Every
blend from 8 to 24 members sits in .714–.720 local test and .644–.650 VAL — inside one blend's rerun noise.

## Reading

The ensemble is saturated at this recipe. Eight more stems whose members average ~1.4 pt lower bring the blend back to
within 0.5 pt of fam8: stem diversity buys about what the members' weaker draw costs, no more. A second fine-tune and
a new stem are interchangeable as members. Size is not the lever; picks unchanged (Deep fam16 + d-old4). Not ruled
out: a stem recipe with a different inductive bias (a different teacher kind was tried and was flat, megspeech-7gf.29.1).
