# Uncentred windows: the slow level inside the model

megspeech-6io · `python -m megspeech.training.cli.arm --base ema-cd25-xsp10sem05-ac05x --seed k --tag unc -- --no-data.centred`
· 2026-10-16

## Question

The model's input window is centred on its own mean, so the slow level (ch. 3's SCP / CNV) never reaches it.
[The paradigm study](2026-10-16_bci_paradigm_features.md) found that level decodes alone (.233, chance .20) but adds nothing
as a booster feature on top of a frozen base. Does it help when the network itself sees it, trained end to end?
`data.centred = False` keeps each window as it is cut; everything else is the base recipe.

## Pre-registration

Single seeds against the base (redraw mean local test .697, SD .013; VAL SD ~.005): 2-seed mean ≥ .712 local test.
Family against d-ac05xfam (4 seeds, VAL .6459, local test .7289): keep iff VAL ≥ .6509 and local test ≥ .7239.

## Result (macro bacc@10, tau 0.5)

| arm | VAL | local test |
|---|---|---|
| base s1-ft-s1 / s2-ft-s2 | .6387 / .6407 | .7088 / — |
| unc s1-ft-s1 | .6494 | .6958 |
| unc s2-ft-s2 | .6567 | .7255 |
| unc 2-seed mean | .6531 | .7107 |
| d-uncfam2 blend | .6559 | .7161 |
| unc s3-ft-s3 | pending | pending |
| unc s4-ft-s4 | pending | pending |
| d-uncfam4 blend | pending | pending |
| d-ac05xfam (control) | .6459 | .7289 |

## Reading

Two seeds move VAL and local test the same way: +.013 VAL over the base pair, +.014 local test over the redraw mean.
The single-seed bar (.712) is missed by .0013, so it is not a pass yet; the 4-member family decides.
