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
| unc s3-ft-s3 | .6577 | .7178 |
| unc s4-ft-s4 | .6605 | .6833 |
| unc 4-seed mean | .6561 | .7056 |
| d-uncfam4 blend | .6687 | .7096 |
| d-ac05xfam (control) | .6459 | .7289 |

## Reading

Fails the pre-registration: VAL passes by a wide margin (.6687 against .6509, the best VAL any Deep blend has
reached), the local test misses (.7096 against .7239, −.019 below the control family). Not kept, no public slot.

The two partitions disagree, and that disagreement is the finding. On VAL every one of the four seeds beats both base
seeds (+.016 mean, VAL SD ~.005), so the slow level is not noise there. On the local test the seeds scatter around the
base redraw mean (+.009, SD .013), and blending them gains less than blending centred seeds does (+.004 over the seed
mean against the control's +.02). VAL is held-out subject-0 sessions of the training sources; the local test is pnpl's
own validation and test recordings. A slow level is mostly recording state (drift, head position, filter settling), so
what it adds is plausibly specific to sessions recorded like the training ones, and it washes out on recordings that
are not. That would also explain why the level decoded alone at only .233 in the paradigm study and added nothing on
top of a frozen base read on out-of-fold rows.

Not ruled out:
- A high-passed level (keep < 1 Hz, drop the DC offset) that keeps an SCP but loses the per-recording offset.
- The level as a separate, per-session normalised input channel beside the centred window.
- The public leaderboard as the arbiter: it is a third recording set, and an owner-approved instrument slot would say
  which partition the final test resembles.
