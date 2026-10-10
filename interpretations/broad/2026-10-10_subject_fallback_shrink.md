# Broad: shrinking subject matrices toward their mean (2026-10-10)

Setup: one model, `ema-bgmix05-all-w15-s0-adapt12-mask-ep40`, scored on the Broad ruler (32 seen subjects,
ses-11, n=25322) with `--fallback.unseen {identity,deep,mean} --fallback.keep K` (`SubjectLayer.shrink`, which keeps
`K` of each subject matrix's deviation from the fallback). The model and scoring are the same in every arm;
only the fallback changes.

## Zero-shot reading (every subject through the fallback)

| fallback | ruler |
|---|---|
| own matrices | .6000 |
| identity | .5812 |
| deep (subject 0) | .5706 |
| mean of trained matrices | .6137 |

Every Broad submission before this one read the zero-shot Kaggle subjects 33–39 (7/39 of the rows) through
**identity**, which is .019 below `own` and .033 below `mean`.

## Keep curve (seen subjects use a shrunk matrix; zero-shot subjects use the mean)

| keep | 0 | .1 | .25 | .4 | .5 | .75 | 1 (own) |
|---|---|---|---|---|---|---|---|
| ruler | .6137 | .6186 | **.6199** | .6109 | .6078 | .6051 | .6000 |

Gain by data size, keep .25 vs own: rich subjects (n≈1300) +.002, mid (≈660) +.012, poor (≈300) +.049. The best
keep per group is .1/.25/.25, so a keep that depends on data size would add about +.0005; one global keep is
enough. Mechanism: subjects with few sessions overfit their own matrix, and the mean pools the other 32 subjects'
estimates (a James–Stein shrinkage).

## Public transfer (same model, 2 submissions)

| arm | ruler | public |
|---|---|---|
| `b-s0own` (own, zero-shot via identity) | .6000 | 0.58199 |
| `b-s0k25` (mean, keep .25) | .6199 | 0.60704 |

The public gain is +.025, against +.020 on the ruler. The ruler cannot see the zero-shot rows, which also gain
(identity → mean). With this fallback the single model matches the earlier 29-model blend without Sinkhorn
(0.604).

## On the 29-member blend (xc5w6all6)

| arm | ruler | public |
|---|---|---|
| plain blend | .6329 | .604 |
| + Sinkhorn β.25 (`b-xc5sink25`) | — | .61513 |
| + mean fallback keep .25 (`b-xc5k25`) | .6389 | .61238 |
| + both (`b-xc5k25s`) | .6591 | **.62262** |
| + both, keep .1 | .6534 | — |

Keep .25 is also the best keep on the blend, not only on the single model.

On public the two post-steps add: shrink alone +.008, Sinkhorn alone +.011, both +.019 over the plain blend. So
neither step uses up the other's gain.

Adding 11 members (the `w15l60` family s0–4, stems and adapts, plus one `smn-ac03` pair) with both steps on
(`b-bxl60k25s`, 40 members) raised the ruler +.004 (.6631) but cost .0055 public (0.61709 vs 0.62262). The ruler
and public disagree on the sign. The ruler cannot see the 7 zero-shot subjects, and it scores ses-11, a text
every subject also read. Which of the two blinds this is, we have not tested. Working rule: the 29-member blend
stays the Broad base, and a family is added only on a public A/B.

On the blend the shrink adds +.006 on the ruler, against +.020 for the single model. On top of Sinkhorn it adds
+.0075 public. Inference: averaging 29 members already cancels part of each member's per-subject overfit, so
the shrink and the ensemble overlap. The two post-steps compose: on the ruler, Sinkhorn adds more on top of the
shrink than the shrink adds alone.

## Prior art and what this does not rule out

7gf.35.1/.3 shrank toward the **Deep** matrix (keep .5: +.009 public as a single model, .5244 as an ensemble).
Pulling toward the mean is the stronger target here (deep zero-shot .5706 vs mean .6137). Training-time
anchoring toward the Deep matrix hurt (7gf.35.2: the trunk co-adapts). Training-time anchoring toward the mean
(`--optim.anchor_toward mean`, 252917b) is queued but not yet tested. `--aligned` made no difference on these
models.
