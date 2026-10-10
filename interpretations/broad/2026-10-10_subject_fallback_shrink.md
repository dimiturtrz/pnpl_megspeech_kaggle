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

## Prior art and what this does not rule out

7gf.35.1/.3 shrank toward the **Deep** matrix (keep .5: +.009 public as a single model, .5244 as an ensemble).
Pulling toward the mean is the stronger target here (deep zero-shot .5706 vs mean .6137). Training-time
anchoring toward the Deep matrix hurt (7gf.35.2: the trunk co-adapts). Training-time anchoring toward the mean
(`--optim.anchor_toward mean`, 252917b) is queued but not yet tested. `--aligned` made no difference on these
models.
