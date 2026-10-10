# Instance-contrastive loss on held-out MASC subjects (2026-10-10)

**Scope.** Three Broad stems trained from the `ema-bgmix05-w15-s0` recipe on subjects 0, 101–123, 201–212, 301–320, 323,
seed 0, one seed each. They are scored on MASC subjects 124–127, which no stem trained on: 8 runs per subject,
21,488 words, top-10 balanced accuracy at tau 1. This measures how well each stem transfers to new subjects. It is
not a Broad test score.

| stem | change | held-out ruler (word head) | vs ctl |
|---|---|---|---|
| `instance-ctl-s0` | none | .2399 | — |
| `instance-pairs-s0` | `--sampler.pairs 0.25` (same-word pairs in the batch) | .2441 | +.004 |
| `instance-w025-s0` | pairs + `--optim.instance 0.25` (contrastive loss on those pairs) | .2631 | +.023 |

`instance-w025-s0` per subject: .257, .276, .277, .242. It beats `pairs` on all four (.237, .256, .254, .228).

**The ruler had to change first.** The context head scored 1.0000 for all three stems. Every MASC subject heard the
same stories, so the context head recognises the story text rather than reading the brain. `--head word` (307c105)
scores only the per-word MEG head, which cannot recall the story. MOCHA's context-head VAL (.93–.97) is inflated in
the same way.

## Reading (inferences)

1. **The contrastive loss carries the gain, not the sampler.** Pairing alone moves +.004. Adding the loss on the same
   pairs moves another +.019.
2. **This is one seed and one transfer measure.** It clears the +.015 bar, so the stem is kept, but the result is
   not yet confirmed.
3. **The deployment test is the Broad adapt.** `bq102` fine-tunes each stem on subjects 1–32 (`adapt12-mask-ep40`).
   Transfer to new subjects only pays off if the gain survives that adaptation.

## Not ruled out

- Whether 0.25 is the right weight. 0.5 and 1.0 are queued (`bq101`) and will be scored on the word head.
- Whether the gain holds on a second seed. Spend one only if the adapt keeps it.
