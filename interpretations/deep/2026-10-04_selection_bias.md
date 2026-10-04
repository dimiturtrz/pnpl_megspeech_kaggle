# Deep selection bias: best-epoch and seed-draw luck (2026-10-04)

Question (bd `megspeech-7gf.57`, `megspeech-7gf.29.1`): a MASC-pretrained fine-tune won 8/8 paired seeds against
its EMA twin, yet its 8-seed ensemble lost to the EMA ensemble. A disjoint EMA ensemble then scored 0.018 below
the first. Which numbers on the Deep ruler (sentence-span novel macro BAcc@10, n=2512) can be trusted?

## Answer

Two selection effects sat on the same ruler and inflated every comparison against the champion:

- **Best-epoch bias (singles).** A single run reports its best of 40 epochs, picked on the ruler that reports it.
  The bias depends on the curve's *shape*, so it differs between recipes. Comparing singles across recipes on
  their maxima compares the biases.
- **Seed-draw luck (ensembles).** The 8-seed champion `ens-ema9998-s0to7` (0.6626) was a lucky draw of seeds.
  The honest 8-seed expectation is ~0.645–0.654.

Neither effect is a code bug:

- **Scorer:** rebuilding the champion with the current scorer reproduces 0.6626 exactly.
- **Training:** retraining seed 0 with the current code tracks the stored loss to ~1e-3 per epoch.

## Best-epoch bias

| recipe | curve shape | max − last | 8-seed mean curve |
|---|---|---|---|
| EMA 0.9998 | smooth, peaks late | ~0.01 | peaks 0.635 |
| MASC fine-tune | peaks at epoch 0, then overfits | ~0.05 | 0.652 at ep0, falls to 0.596 |

The fine-tune's 8/8 per-seed "wins" (+0.006 to +0.016) were its larger bias. On ensembles, which pick no epoch,
it lost: `ens-masc27ft8` 0.6516 vs the 8-seed EMA 0.6626 (−0.011). Against the honest 16-seed baseline (0.6535)
it ties.

## Seed-draw luck

| ensemble | seeds | local (n=2512) |
|---|---|---|
| ens-ema9998-s0to7 | 0–7 | 0.6626 |
| ens-ema8b | 8–15 | 0.6450 |
| ens-ema16 | 0–15 | 0.6535 |
| ens-ema24 | 0–23 | 0.6552 |

Each paired comparison against `s0to7`:

- **Seeds 8–15:** −0.018, CI [−0.036, −0.000].
- **16 seeds:** −0.010, CI [−0.022, +0.001].
- **24 seeds:** −0.008, CI [−0.018, +0.002].

Two disjoint 8-seed draws differ by about as much as the 1.5% bar. Past 16 seeds the score saturates: 24 seeds
add +0.0018 over 16, CI [−0.007, +0.011].

## Consequences

- **A "loses to the champion" verdict from this week used an optimistic bar:** the champion carried about +0.009
  of draw luck.
- **Compare singles on the seed-mean curve or a fixed epoch.** Never use per-seed maxima, and check max − last
  before believing a delta.
- **The final pick is the largest ensemble** (`ens-ema24`), not the highest local score. The 8-seed lead is
  winner's curse on the same n=2512 ruler. The public LB cannot arbitrate it, since it cannot rank models closer
  than ~0.04 (memory `ruler-lb-rank`).

## What this does not rule out

- MASC pretraining over several independent pretrain seeds (pretrain-seed spread ~0.013 dominated one pretrain).
- A low-lr, 1-epoch fine-tune that avoids the overfit tail. Priced at ~+0.003 for ~4 h; parked.
