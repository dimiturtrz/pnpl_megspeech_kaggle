# Deep: which local ruler tracks public (2026-10-10)

Setup: the 32 Deep submissions whose VAL and local test (LT) we can match to a log or to the numbers quoted in
the submission message. All at tau .5 macro, sentence grouping. Public is the Kaggle public leaderboard (~434
rows). No model was re-scored.

## Correlation with public

| subset | n | VAL Pearson / Spearman | LT Pearson / Spearman |
|---|---|---|---|
| all matched | 32 | .49 / .38 | .49 / .28 |
| without the 5 rows tied at .68883 | 27 | .40 / .24 | .58 / .47 |
| blends only | 22 | .54 / .33 | .45 / −.01 |
| singles only | 12 | .34 / .48 | .44 / .46 |

Both rulers are weak. Most of these cells are not significant at this n.

## Direction of a single change

On 9 pairs of submissions that differ by one change, the VAL delta had the same sign as the public delta 4/9
times, and the LT delta 6/9 times. Where VAL and LT disagreed in sign, public followed LT in 3 of 4 cases. In
the right-cut pair (`s-mhub4` → `s-mhub4rc18`), LT predicted both the sign and the size (−.021 LT, −.037 public),
while VAL predicted a gain.

## Why each ruler fails

- VAL chose the epoch and tau, so it is biased upward. It is also cut by sentence, so it cannot see
  span-edge changes, which is how the right-cut regression got past it.
- LT is unbiased by selection but covers two texts (Sherlock1, TheMoth n=775). Adding a whole speech-target
  family raises LT by about .02 while public stays flat: m11 + mhub tied at .68883, and m11wd shows LT +.020
  with VAL +.0015 (submission pending). Inference, not verified: the family is better on these two texts, and
  that does not carry to the public stimuli.

## Saturation

Six distinct Deep CSVs score exactly .68883. Adding a few members does not flip any argmax on the public rows,
so public cannot rank changes that small.

## Greedy member selection fits the selecting ruler

Caruana forward selection with replacement over m11wd's 36 members (tau .5 macro, 15 steps). All 36 members
equally weighted: VAL .6547, LT .7172.

| select on | selecting ruler | other ruler |
|---|---|---|
| VAL | .672 (+.018) | LT ≈.691 (−.026) |
| LT | .743 (+.026) | VAL ≈.646 (−.009) |

Each direction gains on the ruler it selects on and loses on the other. The two rulers want different families:
VAL picks masc27w singles (one alone scores VAL .664, above the full blend), and LT picks the speech-target
fine-tunes. Equal weights over all members has the best held-out score both ways, so no subset was submitted.
Script: `tmp/greedy_deep.py` (job scratch).

## Working rule

Trust a change when VAL and LT move the same way. When they disagree, LT is the better guide to direction,
except for family additions, where public decides. Raw table: `tmp/rulers/` (job scratch; facts recorded in beads 7gf.132).
