# Ensemble kinds, ph05 redundancy, and the local-vs-public gap (2026-10-09)

**Scope.** This is an analysis of saved `probs.pt` files on CPU. No new training was run.

- **Members.** The 27 members of `m11xall16`, grouped into four kinds:
  - masc27w: 11 seeds.
  - sem7: the 7 members of `xspft7`.
  - sem1stem: the 3 `xsp10sem05-ft-s*` members.
  - ph: the 6 `xsp10ph05*` members.
- **Ruler.** Novel (unseen-span) words, top-10 balanced accuracy where ties count against the true word,
  macro-averaged over Sherlock1 and TheMoth. Tau is chosen on val. MOCHA is excluded.
- **Partitions.**
  - val is official-val: Sherlock1 n=1296, TheMoth n=1216, 47 and 47 classes present.
  - LT is the local test (official test): Sherlock1 n=1358, TheMoth n=775, 45 and 47 classes present.
  - 11–14 classes per domain have fewer than 5 words.
- **Check.** The reproduced numbers match the logs to ±.0009. For example, m11xsp7 scores .6593/.6934
  (val/LT) against .6591/.6929 logged.
- **Scripts.** The scripts are one-off and not committed: `why/q1.py` and `why/q23.py` in the job scratch dir.

## Q1 — why the ph05 members add nothing

| ensemble (equal member weight) | val | LT | (val+LT)/2 | public |
|---|---|---|---|---|
| masc11 | .6590 | .6876 | .6733 | — |
| sem7 (= xspft7) | .6464 | .7076 | .6770 | .68283 |
| ph6 | .6455 | .7016 | .6736 | — |
| sem10+ph6 | .6480 | .6916 | .6698 | — |
| masc11+sem7 (= m11xsp7) | .6593 | .6934 | .6763 | .68883 |
| masc11+sem7+ph6 | .6560 | .6944 | .6752 | — |
| masc11+sem10+ph6 (= m11xall16) | .6547 | .6964 | .6756 | .68883 |

Single models (mean ± SD over seeds):

| kind | n seeds | val | LT |
|---|---|---|---|
| masc | 11 | .6599±.0031 | .6818±.0144 |
| sem7 | 7 | .6406±.0065 | .6940±.0115 |
| ph | 6 | .6396±.0048 | .6901±.0137 |

**Measured: correlation of the true-class log-prob between member pairs (LT, raw).**

| pair | r |
|---|---|
| masc–masc | .971 |
| sem7–sem7 | .964 |
| **ph–sem7** | **.966** |
| masc–sem7 | .954 |
| masc–ph | .958 |

- With word identity removed (per-label mean subtracted), the same ordering holds. ph–sem7 is .947, which
  is at least as high as sem7–sem7 (.943).
- Between kind means, ph–sem7 correlates at .993 and masc–sem7 at .982.
- Top-10 hit disagreement on LT (n=2133):

  | pair | disagreement | words only the first gets | words only the second gets |
  |---|---|---|---|
  | masc vs sem10 | 4.8% | 46 | 57 |
  | sem10 vs ph | 3.1% | — | — |

**Interpretation (inference).**

- A ph05 member is statistically another sem05 seed. Its correlation with sem7 is no lower than a sem7 seed's
  correlation with another sem7 seed. The auxiliary target (phoneme vs semantic) does not move the errors.
- The only real diversity axis in the pool is masc versus xsp.
- Adding ph6 and sem1stem to m11xsp7 raises the xsp share from 7/18 to 16/27. That trades val for LT
  (−.005 / +.003) and leaves (val+LT)/2 flat (.6763 → .6756). The public score ties (.68883 on both).
- In short, m11xall16 is a re-weighting of the same two kinds, not added diversity.

**Does NOT rule out.** A ph05 model with a *different* stem or recipe could still be diverse. Every ph member
here shares the xsp10 recipe and fine-tune path.

## Q2 — why xspft7 looked better locally but worse publicly

**Measured.** The paired block bootstrap of sem7 − masc11 (macro, tau .5, blocks of 20 consecutive novel
words, 2000 reps) gives:

| partition | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|
| val | −.0126 | [−.032, +.005] | .08 |
| LT | +.0200 | [−.004, +.034] | .94 |

- **The ranking flips by partition.** masc is better on val in both domains. xsp is better on LT in both
  domains (Sherlock1 +.015, TheMoth +.025).
- Neither interval excludes 0, and LT single-seed SD is about .014.
- The per-class LT advantage of xsp sits in the frequent tercile. By train-prior tercile (rare / mid /
  frequent):
  - LT: −.004 / +.015 / +.046
  - val: −.028 / −.006 / −.019
- r(per-class Δ, log prior) is only .06 on LT and .15 on val. This is not a clean frequency effect.
- On (val+LT)/2, all three submitted blends tie within .0015: xspft7 .6770, m11xsp7 .6763, m11xall16 .6756.
  Public separates them by .006, which is below the .023 single-score SE. A paired public difference is
  smaller, but still far from 1.5 points.

**Interpretation (inference).**

- xspft7's +.015 local edge is one partition's draw. That partition holds 2133 words, 47 classes and about 11
  classes with fewer than 5 words per domain, so it is not a property of the model. The other partition
  showed the opposite sign at similar size.
- Each partition is roughly a ±.02 ruler for a kind-level difference. Selecting on LT alone (or on val alone)
  picks whichever kind that draw favoured.
- The public order (mix ≥ xsp-only) is consistent with "the hedge wins in expectation". It is not evidence
  that xsp generalises worse.
- This matches the earlier finding that LT does not rank small blends (mix22 LT .7172 → public .6488).

**Does NOT rule out.**

- A real but small kind × stimulus interaction. Sherlock vs TheMoth do not explain the flip, because both
  domains flip together. That points to session or partition, not genre.
- The isolated-word Kaggle rows. The xsp word head is better (+.015 to +.02 LT), so those rows would favour
  xsp, not explain its deficit. They sit in the private split anyway.

## Next experiments (ranked by plausible gain)

1. **A genuinely different third kind instead of more xsp seeds.** Candidates are a different window, the
   pooled-stem family, or a different readout. masc vs xsp disagree on 4.8% of words, and ph vs xsp on only
   3.1%.
   - Expected gain: about +.005–.01. Blending two near-equal kinds gave +.003 over the better single kind on
     (val+LT)/2, so even a well-decorrelated kind is unlikely to reach 1.5 points alone.
   - Cost: about 1 GPU-day.
2. **Select blends on (val+LT)/2, or better a 3-ruler mean including kcut.** This replaces selecting on one
   partition. It is zero cost and avoids the ±.02 single-partition draw. It is an instrument, not a score gain.
3. **Kind-balanced weighting** (equal weight per kind, not per member). This is directional only: masc|sem7
   kind-balanced scores .6534/.6928, a tie. Not worth a submission.
