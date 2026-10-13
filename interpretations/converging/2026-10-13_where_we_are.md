# Where we are — 2026-10-13

Supersedes [2026-10-07](2026-10-07_where_we_are.md). Goal epic megspeech-d6r: Deep public ≥ .705, Broad ≥ .620.
Levers close 2026-10-14 23:00 Sofia; 10-15/16 is final selection only (7gf.29.1). Kaggle closes 2026-10-16 12:00 UTC.

## Scores

| track | best public | submission | 10-07 best | leader (10-11) |
|---|---|---|---|---|
| Deep | **.69133** | d-old4 (xsp10sem05-ft, 4 seeds) | .6638 | .77474 |
| Broad | **.63097** | b-xc5k25b0ps (xc5w6all6 29 members + mean fallback keep .25, no Sinkhorn) | .5879 | .73235 |

Broad met its goal. Deep did not: no public gain since d-old4 (10-10). Recipe families score .685–.691
(d-ac05xfam .691, m11 .689, d-old9 .685), so that is this recipe's public ceiling, and seeds or family size move it
only within draw noise.

## What moved 10-07 → 10-13

- **Broad +.038:**
  - Zero-shot rows read through a MEAN subject fallback with keep .25 instead of IDENTITY: +.020 single-model ruler.
  - Transductive Sinkhorn balancing on output probabilities, per subject: +.011 ruler, but NOT on public: toward
    0/.05/.1/.25/.5 = .631/.629/.627/.626/.585 public vs .639/.647/.652/.659/.652 ruler. The ruler's class marginals differ from
    the organisers' test, so the gain is ruler-specific; .5 is a cliff on public.
  - Bigger family (xc5w6all6).
- **Deep +.028:** the xsp/sem/acoustic auxiliaries and family blending, all before 10-10.

## Kills since 10-10 (scope = this use; see beads)

| lever | read | bead |
|---|---|---|
| high-gamma input | null twice before (qi7.1, 7gf.80) | d6r |
| 80-epoch stem | LT −.016 (2 seeds, both groupings) | d6r.5 |
| cross-fit kNN readout | LT −.062 sentence / −.036 kaggle, VAL leaked | d6r.6 |
| time-stretch aug .1 | LT −.031 / −.019 | d6r.7 |
| pool smoothing | best +.008 < .015; purity .20–.24 | d6r.8 → [pool smoothing](../deep/2026-10-13_pool_smoothing.md) |
| ft-stage SupCon for pool smoothing | macro purity +.015, pool LT −.006 | 7gf.86 |
| timing expert fusion (incl. LibriSpeech pretrain) | LT −.0095 at VAL weight | r9m.1 → [timing](../deep/2026-10-13_timing_and_span_cuts.md) |
| nd / ended / noint fusion | parked: ac05x 4-seed LT fail; publics unreadable | 7gf.105.1.x |
| LM span fusion, kd, refiner | ~0 realistic | lvj, 7gf.130, 7gf.145.1 |
| cross-kind / bigger Deep blends | public .655–.691; d-acd2v = d-ac05xfam to the 5th digit with 58% of top-10s moved | ugp |

Cross-kind blends on top of ac05x s1-4 (VAL / local test / public): +d2v 5 .6514 / .7286 / .69083, +whisper 3 .6444 /
.7311 / —, all kinds (22) .6478 / .7183 / .68316, +old4 .6488 / .7026 / —, +SupCon 3 .6486 / .7014 / —, 6 seeds
.6467 / .7221 / .68416. All are within noise of the 4-seed family (.6459 / .7289 / .69083), so family composition is
exhausted as a lever.

## Instruments established (reuse, don't rebuild)

- **Input ablation on public:** the public MEG share is .155 vs .104 locally, so the local test under-prices encoder
  levers by ~1.5× ([input ablation](2026-10-12_input_ablation_public.md)).
- **Information budget, signal ceiling, representation audit, per-word scaling:** the `deep/2026-10-12_*` docs.
- **Public noise:** family-draw SD is ~.015–.02 and a single member's ~.04 (megspeech-hut). A public delta under
  ~.03 between families is unreadable. Select by public within the top family; use the local test to veto drops.
- **Local groupings:** sentence and kaggle disagree on span and timing changes. Report both.
- **Purity → gain curve:** +.016 local test per +.1 k=5 same-word purity, with a ceiling of +.155. This is the
  price list for any retrieval embedding.

## Remaining levers, ranked by plausible size

1. **Same-word SupCon embedding (7gf.86): failed its gate (10-13 16:46); weight 1.0 also failed (d6r.9, purity +.004, plain LT −.019).**
   - SupCon .3 in the ft stage, tap context: macro k=5 purity +.015 (to ~.09; the gate needed Q2–Q3 ≥ .15).
   - Pool-smoothed local test −.006; plain local test −.002. With `--sampler.balance 1` plain local test was −.066.
   - Not ruled out: SupCon from the stem stage, or a much larger weight. The purity curve prices neither above the
     bar before the cutoff, and smoothing on Kaggle would be the owner's call anyway.
2. **No-next-gap rows (25.8% of Kaggle Deep).** Deferred: KAGGLE grouping is already the instrument (7gf.105.1.3).
   - Its tail matches Kaggle: q.1/.5/.9 = 1.10/1.30/1.58 s vs 1.18/1.36/1.58 s.
   - Rows without a next gap: 13.6% locally vs 18% of Kaggle's sentence rows.
   - What is left is the 9.6% of rows that are isolated words, all private. They are already routed to the word
     head (7gf.58). Even +.08 bacc on them is only +.008, below the 1.5% bar.
3. **Encoder levers re-priced ×1.5** for public. Only worth it with a mechanism; knobs are exhausted.


## Final selection (10-15/16)

The final ranking uses organiser data with stimuli "entirely different from anything released", plus runnable
checkpoints (`docs/competition/rules.md:27-30`). The public board is therefore not the final test. The local test
(unseen stimuli) is the closer proxy, and the checkpoint we hand over must run on current code.

- **Deep:** #1 d-ac05xfam8, the same recipe at 8 seeds (megspeech-rsk). Its VAL / local test .6497 / .7192 against
  the 4-seed family's .6459 / .7289 is a tie inside the local test's ~.01 SE; it met its prereg (VAL ≥ .6359, local
  test ≥ .7189). Eight seeds were bought for variance: on new stimuli the final score depends less on which seeds were
  drawn, and the mean is unchanged. Its public is pending (10-14 slot). The 4-seed family's public (.69083) ties d-old4
  (.69133), a gap of .0005 inside the .015–.02 draw SD, and its local test is +.033 higher (.7289 vs .6955), well beyond
  the local test's SE. #2 d-old4 is the hedge for the top public score. All members rebuild from their recorded
  configs on current code (7gf.29.1.1); only retraining d-old4 would need the 10-07 code.
- **Broad:** hedge across the balancing axis, where ruler and public disagree. #1 b-xc5k25b05ps (Sinkhorn toward .05:
  ruler .6471, public .62914): public falls monotonically with dose and cliffs at .5, so the organisers' marginals
  sit away from the ruler's; .05 keeps the public within .002 of none and takes +.008 ruler (prereg met, 10-13).
  #2 b-xc5k25b0ps (public .63097, ruler .6390): no transduction. The two differ on 37% of rows. b-xc5k25ps
  (toward .25, ruler .6591, public .62587) is dropped: its +.012 ruler is below the bar and sits nearer the cliff.
  None uses the cross-track mapping. Disclose the Sinkhorn with #1's checkpoints.
- The rationale is in 7gf.29.1. The pick itself is the owner's action in the Kaggle web UI, and it is required: unticked, Kaggle scores the best-public submission (host, topic 748465; `docs/competition/announcements.md`).
