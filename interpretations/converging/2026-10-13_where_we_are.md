# Where we are — 2026-10-13

Supersedes [2026-10-07](2026-10-07_where_we_are.md). Goal epic megspeech-d6r: Deep public ≥ .705, Broad ≥ .620.
Levers close 2026-10-14 23:00 Sofia; 10-15/16 is final selection only (7gf.29.1). Kaggle closes 2026-10-16 12:00 UTC.

## Scores

| track | best public | submission | 10-07 best | leader (10-11) |
|---|---|---|---|---|
| Deep | **.69133** | d-old4 (xsp10sem05-ft, 4 seeds) | .6638 | .77474 |
| Broad | **.62587** | b-xc5k25ps (xc5w6all6 29 members + mean fallback keep .25 + per-subject Sinkhorn .25) | .5879 | .73235 |

Broad met its goal. Deep did not: no public gain since d-old4 (10-10). Recipe families score .685–.691
(d-ac05xfam .691, m11 .689, d-old9 .685), so that is this recipe's public ceiling, and seeds or family size move it
only within draw noise.

## What moved 10-07 → 10-13

- **Broad +.038:**
  - Zero-shot rows read through a MEAN subject fallback with keep .25 instead of IDENTITY: +.020 single-model ruler.
  - Transductive Sinkhorn balancing on output probabilities, per subject: +.011.
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
| timing expert fusion (incl. LibriSpeech pretrain) | LT −.0095 at VAL weight | r9m.1 → [timing](../deep/2026-10-13_timing_and_span_cuts.md) |
| nd / ended / noint fusion | parked: ac05x 4-seed LT fail; publics unreadable | 7gf.105.1.x |
| LM span fusion, kd, refiner | ~0 realistic | lvj, 7gf.130, 7gf.145.1 |

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

1. **Same-word SupCon embedding (7gf.86).**
   - The only lever with a measured, large ceiling behind it (the pool curve).
   - Run-ready: an ft-stage arm, one knob vs the base; commands are in the bead.
   - Prereg: purity ≥ .30 and pool-smoothed local test ≥ +.015.
   - If it passes, using test-row smoothing on Kaggle is **the owner's call** (it falls under our own
     no-test-fitting rule, not theirs).
2. **No-next-gap rows (25.8% of Kaggle Deep).** Deferred: KAGGLE grouping is already the instrument (7gf.105.1.3).
   - Its tail matches Kaggle: q.1/.5/.9 = 1.10/1.30/1.58 s vs 1.18/1.36/1.58 s.
   - Rows without a next gap: 13.6% locally vs 18% of Kaggle's sentence rows.
   - What is left is the 9.6% of rows that are isolated words, all private. They are already routed to the word
     head (7gf.58). Even +.08 bacc on them is only +.008, below the 1.5% bar.
3. **Encoder levers re-priced ×1.5** for public. Only worth it with a mechanism; knobs are exhausted.

The GPU is lent to rsna_knee (since 10-13 08:43 Sofia), so all megspeech work is CPU until it comes back. Item 1 is
the first GPU job when it does.

## Final selection (10-15/16)

The final ranking uses organiser data with stimuli "entirely different from anything released", plus runnable
checkpoints (`docs/competition/rules.md:27-30`). The public board is therefore not the final test. The local test
(unseen stimuli) is the closer proxy, and the checkpoint we hand over must run on current code.

- **Deep:** #1 d-ac05xfam. Its public (.69083) ties d-old4 (.69133), a gap of .0005 inside the .015–.02 draw SD.
  Its local test is +.033 higher (.7289 vs .6955), well beyond the local test's ~.01 SE, and it runs on current code.
  #2 d-old4 is the hedge for the top public score. Re-open #1 if 7gf.86 lands.
- **Broad:** #1 b-xc5k25ps (.62587). #2 b-xc8k25ps (.62557), the same recipe with 3 more seeds. The top six are all
  within .002, so they tie. None uses the cross-track mapping. The per-subject Sinkhorn is transductive; disclose it
  with the checkpoints.
- The rationale is in 7gf.29.1. The pick itself is the owner's action in the Kaggle web UI.
