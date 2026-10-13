# Transductive pool smoothing — the ceiling is real, the neighbours are not pure enough

Beads: megspeech-d6r.8 (closed), continues as megspeech-7gf.86. Code: `megspeech.evaluation.studies.poolsmooth`
(rounds 1–2), `megspeech.evaluation.studies.poolgraph` (round 3), `poolconfuser`.
Base: `ema-cd25-xsp10sem05-ac05x-s1-ft-s1`, reproduced exactly (VAL .6386, local test .7088).

## Question

Leaders average same-word repeats (d'Ascoli / SimpleB2T). Kaggle gives no labels, but rows of one scored pool that
are the same word should agree. So: mix each row's scores with its `k` nearest other rows (cosine on the embedding
at the context head's input, or on the posterior), with weights `softmax(cos / temp)`. Every knob is chosen on VAL
pools and read on the local test, using Kaggle-sized pools (960 rows × 30 draws). The rules do not address
test-time adaptation (`docs/competition/rules.md:35`). Not fitting on test MEG is our own restriction, so
using this on Kaggle is the owner's call.

## Result

| arm (VAL-chosen) | local test macro gain | neighbour purity k=5 |
|---|---|---|
| embedding, centred, k5 temp .1, prob mix lam .1 | +.0034 ± .0021 | .20–.26 |
| posterior, centred, k5 temp .05, prob mix lam .1 | +.0064 | .22–.25 |
| mutual kNN, posterior | +.0058 | .242 |
| two views (embedding + posterior) | −.0038 | — |
| propagation, two views, 2 rounds | **+.0080** | .234 |
| fold-ensemble graph + propagation | +.0097 (VAL leaked) | — |

The bar was +.015, so the best non-leaked arm misses it. Re-tuning tau under smoothing gave no consistent gain.

## The instrument that matters: purity → gain

The oracle-label ceiling uses the best mix chosen on test, so it is not a result. It measures how much exact
same-word neighbours would pay: k=1 +.090, k=5 +.134, k=10 +.155. Each row has 50–61 same-label rows in its pool.

The purity-matched curve (k=5, same-label share p): p .2 → +.008, p .4 → +.036, p .6 → +.066. That works out to
about **+.016 local test per +.1 purity**. The measured embedding sits at p ≈ .20–.24 (chance is .065), which is
exactly where the curve says +.003 to +.008. The harness is consistent with itself: the poor arm results come from
the low purity they measured, not from a bad mix.

## Why purity is low (what it is not)

- **Not adjacent-word confusion.** Same-span neighbours make up 0.1–0.2% of k=5 neighbours, below chance (.24–.28%).
  Excluding them changes nothing.
- **Not the mixer.** A log-additive mix was never selected, and selection went to the smallest lam. Smoothing
  through the graph (propagation, mutual kNN) moves purity by about .02.
- **Read:** a CE-trained classifier input only has to separate words **linearly**. Nearest neighbours in that space
  are dominated by sentence context and timing, not word identity. Retrieval was never part of the training
  objective.

## Purity is a frequent-word number (13:52 addendum)

Row-mean purity weights rows, and pools follow corpus frequencies (chance .065, not 1/50). BAcc weights labels.
`Smoothing.by_label` splits the base's k=5 purity by training-frequency quartile of the label:

| quartile (rare → frequent) | rows / label / pool | max k=5 purity (pool cap) | VAL purity | LT purity |
|---|---|---|---|---|
| Q1 | 2.3–2.7 | .26–.33 | .02–.03 | .04–.06 |
| Q2 | 8.1–8.6 | .71–.84 | .05–.06 | .05 |
| Q3 | 11.4–12.4 | .99–1 | .05 | .06 |
| Q4 | 45–52 | 1 | .18–.19 | .16–.17 |

- The quoted .20–.26 is carried by Q4.
- On the macro axis the metric scores, the embedding is near chance for three quarters of the vocabulary.
- **The headroom is Q2–Q3:** pools hold enough same-word rows there, and the embedding does not find them.
- Q1 is capped by the pool itself, however good the embedding.

### Consequence for 7gf.86

- The base config draws 64 sentences, about 330 labelled words per batch. The rarest 12 labels get a SupCon
  positive in only 56% of batches (simulated), and they make up 2.3% of the anchors that have one.
- `--sampler.balance 1` lifts that to 88% and 7.3%. It was a null for CE (megspeech-3sk), but its job here is
  different: it feeds positives to the anchors that need them.
- The prereg gate is therefore macro: Q2–Q3 purity, not the row mean.
- **The tap stays `context`.** The encoder's word embedding, read before the span transformer, is worse in every
  quartile. Its macro purity is .03 against .08 at the context head's input, so span context adds same-word
  information rather than diluting it.

## Verdict and scope

Dead for this use: retrieval from a CE-trained embedding or posterior, under any graph trick tried. **Not ruled
out:** an embedding trained for retrieval. Same-word SupCon is 7gf.86, now run-ready (`--optim.contrast` with
`--optim.tap context`, ft stage). If it reaches purity ≥ .30, the curve prices it at about +.016–.03 local test,
and the encoder levers' public multiplier (~1.5×, [input ablation](../converging/2026-10-12_input_ablation_public.md)) applies on top.
