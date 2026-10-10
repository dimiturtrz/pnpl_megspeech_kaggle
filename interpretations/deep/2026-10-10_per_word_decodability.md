# Per-word decodability of the best Deep blend: the loss is function-word confusion and span-final words (megspeech-7gf.106, megspeech-7gf.102)

Study: `megspeech/evaluation/studies/perword.py`; run `python -m megspeech.evaluation.studies.perword m11xsp7hub6d2v3`
(CPU, no training; full per-word table in `models/m11xsp7hub6d2v3/perword.json`). Issues: megspeech-7gf.106, .102.

## Setup (what the numbers are)
- **Model.** The 27 members of blend `m11xsp7hub6d2v3`, context head, log-probabilities averaged (the blend's geometric
  mean), finished at tau 0.5 (the tau3 arm of `crowding`, fit on `val`). Read from each member's saved `probs.pt`.
- **Words.** LOCAL TEST (`local-test` section) unseen-span words in the domains the ruler counts: Sherlock1 1358 +
  TheMoth 775 = **2133** (MOCHATIMIT has 36 unseen words, under `MIN_NOVEL` = 100, so it is not scored).
- **Ruler.** The selection ruler (`Scored.macro`): balanced top-10 accuracy per domain, mean over domains.
  Local-test macro **0.697** (Sherlock1 0.738, TheMoth 0.656). The plain (word-weighted) top-10 hit rate is **0.811**
  and top-1 is 0.345; the 11-point gap is the metric's balance: frequent words are easy, rare ones are not.
- **Not measured here.** LM surprisal and lexical stress of 7gf.106 (no GPT-2, no stress table in this run); word
  head; acoustic duration is a per-word-type mean from the forced alignment, not per occurrence.

## Per word (50 words; n = local-test unseen-span words, median 26, 14 words have n < 10)
- **Best** (hit10, n): a 0.970 (133), the 0.913 (265), and 0.912 (249), i 0.910 (78), of 0.865 (133), to 0.847 (124).
- **Worst with n >= 18**: be 0.444 (27), not 0.500 (18), they 0.528 (36), is 0.556 (18), on 0.605 (38), were 0.609 (23),
  have 0.615 (26), there 0.618 (34). `has` is 0/9 (8 of them TheMoth) and `good` 0/1.
- The best words are also the most predicted: the/and/a/i/of/to take 14.5/13.3/7.5/7.0/6.4/6.4% of top-1 calls against
  12.4/11.7/6.2/3.7/6.2/5.8% of true labels (`i` is over-called 1.9x).
- **Source.** A word's two sources differ more than the sources' macros do: `he` 0.807 Sherlock1 (n=114) vs 0.308
  TheMoth (n=13); `is` 0.857 (7) vs 0.364 (11); `it` 0.882 (34) vs 0.500 (14). Per-source cells are mostly n < 15, so
  these are directional only.
- **Frequency / duration / count.** Spearman of word hit rate (48 words with all fields) vs log training prior +0.09,
  vs acoustic duration +0.09, vs mean onset-to-next-onset +0.12, vs n +0.15. None is distinguishable from 0 at this
  n; hit rate is *not* explained by frequency or duration once the metric is class-balanced.
- **Function vs content** (closed-class spaCy tags: ADP AUX CCONJ DET PART PRON SCONJ vs the rest): mean per-word
  hit10 0.719 over 40 function words vs 0.782 over 10 content words. Only 10 content words exist in the 50, so this is
  a 6-point difference on a base of 10 words: not a finding.

## Confusions (top-1 errors: 1397 of 2133 words)
- Most frequent (true -> predicted, count): the->a 38, a->the 25, and->i 25, to->the 19, he->the 15, the->to 14,
  the->i 14, he->and 13, of->to 13, of->in 12. Every top-15 pair is function word to function word.
- **Same part of speech.** 23.3% of errors land on a word of the true word's tag; 11.8% is expected if the predicted
  word were drawn from the errors' own predicted marginal (never the true word). Ratio 2.0x, so errors are
  syntactically structured, though 77% still cross tags.
- **Same first phoneme:** 5.2% observed vs 6.0% expected; **first two phonemes:** 0.8% vs 0.8%. No phonological
  clustering at all, consistent with 7gf.102's earlier finding that confusion does not track word shape.

## Position in span and interval (hit rate; excess = hit minus the word's own overall rate, which removes word identity)
| bucket | n | hit10 | excess |
|---|---|---|---|
| first word | 404 | 0.829 | +0.008 |
| middle | 1516 | 0.823 | +0.011 |
| **last word** | 213 | **0.695** | **-0.093** |
| gap to next onset <= 0.10 s | 481 | 0.906 | +0.037 |
| 0.10-0.13 s | 517 | 0.832 | +0.026 |
| 0.13-0.17 s | 463 | 0.806 | +0.009 |
| > 0.17 s | 459 | 0.747 | -0.034 |
| gap from previous onset <= 0.14 s | 438 | 0.795 | -0.013 |
| 0.14-0.23 s | 429 | 0.758 | -0.047 |
| 0.23-0.41 s | 432 | 0.815 | +0.008 |
| > 0.41 s | 430 | 0.860 | +0.044 |

(Interval buckets are quartiles of the scored words; SE of one bucket's hit rate ~0.02, and words within a sentence
are correlated, so the SE is understated.)

## Reading
- The remaining loss is **concentrated in the few frequent closed-class words the model swaps among themselves**
  (the/a/and/i/to/of/in/he), not in rare words and not in phonologically close pairs. That fits the known crowding of
  frequent classes in the top 10, and the 2x same-tag excess shows the swap is partly grammatical, which a context
  signal could in principle exploit.
- The largest within-word effect is **position**: the span's last word is 9 points below its own average (n=213,
  about 4.5 SE ignoring clustering). A last word has no right context in the span; this points at right-context
  availability as a lever (the `augment.right_cut` and span-length work), not at the word's identity.
- Short onset-to-next-onset words (fast speech, rapidly followed) decode better; words after a long pause decode
  better. These are within-word effects of 3-7 points, directional.
- What this does NOT show: a mechanism for the interval effects (timing could carry word identity, 7gf.71), nor that
  function words are a "wrong target" for MEG: frequency and duration do not predict a word's hit rate here.
