# Broad adapt: why pooled all-w15 stems win and the variants lose (2026-10-09)

**Scope.**

- **Source.** Saved `*-adapt12-mask-ep40-broad/metrics.json`. The stem configs were diffed against
  `ema-bgmix05-all-w15-s0`.
- **Ruler.** The Broad ruler: n=25322 words, top-10 balanced accuracy, tau .5 best for every model.
- **Seeds.** Single seed per variant unless stated. The seed spread comes from s0/s1/s2.
- **Training.** No training was run for this note.

| adapt stem | stem change vs all-w15-s0 | Broad ruler | Δ vs 3-seed mean |
|---|---|---|---|
| all-w15-s0 / s1 / s2 | — | .6003 / .5973 / .5915 (mean .5964, SD .004) | 0 |
| all-w15**l60**-s0 | 60 epochs, last epoch | .6019 | +.006 |
| all-w15-**cx0**-s0 | context transformer trained on subject 0 only | .5798 | **−.017** |
| all-w15-**inone**-s0 | no IOI-interval inputs to the context (`intervals: none`) | .5582 | **−.038** |
| all-w15-**j10**-s0 | 0.1 s onset jitter on context inputs | .5371 | **−.059** |
| Deep xsp10sem05-s4-ft-s5 | 1.0 s window, subject 0 only, no bgmix, cdrop .25 | .5054 | **−.091** |

The pooled stem's `data.subjects` are subject 0 plus the external corpora (ids 101–127, 201–212 and
301–323), with bgmix .5.

**Interpretation (inferences; one seed each, except the reference).**

1. **Timing inputs carry the Broad context head.**
   - Removing IOIs costs 3.8 points. Blurring onset timing by 0.1 s costs 5.9.
   - This matches the Deep attribution probe, where timing alone (.437) beats the MEG embedding (.315).
     Much of the context head's lift is rhythm and duration over the span.
   - Jitter is the bigger loss because it corrupts both IOIs and the word-window alignment.
2. **The context transformer must see many subjects.** cx0 (−1.7) trains the span model on subject 0's
   spans only. Adapt then moves it to subjects 1–32 from fewer examples.
3. **The Deep xsp stem is the worst adapt start (−9.1).** It differs on four axes at once: window, data
   pool, bgmix and dropout. The fitness-to-subject-0 story in 7gf.132 is plausible but not isolated.
   - Stem geometry supports it. Measured on subject 0, the pooled stem's embedding has a larger session
     offset and lower rank (see `../deep/2026-10-09_ovmi_and_geometry.md`).
   - Its cross-subject word alignment was only measured for the pooled stem (centroid cos .23).
4. **l60 (+.006) is noise-sized** against a seed SD of .004. It is directional, not a finding.

**Does NOT rule out.**

- The Deep xsp deficit could be mostly the 1.0 s window or the missing bgmix, rather than subject-0 fit. No
  single-axis ablation exists.
- jitter or inone could help on the final test's unseen stimuli if timing is a stimulus-specific shortcut.
  The ruler shares stimuli statistics with training, so it would reward the shortcut.

## Next experiments (ranked by plausible gain)

1. **Richer timing to the context head**, such as word duration and syllable-rate proxies from onsets.
   - Rationale: removing IOIs costs 3.8 points, so the timing channel is high-leverage and plausibly
     unsaturated.
   - Cost: one stem plus adapt run.
   - Plausible gain: +1–2 points.
2. **A one-axis ablation of the xsp adapt gap**: a pooled stem with a 1.0 s window, or a Deep stem with
   bgmix.
   - This is an instrument. It decides whether Deep-side improvements (sem05 aux) can transfer to Broad.
3. **Per-subject test-time centring of embeddings** before adapt and readout.
   - Rationale: the session/subject offset is the dominant variance (domain AUC 1.0, mean shift 0.8 of the
     spread).
   - Cheap to test. Plausible gain: +0–2 points.
