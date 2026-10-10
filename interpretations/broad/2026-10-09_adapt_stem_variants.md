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

## Update 2026-10-10: subjects 1–32 ses-12 in the stem hurts

`ema-bgmix05-all32-w15-s0` adds labelled subjects 1–32 ses-12 to the pooled stem (`--data.adapt 12`), then
adapts with the usual recipe. Broad ruler (ses-11), single seed:

| model | ruler |
|---|---|
| all-w15-s0 adapt (base) | .6003 |
| all32-w15-s0 stem, unadapted | .5823 |
| all32-w15-s0 adapt | .5712 |

Adding the data to the stem costs .029 after adapt, and adapting makes the all32 stem worse rather than better.
Inference, not tested: ses-12 is one text that 32 subjects heard, so the stem's context head memorises it, and
adapt on the same text doubles that exposure, while the ruler reads a new text. This is the same direction as
the ended-intervals and cx0 results. Seed s1 was stopped. Not ruled out: ses-12 used at adapt only, which is
the base recipe already.

## Update 2026-10-10: subject matrices anchored toward their mean at adapt

`--optim.anchor a --optim.anchor_toward mean` leaves share `a` of each subject matrix's deviation from the mean
of all trained subjects' matrices by the end of adapt; the base decays toward 0 (`a = 1`). The stem is
all-w15-s0 and the seed is single.

| adapt | Broad ruler |
|---|---|
| base (`a = 1`, toward 0) | .6003 |
| `a = .5` toward mean | .5925 |
| `a = .25` toward mean | .5909 |

- Both anchored adapts sit about .008 under base, which is twice the seed SD of .004.
- Halving the anchor again changes nothing. Pulling subject matrices together costs a little and the strength
  barely matters, so subjects 1–32 do not suffer from too-free matrices at 12 sessions of adapt.
- **Not ruled out:**
  - Anchoring at stem time.
  - Anchoring toward a per-source mean, since subjects 1–12, 13–22 and 23–32 come from different sources.
  - Using the anchor only for subjects with few words; subjects 23–32 score .594 against .661 for 1–12, on
    310 vs 1296 ruler words each.
