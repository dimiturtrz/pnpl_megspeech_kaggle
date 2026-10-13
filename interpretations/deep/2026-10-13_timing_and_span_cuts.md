# Timing levers 10-12 → 10-13: experts don't fuse, span cuts don't transfer

Beads: megspeech-r9m, r9m.1 (timing expert + LibriSpeech pretrain), 7gf.105.1.1 (MEG-only fusion),
7gf.105.1.3 (intervals ended), the nd notes in 7gf.105.1.1, megspeech-hut (public noise).
Base everywhere: `ema-cd25-xsp10sem05-ac05x-s1-ft-s1` (VAL .6386, local test .7088) unless named.

## Why timing was the target

The information budget puts timing + sequence at ~70–77% of the blend's information
([information budget](2026-10-12_information_budget.md)). Next gap alone carries .18 local bacc. And 25.8% of Kaggle
Deep rows have **no next gap**: 156 epoch-last rows + 92 isolated-word rows, 9.6% of all rows. Two lever families
followed: a stronger timing channel, and robustness to the missing next gap.

## 1. Timing expert, late-fused — negative on the local test at every weight

| expert | expert VAL | fused VAL (VAL-chosen w) | fused local test |
|---|---|---|---|
| local TimingHead (14 feats, ±2 nbrs) | .5605 | .6465 | .7013 (w .1), lower above |
| span-wide TimingSequence (L2 transformer) | .5894 | .6510 | .6915–.7010, all < base |
| TimingSequence pretrained on 1.33M LibriSpeech spans | .5950 | .6528 (w .2) | .6993 (−.0095) |

- The **learning-curve gate** behaved as a mechanism should. The local head is flat from 25% of the data upward.
  The span-wide model gains about +.01–.015 per doubling, which licensed the LibriSpeech corpus. LibriSpeech alone
  gives .5254 zero-shot on MEG-session timing, so the pretrain is real.
- **Fusion fails anyway.** TheMoth drives it: VAL .551 → .587 rises, while local-test TheMoth is already .686 and
  falls. The context model already reads the same onsets, so a late-fused expert counts that information twice.
- **Not ruled out:** timing as an *input* to context-model pretraining, instead of a late-fused expert. The
  `--pretrain` code is merged and opt-in.

## 2. Train-time span cuts (nd = NextDrop, ended, rc) — local gains, no public read up

- **nd25 on the non-acoustic family is a real local effect.** It was paired over 6 seeds and LT rose under every
  grouping: sentence +.009, kaggle +.020, spec +.004. The thin proxy (eval-time next-gap drop, p .26) made it look
  larger still.
- **nd25 on the adopted ac05x recipe fails.** Over 4 paired seeds: sentence −.009 (0/4 up), kaggle +.001. The
  suspect is untested: the acoustic stem aux may already supply the envelope / next-gap cue that nd teaches.
- **Public reads:**
  - Single members: d-endnd25 .640, d-ndstem .664, d-fusenoint73 .661, against d-ablintact .671.
  - Family: d-nd6fam .666.
  - Broad: b-nd25k25 .579 vs .607, and b-fusenoint73 .598 vs .607.
  - Not one of them points up.

### Public reads are noisier than the preregs assumed

The megspeech-hut bisect shows no code drift, only an RNG-stream shift. Same-recipe 4-member families span public
.653–.691, so the family-draw SD is about .015–.02, and a single member's is about .04. Every single-member
prereg bar above sat inside that noise, so those reads are **unreadable, not refutations** (the 05:06 correction in
7gf.105.1.1). The local test is the verdict; public can only veto large drops.

### Local groupings disagree with each other

The SENTENCE grouping (spans of ~7 words) had the wrong sign on ended (+.014 local vs −.031 public). KAGGLE grouping
(spans of 5.56 words plus isolated words) fixed that sign. SPEC grouping had the right sign 2/2, but on only 2
points. No grouping tracks public magnitude. Default for span or timing changes: report both sentence and kaggle.

## Verdict and scope

- **Late timing fusion:** dead for this use (subject-0 expert or LibriSpeech-pretrained; log-linear, VAL weight).
- **nd:** parked. Not adopted on Deep or Broad, but its local gains on the non-acoustic family are not refuted.
- **ended:** does not stack on the nd stem (LT −.017).
- **Still open:** the 25.8% no-next-gap rows remain the one place where local test and Kaggle differ structurally.
  Kaggle's last word always has ≥1.0 s of post-onset MEG (median 1.36 s). Any lever for those rows needs a local
  instrument that reproduces that shape, and the thin proxy does not (it hides the span end, which Kaggle gives).
