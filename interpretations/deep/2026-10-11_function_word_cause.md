# Why function words are confused: the word head cannot tell `the` from `a`, and context fails on surprising or "thee" occurrences (megspeech-7gf.106)

Study: `megspeech/evaluation/studies/cause.py`. It extends perword's `Scene` with the word head, span ids and the spoken
words. Run `python -m megspeech.evaluation.studies.cause m11xsp7hub6d2v3` (CPU, no training, about 3 minutes). Output
goes to `models/m11xsp7hub6d2v3/cause.json`. Follows `2026-10-10_per_word_decodability.md`, which left stress and LM
surprisal unmeasured.

## Setup (what the numbers are)
- **Words.** The same LOCAL TEST unseen-span words as perword: Sherlock1 1358 + TheMoth 775 = **2133**.
  - Every one matched its forced-alignment row: same run, onset within 5 ms, same spelling. 0 unmatched, so every
    regression has n = 2133.
  - `the` n = 265, `a` n = 133.
- **Heads.** Both heads come from blend `m11xsp7hub6d2v3` (27 members, geometric mean of the members' log-probs).
  - CONTEXT head (`logits`): plain top-10 0.811, top-1 0.345.
  - WORD head (`word_logits`: the word's own window, no context): top-10 0.631, top-1 0.189.
  - Each head is finished at its own tau, fit on `val` with the tau3 arm. Both came out at tau 0.5.
- **Per-occurrence features.**
  - **surprisal:** GPT-2 large, in bits, summed over the word's tokens. Context is every earlier word of the same
    released stimulus run, taken from its alignment transcript (up to 1024 tokens).
  - **log_duration:** the aligned word duration.
  - **log_prior:** log training prior of the word.
  - **first / last:** position in the span.
  - **stressed:** lexical stress (see below).
  - Continuous columns are z-scored, so a coefficient is log-odds per SD. Binary columns are per unit.
- **Stress (CMUdict 1.1.3, GPL-3.0, an OSI licence).** The aligner's phonemes carry no stress digit.
  - Each occurrence takes the CMUdict variant whose bare phonemes equal the aligned ones.
  - If several variants match (`the` as DH AH0 or DH AH1), or none does, a function word takes its least-stressed
    variant (its reduced form, as in connected speech) and a content word its most-stressed.
  - `stressed` = primary stress.
  - Consequences: `the` is always unstressed, since "thee" is DH IY0. `a` is stressed only when realised EY (4 of 133).
  - Within a word, `stressed` varies only for function words with an aligned full-vowel variant, so its word-fixed
    coefficient rests on few occurrences.
  - `schwa` = the aligned vowels include AH. For `the`: dh ah vs dh iy, the "thee" form. For `a`: ah vs ey.
- **Model and CI.**
  - Logistic regression with weak L2 (C = 10). The fixed-effect spec adds one dummy per word. It cannot estimate
    log_prior, which is a word property.
  - 95% percentile CIs come from 200 bootstrap resamples of whole spans, since words in one span share MEG and context.
  - "Clear" below means the CI excludes 0. With 4 specs x 2 heads x 6 coefficients, a few such results are expected by
    chance, so a single clear coefficient is **directional** unless it is large or replicates across specs.

## Coefficients (n = 2133 in every fit; log-odds per SD, [95% CI])
| feature | context top-10 pooled | context top-10 word-FE | word-head top-10 pooled | word-head top-10 word-FE |
|---|---|---|---|---|
| surprisal | -0.18 [-0.30, -0.06] | -0.24 [-0.38, -0.11] | -0.20 [-0.32, -0.09] | -0.19 [-0.34, -0.08] |
| log_duration | -0.06 [-0.20, 0.03] | -0.04 [-0.22, 0.12] | 0.10 [-0.02, 0.20] | **0.19 [0.07, 0.34]** |
| stressed | -0.02 [-0.27, 0.27] | -0.23 [-0.75, 0.33] | -0.11 [-0.31, 0.12] | -0.60 [-1.04, -0.12] |
| log_prior | 0.38 [0.27, 0.50] | — | **1.00 [0.85, 1.16]** | — |
| first | 0.14 [-0.15, 0.47] | 0.12 [-0.16, 0.46] | **0.44 [0.22, 0.72]** | 0.40 [0.15, 0.75] |
| last | **-0.72 [-1.01, -0.36]** | -0.73 [-1.09, -0.30] | 0.17 [-0.16, 0.43] | 0.10 [-0.26, 0.41] |

| feature | context top-1 pooled | context top-1 word-FE | word-head top-1 pooled | word-head top-1 word-FE |
|---|---|---|---|---|
| surprisal | -0.16 [-0.28, -0.05] | -0.15 [-0.28, -0.05] | -0.25 [-0.41, -0.08] | -0.19 [-0.34, -0.02] |
| log_duration | -0.05 [-0.15, 0.06] | 0.00 [-0.13, 0.12] | 0.23 [0.08, 0.38] | 0.13 [-0.02, 0.31] |
| stressed | -0.05 [-0.27, 0.17] | -0.09 [-0.50, 0.30] | 0.20 [-0.04, 0.47] | 0.03 [-0.33, 0.39] |
| log_prior | 0.70 [0.56, 0.83] | — | 1.52 [1.31, 1.81] | — |
| first | 0.16 [-0.08, 0.42] | 0.06 [-0.18, 0.32] | 0.75 [0.48, 1.09] | 0.35 [0.05, 0.66] |
| last | -0.65 [-0.97, -0.29] | -0.74 [-1.12, -0.40] | 0.15 [-0.28, 0.51] | -0.02 [-0.51, 0.40] |

What holds across specs and outcomes:
- **Surprisal hurts both heads about equally**, -0.15 to -0.25 per SD in all 8 fits, including within word. The
  word head sees no context, yet it is still worse on surprising occurrences, so this is not context leaking in.
  - Candidate mechanisms, none tested: surprising words are articulated differently, or they sit after disfluent or
    unusual context that shifts the window's alignment.
  - Unexplained, and **directional** in size: about 0.2 log-odds per SD is a few points of hit rate.
- **The word head leans on frequency more than the context head, not less.** log_prior is 1.00 vs 0.38 (top-10)
  and 1.52 vs 0.70 (top-1). The predicted contrast ("word head tracks surprisal/stress, context head tracks
  frequency") **does not appear in that form**.
- **The heads differ on position and duration.**
  - The span-final deficit is the context head's alone: last -0.72, word head ~0. The context head has no right
    context there; the word head's evidence is unaffected.
  - The word head instead gains on span-first words (+0.44 top-10). Those follow silence, which leaves a cleaner
    onset response; this is a mechanism guess.
  - Longer occurrences help only the word head (+0.19 within word, top-10), which fits acoustic evidence driving it.
- **Stress has no reliable effect.** The one clear value, word head top-10 word-FE -0.60, rests on the few
  within-word variants, has no counterpart at top-1, and changes sign in other specs. It is **directional / likely
  noise**.

## `the` vs `a` (n: the 265, a 133; "an" is a separate word and is excluded)
Each cell gives top-1 rate and mean P(the) - P(a), as context / word head. Tertile cells use within-word tertiles.
Durations are quantised to alignment frames, so the tertiles of `a` are uneven.

| cell | a: n | a: top-1 ctx / word | a: margin ctx / word | the: n | the: top-1 ctx / word | the: margin ctx / word |
|---|---|---|---|---|---|---|
| all | 133 | 0.66 / 0.23 | -0.26 / +0.008 | 265 | 0.59 / 0.39 | +0.15 / +0.019 |
| surprisal low | 45 | 0.73 / 0.33 | -0.33 / +0.006 | 89 | 0.66 / 0.44 | +0.20 / +0.024 |
| surprisal mid | 44 | 0.73 / 0.23 | -0.29 / +0.009 | 88 | 0.56 / 0.42 | +0.13 / +0.022 |
| surprisal high | 44 | 0.52 / 0.14 | -0.16 / +0.009 | 88 | 0.53 / 0.30 | +0.12 / +0.012 |
| duration low | 69 | 0.86 / 0.25 | -0.46 / +0.001 | 103 | 0.52 / 0.36 | +0.07 / +0.014 |
| duration mid | 22 | 0.77 / 0.18 | -0.22 / +0.014 | 82 | 0.76 / 0.39 | +0.27 / +0.019 |
| duration high | 42 | 0.29 / 0.24 | +0.05 / +0.016 | 80 | 0.50 / 0.41 | +0.13 / +0.027 |
| schwa yes | 129 | 0.67 / 0.23 | -0.27 / +0.009 | 218 | 0.62 / 0.40 | +0.16 / +0.021 |
| schwa no (a = ey; the = "thee") | 4 | 0.25 / 0.25 | -0.01 / -0.008 | 47 | 0.45 / 0.32 | +0.09 / +0.011 |
| next word vowel-initial | 0 | — | — | 34 | 0.32 / 0.29 | +0.07 / +0.009 |
| next word consonant-initial | 133 | 0.66 / 0.23 | -0.26 / +0.008 | 231 | 0.62 / 0.40 | +0.16 / +0.021 |

Fixed-effect logistic fit of top-1 within the/a (n = 398):
- context head: surprisal -0.27 [-0.52, -0.07], next_vowel **-1.20 [-2.45, -0.20]**, duration and schwa not clear.
- word head: surprisal -0.37 [-0.71, -0.13], duration **+0.39 [0.05, 0.73]**, schwa and next_vowel not clear.

Reading:
- **The word head barely separates `the` from `a`.** Its mean margin is +0.019 on `the` and +0.008 on `a`. Both are
  positive and about 0.01 apart, while the context head's margins are +0.15 vs -0.26. Context and prior carry nearly
  all of the the/a decision.
  - In that sense, function-word confusion is **target-limited**: the per-window MEG evidence for which article was
    heard is close to nil.
  - The 7gf.71/.102 timing control agrees (other model, `sent40-ioi-ema9998-s0`, official-val unseen words). With
    MEG permuted and only word timing left, recall is `the` 0.507 (n=349) vs 0.527 with real MEG, and `a` 0.618
    (n=144) vs 0.646. Timing alone carries all but 2-3 points of article recall; those 2-3 points are
    **directional**.
- **Where the context head fails on articles:**
  - Surprising occurrences: `a` top-1 0.73 to 0.52 from low to high tertile; `the` 0.66 to 0.53.
  - `the` before a vowel-initial word, which is mostly the "thee" form: 0.32 vs 0.62 top-1, n = 34. Context there
    does not favour `the` over `a`, though a vowel-initial next word should exclude `a` (that would be `an`). The
    context head has not learnt the a/an rule from the next word. **Directional** (n = 34, CI wide), but a concrete
    target.
  - Long `a` occurrences: top-1 0.29, margin +0.05, n = 42. These are slow or emphasised "a", likely mis-fit as
    `the`. Directional.

## What this does NOT rule out
- The word head's articles could be decodable with a different target (phoneme-level: dh vs vowel onset) or a
  longer or earlier window. This study only shows that the current word head does not separate them.
- Surprisal comes from GPT-2 large over the stimulus transcript. A smaller or different LM could rank occurrences
  differently. The effect's size is small, and its cause is open.
- Single blend, single local-test set: Sherlock1 run 2 and TheMoth run 1, two stimuli.
