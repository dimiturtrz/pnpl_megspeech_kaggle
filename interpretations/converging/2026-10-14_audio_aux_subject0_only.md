# Audio aux lifts subject 0, not the subjects Broad is scored on

Audio-aux stems (frozen wav2vec2-xlsr layers 15-18 per word, sem .5, broadband log-energy .5) on the all-data
Broad recipe, two seeds, paired against the same-seed base:

| seed | subj-0 local test | Broad ruler (subj 1-32, n=25,322) | Broad public |
|---|---|---|---|
| s0 | +.070 | −.008 (.6136 vs .6214) | −.015 (.60508 vs .61959) |
| s1 | +.052 | −.003 (.6042 vs .6074) | +.003 (.60728 vs .60403) |

- The mechanism fired: the subject-0 gain sits on TheMoth sentences training never heard (+.061 by the
  memorisation tally), so it is not sentence recall.
- It does not cross subjects. The Broad ruler (Spearman .91 vs public, `2026-10-10_selector_transfer.md`)
  called the public sign on both seeds; the subject-0 local test called it wrong on both.
- Deep's acoustic aux likewise tied on public (.69083 vs .69133).

Harness lesson, not an idea verdict: Broad changes are gated on the Broad ruler line (printed by
`megspeech.submission` without `--local`), never on `--local`. Not ruled out: aux in a subject-specific
adapt stage, or with a per-subject target alignment.
