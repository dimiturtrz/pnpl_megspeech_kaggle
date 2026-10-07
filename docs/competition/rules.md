# PNPL Competition 2026 — rules

Sources: the Kaggle Rules tabs of `pnpl-competition-2026-deep` and `-broad`, pasted by the owner on 2026-10-07
(the Kaggle CLI cannot fetch them; the two tabs are word-for-word identical), and libribrain.com/editions/2026/rules/ + /faq/ (fetched 2026-10-05). Quotes are verbatim.

## Competition-specific rules (Kaggle Rules tab)

Tracks, both evaluated on word classification (top-10 balanced accuracy over a vocabulary of 50 words):

| | Track 1 — Deep | Track 2 — Broad |
|---|---|---|
| Goal | "Word classification on a single, deeply-sampled subject (subj0)" | "Word classification across 32 held-out subjects (subj1–subj32)" |
| Training data | "Extensive recordings from subj0 across audiobooks, phonetically balanced speech corpora (TIMIT, MOCHA-TIMIT), and narrative podcasts" | "Varying amounts of labelled data per subject" |

Track 2 data groups — "To study how performance scales with the amount of fine-tuning data":

| Group | Subjects | Labelled data per subject |
|---|---|---|
| 100% | 12 | ~40 min of audiobook listening |
| 50% | 10 | ~20 min of audiobook listening |
| 25% | 10 | ~10 min of audiobook listening |

The Overview (`overview.md`) places this data in Sherlock 1 sessions 11 and 12 and adds a 0% group (no listening
data, subject IDs only).

- "You may submit to either or both track."
- "Final rankings are determined on independent holdout data recorded with stimuli that are entirely different
  from anything released to participants."
- "Submissions must beat the reference baselines to be eligible for prizes."
- Prizes: top 3 confirmed teams per track; "Each team may only win prize money once across the entire competition."
- Verification: "all top-ranking teams will be asked to provide runable model checkpoints. We may also ask specific
  teams to provide their training code so we can verify no cheating took place."
- "We strongly encourage all participants to share their code openly."

Not addressed: external data, pretrained models, language models, use of stimulus text, test-time adaptation.
Clarification channel: libribrain@robots.ox.ac.uk or the competition Discord.

## Kaggle Foundational Rules — the clauses that bind our method

They "supersede the competition-specific rules in the event of any conflict". The rest (eligibility, prizes,
taxes, publicity, privacy, indemnity) is Kaggle's standard text.

- 4b: "Submissions may not use or incorporate information from hand labeling or human prediction of the validation
  dataset or test data records."
- 5d: "No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's
  okay to share code if made available to all Participants on the forums."
- 6c: "if open source code is used in the model to generate the Submission, then you must only use open source code
  licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits
  commercial use of such code or model containing or depending on such code."
- 7a: winners are "determined solely by the leaderboard ranking on the Private Leaderboard"; "The Public
  Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set."
- 18a: "The Competition Data will contain private and public test sets. Which data belongs to which set will not be
  made available to Participants."

## Organiser announcements

- Kaggle topic 741919 (2026-09-18): mapping predictions between tracks via `word_onset_s`/`shuffle_seed` is
  cross-track leakage, disqualifying in both tracks; "information should not be leaked between tracks".
