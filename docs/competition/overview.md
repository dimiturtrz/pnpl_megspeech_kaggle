# PNPL Competition 2026 — Overview + Evaluation (both tracks)

Verbatim copy of the Kaggle `pnpl-competition-2026-deep` Overview and Evaluation tabs, pasted by the owner on
2026-10-05 (close: 2026-10-16). The Kaggle CLI `competitions pages` subcommand could not fetch it. The
`pnpl-competition-2026-broad` tabs (pasted 2026-10-07) are the same text except the landing line below and the
leaderboard announcement in [announcements.md](announcements.md). Rules: [rules.md](rules.md). Data tab: not
captured.

Broad landing line: "This is the landing page for the Broad (i.e., multi-subject) track. If you want to try
maximizing performance on a single subject, take a look at the Deep track here."

## Overview

Welcome to the PNPL Competition 2026! Our goal is to advance the societal impact of speech decoding technologies by tackling a new large-scale word classification benchmark.

This is the landing page for the Deep (i.e., single-subject) track. If you want to try generalizing across multiple subjects, take a look at the Broad track here.

More details can be found on LibriBrain.com

## Description

The PNPL Competition 2026 invites researchers, students, and practitioners to advance non-invasive speech decoding from MEG by tackling a new large-scale word classification benchmark. Building on the LibriBrain effort and the 2025 PNPL competition, this year's challenge is designed to push progress in two complementary directions: depth and breadth. The long-term goal is to accelerate the development of safe, effective brain–computer interfaces for speech and communication, while lowering the barrier to entry through standardized data splits, benchmark baselines, tutorials, and an open leaderboard. ([neural-processing-lab.github.io][3])r's competition centers on LibriBrain100 (Mantegna et al., in prep), a new dataset that expands the LibriBrain family in two ways. First, it expands within-subject depth through substantially more data from a single participant (subj0), including the full Sherlock Holmes canon as well as carefully selected stimuli that better balance phonetic and semantic structure. Second, it expands across-subject breadth through a larger group of participants (subj1–N), enabling the study of cross-subject generalization and data-efficient adaptation. The result is a benchmark designed not only to reward raw performance, but also to encourage methods that transfer across people and across levels of supervision.

## Competition Tasks

The competition consists of two tracks, both focused on word classification from MEG.

Track 1: Word Classification on subj0 Participants develop models for word classification using the deepest single-subject non-invasive speech-decoding dataset in the LibriBrain family. This track is intended to encourage approaches that exploit large amounts of within-subject data to learn stronger neural representations of spoken language.

Track 2: Word Classification on subj1–N Participants develop models for word classification across multiple subjects. This track is intended to encourage approaches that generalize beyond a single individual and make efficient use of limited supervised data for adaptation.

Together, these tracks reflect two complementary challenges in non-invasive speech decoding: scaling performance with more data from one person, and achieving robust transfer across many people.

## Dataset Overview

LibriBrain100 is released as a superset of the existing LibriBrain resources, extending them along both the depth and breadth dimensions.

For subj0, the competition data will include substantially deeper recordings, spanning:

- the entire Sherlock Holmes canon,
- TIMIT and MOCHA-TIMIT, included to provide stronger phonetic coverage,
- The Moth and Modern Love podcast stimuli, included to provide richer semantic diversity.

For subj1–N, the competition will use broader multi-subject data derived from the public validation and test portions of LibriBrain, based on Sherlock 1 sessions 11 and 12. This track is additionally designed to probe performance under varying amounts of supervised fine-tuning data.

To support this, subjects in the multi-subject track will be released in supervision groups with different amounts of labeled data:

- 100% group: approximately 40 minutes of audiobook listening data
- 50% group: approximately 20 minutes
- 25% group: approximately 10 minutes
- 0% group: no released listening data, but subject IDs will be provided

This setup is intended to encourage methods that combine pretraining on deep single-subject data with efficient adaptation to new subjects under realistic supervision constraints.

## Why This Competition?

Word classification is a natural next step for large-scale non-invasive language decoding. Relative to simpler detection tasks, it requires richer representations of linguistic content; relative to fully open-vocabulary decoding, it remains sufficiently controlled to support reproducible benchmarking and clear scientific comparisons.

The 2025 LibriBrain competition focused on speech detection and phoneme classification, while the associated LibriBrain dataset paper already positioned word classification as one of the benchmark task families supported by the dataset and its tooling. The 2026 competition builds directly on that foundation by making word classification the central challenge.

To make participation accessible, we provide:

- starter code and tutorials,
- standardized train/validation/test splits,
- baseline submissions,
- an evaluation server and public leaderboard.

As in 2025, the goal is to make it possible for new participants to get started quickly while still leaving room for highly ambitious approaches. The previous competition emphasized plug-and-play access through the pnpl Python package, online tutorials, and community support, and this competition follows the same spirit.

The primary evaluation metric will be top-10 balanced accuracy over a fixed vocabulary. This metric is intended to measure how reliably models recover the correct target word while accounting for class imbalance and maintaining comparability across approaches.

## Competition Philosophy

This competition is designed to encourage progress along two axes that matter for real-world non-invasive speech BCIs:

- How far can we push performance with deep, single-subject data?
- How well can we generalize to new subjects with limited supervision?

By pairing a deep single-subject benchmark with a broader multi-subject adaptation benchmark, LibriBrain Competition 2026 aims to provide a clearer picture of what current methods can achieve, where they break down, and which ideas are most promising for scalable non-invasive speech decoding.

## Evaluation

Submissions are evaluated on Top-10 Balanced Accuracy (BAcc@10) over the competition vocabulary of 50 words.

Each trial in the test set is one word the participant heard. For every trial your model outputs a probability distribution over the 50 competition words, and a trial counts as correct if the true word is among your 10 highest-scoring words. The score is balanced across classes: we average the per-word Top-10 recall and then average those over the 50 words, so rare words count equally.

Formally, for each word w in the vocabulary let

    Recall@10(w) = the fraction of trials whose true word is w for which w appears in the model's top-10 predictions.

The final score is the mean of Recall@10(w) across all 50 words.

Higher is better. Uniform random guessing scores about 20% (a random top-10 covers 10 of the 50 words). Ties are broken with Top-1 Balanced Accuracy (BAcc@1) — the same quantity computed with a top-1 shortlist.

The leaderboard ranks on the competition vocabulary. The Moses-50 vocabulary (moses-vocabulary.csv) is tracked as a secondary metric using the same BAcc@10 formula and does not affect the primary ranking.

## Submission File

For each index in the test set you provide a probability distribution over the competition vocabulary — a single predicted word is not enough, because the metric needs your top-10 ranking. The file has a header and one row per test index:

- index,
- one column per competition word, in vocabulary.csv order (is, the, a, …, as, on),
- optionally, one column per Moses-50 word, each prefixed moses_ (moses_am, moses_are, …, moses_yes, moses_you).

```
index,is,the,a,...,on,moses_am,moses_are,...,moses_you
0,0.041,0.016,0.022,...,0.030,0.008,0.030,...,0.015
1,0.010,0.001,0.003,...,0.020,0.014,0.001,...,0.009
2,0.007,0.033,0.004,...,0.011,0.002,0.004,...,0.021
```

Notes:

- Provide 51 or 101 columns: index + 50 competition-vocab columns, plus optionally the 50 moses_ columns.
- Values are treated as scores — only the ranking within each block matters, so they need not sum to exactly 1 (a softmax is a natural choice). They must be non-negative and finite.
- Include exactly one row per test index, in any order.

The competition vocabulary is in vocabulary.csv and the Moses-50 vocabulary in moses-vocabulary.csv (see the Data tab).

## Generating a submission

The pnpl library builds a correctly-formatted file for you. Load the holdout, run your model over the released MEG windows, and write the CSV:

```python
import numpy as np
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

holdout = LibriBrainCompetitionHoldout(track="deep")   # "deep" or "broad"

primary, moses = [], []
for meg, _meta in holdout.iter_windows(batch_size=256):  # meg: (B, 306, 250)
    p, m = your_model(meg)                               # each (B, 50) probabilities
    primary.append(p); moses.append(m)

write_submission(
    "submission.csv",
    indices=holdout.indices,                 # required row order
    primary_probs=np.concatenate(primary),   # (N, 50) competition vocab, scored
    secondary_probs=np.concatenate(moses),   # (N, 50) Moses-50, optional/secondary
)
```

holdout.indices provides the exact index order the leaderboard expects. See the starter notebook for a full example.
