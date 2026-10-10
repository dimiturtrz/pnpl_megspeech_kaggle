# Do our local numbers rank the Kaggle public score? (2026-10-10)

**Scope.**

- **Rows.** Every Kaggle submission with a public score whose model could be traced to a local number.
- **Track-level rows.** Taken from the number quoted in the submission description.
- **Per-source rows.** Mapped by CSV hash, or by the model name in the description: 36 of 50 Deep submissions.
- **Deep public.** About 434 rows, SE ≈ .023 for one score.
- **Scripts.** Scratch only, not committed.

| track | local number | n | Spearman | Pearson |
|---|---|---|---|---|
| Broad | subject 1–32 ruler | 19 | .91 | .90 |
| Deep | VAL | 32 | .23 | .28 |
| Deep | local test (pnpl test) | 30 | .04 | .23 |
| Deep | local test, Sherlock1 only (n = 2133 models) | 30 | .05 | .44 |
| Deep | local test, TheMoth only (n = 2133 models) | 30 | −.17 | .20 |

Examples of the Deep mismatch:

- `s-hdnd200`: local test .726, public .649.
- `s-mhub4`: local test .679, public .687.

Four different blends score exactly .68883 public, though their local tests differ (.693–.704). Blends that share most
members agree on the top-10 of nearly every public row, so the public score cannot separate them.

## Reading (inferences)

1. **The Broad ruler is a working selector.** It ranks public almost in order. The rule against selecting members
   on it stands because it is our only Broad estimate, not because it fails to transfer.
2. **No Deep local number ranks the public score.** Neither split nor either source does. Deep local-test gains,
   such as the m11wd blends' +.02, are not evidence of a public gain. The Kaggle Deep test uses stimuli different
   from both Sherlock1 and TheMoth (`../../docs/competition/overview.md`).
3. **Pearson above Spearman** means the local numbers do separate bad models from good ones, but do not rank among
   the good ones. Most of the correlation comes from a few weak singles.
4. **What follows for Deep.** Choose by mechanism and by public A/B, not by local test. Deep blends near the top are
   below the public score's resolution.

## Not ruled out

- **A selector we have not computed.** For example, one restricted to words or sentence lengths like the Kaggle
  test's.
- **Description numbers.** Their definitions changed over the week (legacy full-data vs stratified VAL), so pooled
  correlations mix rulers.
