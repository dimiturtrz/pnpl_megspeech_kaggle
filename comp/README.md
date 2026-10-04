# Submission — PNPL 2026 / LibriBrain100

Deadline **2026-10-15 AoE**. Facts, metrics and references live in [`docs/PLAN.md`](../docs/PLAN.md).

This directory is named `comp/`, not `kaggle/`: a top-level `kaggle/` shadows the installed `kaggle`
package on `sys.path[0]`.

## Data fetch

The Kaggle test set ships as `.npz` in HF `pnpl/LibriBrain-Competition-2026` (5.7 GB); training data comes through
`pnpl` from HF `pnpl/LibriBrain` (preprocessed `.h5` only — never the raw `.fif`). Both land under
`<data>/raw/pnpl_libribrain/` (see `paths.example.yaml`).
