# megspeech

[![ci](https://github.com/dimiturtrz/pnpl_megspeech_kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/dimiturtrz/pnpl_megspeech_kaggle/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Decoding heard words from non-invasive MEG — an entry to the PNPL 2026 / LibriBrain100 competition
(Deep: one subject, ~80 h; Broad: 32 subjects, minutes of calibration). Plan and numbers: [`docs/PLAN.md`](docs/PLAN.md).

## Result

Metric: balanced top-1 accuracy over the 50-word vocabulary.

| track | best public | local test | submission |
|---|---|---|---|
| Deep | .695 | .720 | 16 fine-tunes over 8 pretraining seeds |
| Broad | .631 | — | 29 members, mean-subject fallback for unseen subjects |

For scale: the d'Ascoli et al. recipe reproduced here scores .444 public. The Kaggle leaders at 10-11 were at .775
(Deep) and .732 (Broad), so this entry is not at the top. Every public delta under ~.03 between families is inside the
measured draw noise ([where we are](interpretations/converging/2026-10-13_where_we_are.md)).

## Method (Deep)

1. **Word encoder** — 1 s of MEG per word, decimated 5×, through a dilated conv stack plus a spatial-component span
   branch.
2. **Sentence context** — a 4-layer transformer over every word of the sentence. Positions are the word onsets,
   encoded as log time intervals (40 ms – 8 s). Dropping that encoding costs 2.3 pt on validation and 3.6 pt on the
   local test: the timing of the words around a word carries information about it.
3. **Pretraining without word labels.** The stem never trains the 50-word classifier. Instead, each word window is
   contrastively aligned to three targets:
   - the wav2vec2 latent of the speech heard in that window;
   - the t5 embedding of the word, which also covers the ~60% of windows outside the vocabulary;
   - the speech envelope.
4. **Fine-tune** — the 50-word heads only, warm-started from the stem. Regularised with context dropout .25, EMA,
   label smoothing and weight decay.
5. **Ensemble** — diversity comes from the pretraining seed. A second fine-tune seed on the same stem is ~0, because
   it shares the backbone's errors.

What was tried and did not survive the local test is recorded in [`interpretations/`](interpretations/) and in the
beads tracker. That includes KD, kNN readout, SAM, refits, TTA, time-stretch and source-mixing.

## Layout

- `core/`, `megspeech/` — your packages. The guardrails target all of them; organise modules however the project needs.
- `tests/` — `unit/`, `integration/`, `e2e/`.
- `devtools/` — the guardrail tools (`graph.py` arch-fitness, plus the class-shape explorers).

The set of packages the gates lint/test/graph is the `packages` answer (comma-separated) — add more
without touching any gate file; every gate renders over the list. Directional layer contracts between
packages are **opt-in** (add import-linter yourself — see `devtools/README.md`), never imposed.

## Quickstart

```bash
git clone https://github.com/dimiturtrz/pnpl_megspeech_kaggle
uv sync --extra dev --extra devtools
nox -s lint test cov
```

## Sample viewer

A browser tool for reading what a model gets wrong, word by word. It shows a stratified set of sentences:

- **Sentence card** — every scored word is underlined and coloured by its error percentile within the active model. Click a word to select it. Its rank is coloured on the same scale.
- **MEG traces** — channels picked by spread, model weight, word-evoked RMS, sensor kind, region, or helmet neighbourhood. A resizable helmet map shows each location's saliency.
- **Model panel** — the selected models' outputs for the same word, side by side.

```bash
uv run python -m megspeech.evaluation.viz export --models <run> [<run> ...]
uv run python -m megspeech.evaluation.viz serve   # http://localhost:8000
```

Details: [`megspeech/evaluation/viz/README.md`](megspeech/evaluation/viz/README.md).

## Gates

`nox -s lint test cov` (or `nox -s gates`) runs exactly what CI runs:

- **lint** — ruff check (pinned) + ruff format --check (enforced) + vulture + arch-fitness (`graph.py --assert`) + ast-grep + jscpd.
- **test** — pytest.
- **cov** — pytest --cov with a `--fail-under=90` floor.

## Architecture site

`nox -s archmap` writes `docs/architecture/graph.json` (committed, diffable — nodes + weighted import edges)
plus a self-contained interactive viewer (`index.html`, gitignored, rebuilt on demand). **Commit the
graph.json diff** — it's the architecture-erosion record.


Publishing to GitHub Pages — one repo = one Pages site, so pick ONE of:

- **Sole owner** — this repo has no other Pages deployer. Answer `archviz_pages` yes at scaffold time; the
  generated `pages.yml` owns a staged site (`/architecture/` = main, `/architecture/preview/` = dev, `/`
  redirects) and enables it. Nothing else to do.
- **Compose** — this repo ALREADY deploys Pages (a docs/demo site). Leave `archviz_pages` off and fold the
  archmap build into that existing workflow as an `/architecture/` subpath. Build it isolated (no project
  sync, no touch to the pinned devtools) — `write_viewer` only needs the vendored template + libs:

  ```yaml
  - uses: astral-sh/setup-uv@v5
  - name: Architecture view (/architecture/)
    run: |
      uv run --no-project --with "sdlc-devtools @ git+https://github.com/dimiturtrz/sdlc-scaffold.git@v1.28.0#subdirectory=sdlc-devtools" \
        python -c "from devtools.graph.archmap import Archmap; Archmap(['core', 'megspeech']).write_viewer(project='megspeech')"
      mkdir -p _site/architecture
      cp docs/architecture/graph.json docs/architecture/index.html _site/architecture/
  ```

  graph.json stays committed (the diff-truth); regen it locally with `nox -s archmap` when the import
  structure changes. (`../synthscape` is the reference — archmap folded into its `deploy-fit-explorer`
  workflow beside the fit-explorer views.)

## Development

Tooling + quality gates are provisioned by an in-house copier template (**sdlc-scaffold**). Refresh with
`uvx copier update`; `.copier-answers.yml` pins the template version.

### Local hooks

The same gates CI enforces are bound to git events via pre-commit. Install **both** stages:

```bash
pre-commit install                        # commit stage — fast static gates (ruff/vulture/arch-fitness/…)
pre-commit install --hook-type pre-push   # push stage — fast unit suite (tests/unit)
```

The pre-push hook runs `pytest tests/unit` so a change that lints clean but breaks a test **contract** (a
signature change a mirror test still calls the old way) is caught before the push, not after CI goes red. It
is deliberately push-only (not every commit) and unit-only (no integration/e2e); the coverage floor stays a
CI job. The default `pre-commit install` does **not** wire pre-push — run the second line once per clone.
