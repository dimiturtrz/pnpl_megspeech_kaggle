# megspeech — agent instructions

Keep this file and `AGENTS.md` IN SYNC (different harnesses read each; same content).
## Issue tracking — beads (bd)

This project uses **bd (beads)** for issue tracking. Run `bd init` once, then `bd prime` for the full
workflow. `.beads/issues.jsonl` is git-tracked; the Dolt engine is gitignored.

- `bd ready` — available work · `bd show <id>` — details · `bd update <id> --claim` · `bd close <id>`
- Use `bd` for ALL task tracking — not TodoWrite or markdown TODO lists.

## Build & Test

`uv` is the env + dependency manager; `nox` is the task runner (every session calls `uvx`/`uv run`
so local runs match CI exactly).

```bash
uv sync --extra dev --extra devtools   # create/refresh .venv
nox -s lint test cov                   # the merge gate, locally
nox -s gates                           # same, one entrypoint
```

## Package structure

Your code lives in your declared packages: `core/`, `megspeech/` — organise modules however the project needs.
The scaffold ships the **guardrails**, not a layering. `graph.py --assert` keeps each package's
internal import graph honest (no cycles, no god-modules, no oversized files, no unmirrored logic modules);
thresholds live in `[tool.structure]`.

Directional layer rules (the kernel imports none of the others; a viewer never imports a trainer) are
enforced by **import-linter** — a one-way forbidden import is no cycle, so it's the axis `graph.py` can't
see. Contracts live in `[tool.importlinter]` (a `# >>> LOCAL-SLOT: import-contracts` region you extend),
run via `lint-imports` in nox/CI/pre-commit.

## Static-analysis gates — graduated, never suppressed

Each gate is engine + params + invocation, run identically in `noxfile.py` and CI. Philosophy: a check
starts advisory, and once the code is clean it **graduates** to enforced; from then on any regression
fails. Fix the finding, don't suppress it; suppressions stay minimal and meaningful (a `# noqa: RULE` with
no prose, never a blanket ignore). Thresholds are legislated values (file_max, CC 10), never seeded from
the repo's current state. The linter is a bug-finder, not cosmetics.

### Enforced — these BLOCK, in nox, CI and pre-commit alike

Every one of them is a rule you can write TO. Discovering a gate by failing it means the doc failed first.

- **ruff** (pinned `0.15.13`) — `check` on the full house select, enforced; `format --check` enforced too — pre-commit always ran it for real, so an advisory nox/CI meant the merge gate passed a tree that could not be committed.
  Encodes: logging not prints (T20), named constants not magic numbers (PLR2004), keyword-only bools
  (FBT), specific-exception-or-crash (BLE001/S), low complexity (C90/PLR), imports-at-top (PLC0415).
- **vulture** (pinned `2.16`) — dead-code at `min_confidence = 80`; scope is `[tool.vulture]`, not the CLI.
- **coverage** — pytest-cov with a `--fail-under=90` floor.
- **deptry** (pinned `0.25.1`) — undeclared (DEP001) / unused (DEP002) / transitive (DEP003) imports; config in `[tool.deptry]`.
- **pyrefly** (pinned `1.1.1`) — strict static types: every `def` annotated, plus the annotations check.
- **jscpd** — copy-paste over the threshold in the packaged `jscpd.json`.
- **import-linter** — directional forbidden-import contracts (`[tool.importlinter]`); the layer axis graph.py can't express.

The rest run in ONE process — `python -m devtools.run <packages> --gate <list>` — because each would
otherwise pay interpreter startup separately. All of them still answer `python -m devtools.<module>
<packages> --assert` individually, which is how you check ONE while fixing it:

- **arch-fitness** — `python -m devtools.graph.fitness`: god-module (fan-in/out), god-file, import-cycle,
  and the FILE-level test mirror (a logic module needs its `tests/unit/<pkg>/…/test_<name>.py`); thresholds
  in `[tool.structure]`. Advisory sub-checks: line-floor (off) + chokepoint.
- **demeter** — `python -m devtools.coupling.demeter`: CALLING a method through a field into a stranger.
  Ceiling `[tool.structure] demeter_max_depth` (default 2). Only chains ending in a CALL count — reading a
  deep config tree is data navigation, not coupling.
- **purity** — `python -m devtools.coupling.purity`: a `@property` that assigns to `self`. A read dressed as
  attribute access has to BE a read.
- **composition** — `python -m devtools.coupling.composition`: a `holds` cycle (A owns a B that owns an A),
  the intra-file case an import cycle cannot represent.
- **envy** — `python -m devtools.coupling.envy`: a METHOD more interested in another class than its own.
  Floor `[tool.structure] feature_envy_min` (default 4); value objects are never the target.
- **contracts** — `python -m devtools.coupling.contracts`: directional forbidden-USE rules over the resolved
  arrows (not imports). Empty by default — you add them.
- **ast-grep** — `python -m devtools.astgrep`: the packaged module-shape rule set, the syntactic axis ruff
  has no plugin API for.
- **method mirror** — `python -m devtools.hygiene.mirror`: for each public method `A.a`, the module's mirror
  test file holds a `test_a` that CALLS it and ASSERTS. STRICTER than graph's file-level mirror, which one
  smoke test satisfies for a module of twenty methods. Resolved through `[tool.structure] test_layout`
  (`off` disables both mirrors); per-method escape `# devtools-ignore: test-mirror`. See `docs/UNIT_TESTS.md`,
  which also states the case AGAINST the convention.
- **small** — `python -m devtools.hygiene.small`: a unit test touches nothing it did not create — no external
  data root, no network, no sleep, no unseeded RNG. UNIT ONLY; an integration test SHOULD read a real fixture.
- **shape-contracts** — `python -m devtools.shape_contracts`: a public array/tensor boundary must carry a
  **jaxtyping** shape (`Float[Tensor, "b c h w"]`), not a bare `np.ndarray`/`Tensor`. Aliases in
  `[tool.shape_contracts]`; make shapes live at runtime with a `@shapecheck` decorator (jaxtyping+beartype,
  see `devtools/README.md`). A fresh generation has 0 boundaries and passes — a bare boundary then fails.

### Advisory — these REPORT and never block

Reviewer signal. Reading them is how you find the next thing to fix, not a thing you must fix now.

- **class-shape smells** — `devtools.cohesion.state_candidates` (namespace-state), `devtools.cohesion.lcom`
  (LCOM4 cohesion), `devtools.cohesion.data_clumps` (Fowler clumps).
- **complexity ranking** — `devtools.cohesion.complexity` (radon; ruff C901 is the FIXED gate, this ranks).
- **magic literals** — `devtools.hygiene.magic_literals`: recurring literals that want a StrEnum/dataclass.
- **behavioural arrows** — `devtools.graph.arrows` and `devtools.graph.calls`: the resolved call/construct
  graph the architecture viewer draws.
- **ruff whole-tree** — the enforced select over `.` with `--statistics`, wider than the blocking `lint_paths`.
- **vulture at 60** and **coverage at 95%** — the same engines one notch stricter than their blocking bar.
- **archmap** (doc-gen) — `python -m devtools.graph.archmap <packages>` emits
  `docs/architecture/graph.json` (the committed, diffable architecture — nodes + weighted import edges) plus
  a self-contained interactive **cytoscape viewer** (`index.html`, regenerated + gitignored) that folds/expands
  packages to any depth and focuses a module's neighbourhood. Regenerate with `nox -s archmap` and **commit the
  graph.json diff** — it's the architecture-erosion record. The pre-commit hook regenerates on commit; CI runs
  `--check` (advisory) to flag a stale graph.json. **Publishing** (one repo = one Pages site): if the repo has
  no other Pages deployer, opt in with `archviz_pages` — it owns a staged site (`/architecture/` = main,
  `/architecture/preview/` = dev, `/` redirects). If the repo ALREADY deploys Pages, do NOT enable it; fold
  archmap into the existing workflow as an `/architecture/` subpath instead (see README: "Compose archmap into
  an existing Pages site"). Doc-gen, not enforcement — import-linter is the directional gate.

## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate
config, `devtools/`, and the nox/CI/pre-commit runners are **template-owned**: don't hand-edit them to pass
a gate — fix upstream in the scaffold and `uvx copier update`, or edit only within `# >>> LOCAL-SLOT`
regions. `copier update` pulls scaffold improvements as reviewable steps.

**Don't test template-owned code, either.** The `devtools/` detectors are unit-tested in the scaffold
itself; a `tests/unit/devtools/` here would re-litigate every template change locally (it duplicates
upstream coverage of byte-identical code, and breaks on the next `copier update`). Trust the upstream
tests — this repo's tests cover only ITS packages. Consumer-specific detector *config* (e.g. that your
`omit` list exempts your shells) is proven through the detector's public behavior, not a copied unit test.

## Conventions

- **Data lives out of the repo** (under a root pointed at by `paths.yaml`, gitignored). No datasets,
  weights, or run outputs in git.
- **No notebooks** (`.ipynb`) — a one-off run becomes a committed CLI / `python -m` entrypoint
  (reproducible + reviewable), never a REPL session.
- **Minimal comments** — prefer self-documenting names; a comment is a liability, not a default.
- **Config objects over long arg lists**; strategy pattern over `if`/type dispatch.
- **Doc layers** — `learning/<date>_<topic>.md` = the study ramp / general understanding;
  `research/` = external / field synthesis (theirs); `interpretations/<task>/<date>_<topic>.md` =
  sense-making of *our own* results (per task, `converging/` for cross-task). The build log is git history;
  portfolio READMEs carry the result + link out.
- **Gate shapes to write TO** — a unit test mirrors its module WITHOUT a `test_` prefix
  (`megspeech/data/raw/fetch.py` → `tests/unit/megspeech/data/raw/fetch.py`); no top-level functions except `main` (make it a
  method on the class that owns it); every import is a declared dependency (deptry DEP003); generics are fully
  parameterised (pyrefly: `list[dict[str, object]]`, not `list[dict]`).
- **Competition loop** — data comes from `python -m megspeech.data.raw.fetch` (the `pnpl/*` HF repos are public, so it
  fetches with `token=False`; a stale local HF token cannot break it). Spend a Kaggle submission only on a
  model scored locally on the official val split, and record LB vs local in its beads issue. The GPU is shared
  with other projects — check `nvidia-smi` before launching.




<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
