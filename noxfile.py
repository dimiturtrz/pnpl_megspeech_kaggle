"""Task runner for megspeech.

Every session shells out to ``uvx``/``uv run`` with pinned tool versions so a local ``nox`` run
executes exactly what CI executes. Sessions declare ``venv_backend="none"`` — nox does not build an
environment; ``uv`` owns dependency resolution.
"""

import nox

nox.options.sessions = ["lint", "test", "cov"]

RUFF = "ruff@0.15.13"
VULTURE = "vulture@2.16"
# The enforced ruff select, chunked purely so no line runs past the 120 limit — the value is the flat
# comma list ruff's `--select` takes, and `,`.join restores it exactly. Written wrapped because the flat
# literal is a single 274-char string that `ruff format` cannot split, so it would ship an E501 to every
# consumer forever (bd 98d); the chunking also makes a rule arriving or leaving a one-line diff.
SELECT = ",".join(
    (
        "F,B,I,T201,FBT,BLE001,S101,S110",
        "C901,PLR0912,PLR0913,PLR0915,PLR2004,PLC0415,RUF100,N",
        "E741,E742,E743,PLR0124,PLR1714,PLW3301,RUF012,RUF005",
        "RUF007,RUF010,RUF022,RUF046,C408,C420,SIM,PERF401",
        "PLW0108,E731,E402,ICN001,S603,S607,PTH123,E501",
        "SLF001,PGH003,PGH004,TID251",
    )
)
LAYERS = ["core", "megspeech"]
# ruff + jscpd are R1 HYGIENE gates — they may scan WIDER than the R2/R3 arch set LAYERS (a viewer / tests
# tree worth linting). Default = LAYERS; widen via lint_paths/jscpd_paths in .copier-answers.yml (bd 9mu).
LINT_LAYERS = ["core", "megspeech", "tests", "comp"]
JSCPD_LAYERS = ["core", "megspeech"]


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """ruff check + format + vulture + import-linter + arch-fitness + ast-grep + jscpd."""
    # --select on the CLI bypasses pyproject [tool.ruff.lint] ignore, so the ml jaxtyping waiver is repeated
    # here — matching CI (bd skr GAP1/kqk). F722 = multi-token dim strings ("b c h w"); F821 = single-axis
    # ("n") which parses as an undefined forward-ref. Off domain=ml both are absent (no jaxtyping dep).
    session.run("uvx", RUFF, "check", *LINT_LAYERS, "--select", SELECT, "--ignore", "F722,F821", external=True)
    # Advisory (mirrors CI, never blocks): the full enforced select over the whole tree.
    # ruff_advisory_select is empty by default (E501/SLF001 graduated into the enforced union, bd 4c2/8ex),
    # so --extend-select is appended only when it carries a code.
    session.run(
        "uvx",
        RUFF,
        "check",
        ".",
        "--statistics",
        external=True,
        success_codes=[0, 1],
    )
    # ENFORCED, and it has to be: pre-commit has always run this hook for real, so a tree that passed
    # `nox -s gates` could still be refused at commit time — the documented merge gate said green on a
    # tree that could not be committed. A generated repo is formatted by construction, so the graduation
    # bar is met on arrival; the only way to fail this is drift introduced after generation.
    session.run("uvx", RUFF, "format", "--check", ".", external=True)
    # vulture takes NO package args — scope is config-driven via [tool.vulture] paths (the vulture-scan
    # LOCAL-SLOT), same as CI + pre-commit. A repo widens the dead-code scan there, not on the CLI.
    session.run("uvx", VULTURE, "--min-confidence", "80", external=True)
    # Advisory: conf60 dead-code (real signal, but eats no-fan-in FPs so it can't block). vulture exits 3
    # when it finds dead code — success_codes swallows that (0=clean, 3=found), only a usage error blocks.
    session.run("uvx", VULTURE, "--min-confidence", "60", external=True, success_codes=[0, 3])
    session.run("uvx", "--from", "import-linter", "lint-imports", external=True)
    # (graph — god-module / import-cycle / god-file / test-mirror — rides the single batch run below.)
    # (ast-grep module-shape rides the single batch run below — `devtools.astgrep` owns locating the
    # packaged config and invoking the CLI, so nox no longer re-encodes that invocation.)
    # jscpd still reads its config from the INSTALLED devtools package; `python -m devtools.tools.config` prints
    # the packaged path for the external CLI to consume.
    # DRY gate — ENFORCED (blocks over the jscpd.json threshold), matching the cardiac/mindscape majority.
    _jscpd_cfg = session.run(
        "uv",
        "run",
        "-q",  # silence uv's VIRTUAL_ENV-mismatch notice so only the config path is captured
        "--extra",
        "devtools",
        "python",
        "-m",
        "devtools.tools.config",
        "jscpd",
        external=True,
        silent=True,
    ).strip()
    session.run(
        "npx",
        "--yes",
        "jscpd",
        *JSCPD_LAYERS,
        "--config",
        _jscpd_cfg,
        external=True,
    )
    # Advisory explorers — print a ranked report, always exit 0 (never block): class-shape smells, recurring
    # magic literals (StrEnum/dataclass candidates), and radon cyclomatic complexity (ruff C901 is the FIXED
    # complexity gate; this just ranks). Reviewer signal only.
    _advisory = (
        "cohesion.state_candidates",
        "cohesion.lcom",
        "cohesion.data_clumps",
        "hygiene.magic_literals",
        "cohesion.complexity",
    )
    for _tool in (*_advisory, "graph.arrows", "graph.calls"):
        session.run("uv", "run", "--extra", "devtools", "python", "-m", f"devtools.{_tool}", *LAYERS, external=True)
    # Advisory (never blocks): flag a stale docs/architecture/graph.json — regenerate with `nox -s archmap`.
    # archmap --check exits 1 on drift; success_codes swallows it so CI/nox report without failing the gate
    # (doc-gen, not enforcement — import-linter is the directional gate). Graduates to a gate per the creep.
    session.run(
        "uv",
        "run",
        "--extra",
        "devtools",
        "python",
        "-m",
        "devtools.graph.archmap",
        *LAYERS,
        "--check",
        external=True,
        success_codes=[0, 1],
    )
    # ENFORCED dependency hygiene — deptry: undeclared (DEP001) / unused (DEP002) / transitive (DEP003)
    # imports. `--with` adds deptry to the run so it reads installed dist metadata; config in [tool.deptry].
    session.run("uv", "run", "--with", "deptry==0.25.1", "deptry", ".", external=True)
    # EVERY python gate, in ONE process. Each `python -m devtools.<tool>` pays to start an interpreter and import
    # devtools before it analyses anything, and that startup is the larger half of a gate's cost — so six of
    # them paid it six times. The batch runner pays once and shares a single parse of the tree between the
    # engines that resolve names.
    #   graph       god-module / import-cycle / god-file / test-mirror. `--extra devtools` because it needs
    #               grimp+networkx (optional extra), same as CI's synced tests job.
    #   demeter     CALLING a method through a field into a stranger couples this class to a type it never
    #               declared. Ceiling: [tool.structure] demeter_max_depth (default 2). Only chains that end
    #               in a call count — reading a deep config tree is data navigation, not coupling.
    #   astgrep     the packaged ast-grep rule set (module shape). It shells out to the vendored CLI, but it
    #               answers the same two verbs as the AST engines, so the runner drives it identically.
    #   purity      a @property that assigns to self. A read dressed as an attribute has to BE a read, and
    #               it is also what makes demeter's "an attribute chain is only data" true in a language
    #               where a property is a method call spelled as attribute access.
    #   composition a `holds` cycle (A owns a B that owns an A) means neither can be built or tested alone.
    #               Catches the intra-file case the IMPORT cycle check cannot represent.
    #   envy        a METHOD more interested in one other class than its own object. Floor:
    #               [tool.structure] feature_envy_min (default 4); value objects are never the target.
    #   contracts   directional forbidden-USE rules over the decomposed arrows. Empty by default.
    #   mirror      the METHOD-level test mirror: for each public method A.a, the module's mirror test file
    #               holds a `test_a` that CALLS it and ASSERTS. graph's file-level mirror only asks whether a
    #               module has a test FILE, which one smoke test satisfies for a module of twenty methods.
    #               Resolved through [tool.structure] test_layout, so `off` turns both mirrors off together;
    #               per-method escape is `# devtools-ignore: test-mirror`. See docs/UNIT_TESTS.md, which also
    #               states the case AGAINST the convention. A fresh gen has no methods and passes.
    #   small       a unit test touches nothing it did not create: no external data root, no network, no
    #               sleep, no unseeded RNG. UNIT ONLY — an integration test SHOULD read a real fixture and an
    #               e2e SHOULD shell out, so holding those to this rule would be the gate misreading its own
    #               subject. A fresh gen has no tests and passes.
    #   shape_contracts  ML-only (bd vip.4): every public array/tensor boundary carries a jaxtyping shape,
    #               not a bare np.ndarray/Tensor (aliases in [tool.shape_contracts]). A fresh gen has 0
    #               boundaries and passes; a bare boundary then fails. It BLOCKS — no success_codes swallow.
    # Every one of them RUNS even if an earlier one fails, so a commit that breaks three gates is reported
    # once rather than discovered three times. `python -m devtools.<tool> --assert` still works unchanged.
    session.run(
        "uv",
        "run",
        "--extra",
        "devtools",
        "python",
        "-m",
        "devtools.run",
        *LAYERS,
        "--gate",
        "graph.fitness,coupling.demeter,coupling.purity,coupling.composition,coupling.envy,coupling.contracts,astgrep,hygiene.mirror,hygiene.small,shape_contracts",
        external=True,
    )
    # ENFORCED static type gate — pyrefly strict: every def annotated + the annotations check. `--with` adds
    # pyrefly to the synced run so it reads installed dep stubs (pydantic/numpy); scope=lint_paths (LINT_LAYERS).
    session.run("uv", "run", "--with", "pyrefly==1.1.1", "pyrefly", "check", *LINT_LAYERS, external=True)


@nox.session(venv_backend="none")
def test(session: nox.Session) -> None:
    """pytest."""
    session.run("uv", "run", "pytest", "tests", "-q", external=True)


@nox.session(venv_backend="none")
def cov(session: nox.Session) -> None:
    """pytest with coverage, failing under the floor."""
    session.run(
        "uv",
        "run",
        "pytest",
        "tests",
        "--cov",
        "--cov-report=term-missing",
        external=True,
    )
    session.run("uv", "run", "coverage", "report", "--fail-under=90", external=True)
    # Advisory: 95% target — warns, never fails (coverage exits 2 when under; success_codes swallows it).
    session.run("uv", "run", "coverage", "report", "--fail-under=95", external=True, success_codes=[0, 2])


@nox.session(venv_backend="none")
def gates(session: nox.Session) -> None:
    """Run every gate: lint + test + cov."""
    lint(session)
    test(session)
    cov(session)


# NOT in nox.options.sessions — opt-in (`nox -s audit`), mirroring the nightly `audit` workflow. Security
# advisories change under you, so this is a scheduled/on-demand scan, not part of the per-commit gate set.
# NOT in nox.options.sessions — the manual regen call. archmap is doc-gen: it (re)writes docs/architecture/
# {graph.json, index.html} from the current import graph — graph.json is the diffable erosion signal,
# index.html the self-contained interactive viewer (a GitHub Pages architecture site). Run it after a
# structural change and commit the graph.json diff. `lint`/CI only --check that graph.json is current.
@nox.session(venv_backend="none")
def archmap(session: nox.Session) -> None:
    """Regenerate docs/architecture/ (graph.json + the interactive viewer) — commit the graph.json diff."""
    session.run("uv", "run", "--extra", "devtools", "python", "-m", "devtools.graph.archmap", *LAYERS, external=True)


@nox.session(venv_backend="none")
def audit(session: nox.Session) -> None:
    """Known-CVE scan of the dependency closure (pip-audit); --skip-editable drops local/git source installs."""
    session.run("uv", "sync", "--all-extras", external=True)
    session.run("uv", "run", "--with", "pip-audit==2.10.1", "pip-audit", "--skip-editable", external=True)
