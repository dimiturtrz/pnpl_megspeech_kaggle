# Guardrail analyzers — how they run in this project

The structural-guardrail **analyzers are an installed package** (`sdlc-devtools`), pinned by scaffold tag
in `pyproject.toml`'s `devtools` optional-dependency — not vendored source, so an engine update is a
one-line pin bump on `copier update` (no source diff in your PRs). They import as `devtools`, so every gate
is invoked identically to before:

```bash
uv run --extra devtools python -m devtools.graph.fitness --assert <packages>   # arch-fitness (+ test-mirror)
uv run --extra devtools python -m devtools.hygiene.magic_literals <packages>   # recurring-literal report (advisory)
uv run --extra devtools python -m devtools.cohesion.lcom <packages>             # + data_clumps / state_candidates (advisory)
```

The ast-grep module-shape rules and the jscpd config ship **inside** the package (no vendored `devtools/`
config); the gates locate them from the install:

```bash
uvx --from ast-grep-cli ast-grep scan -c "$(uv run --extra devtools python -m devtools.tools.config sgconfig)" <packages>
npx --yes jscpd <packages> --config "$(uv run --extra devtools python -m devtools.tools.config jscpd)"
```

Per-engine detail (what each flags, thresholds, exit semantics) lives in the package's own README. The
`[tool.structure]` / `[tool.magic_literals]` / `[tool.shape_contracts]` config in this repo's
`pyproject.toml` tunes them; the LOCAL-SLOT regions there are yours to edit.

## Making jaxtyping shapes LIVE — `@shapecheck`

A bare `Float[Tensor, "b c h w"]` annotation is **inert** at runtime — it documents the shape but checks
nothing. The scaffold ships the deps (`jaxtyping` + `beartype`) but **no package code** (the template is
guardrails-only). Add a one-line decorator in your own types module to make the annotations enforce:

```python
# core/types.py (or wherever your shared type vocabulary lives)
from beartype import BeartypeConf, beartype as _beartype
from jaxtyping import jaxtyped

# is_pep484_tower: an int also satisfies a float annotation (numpy scalars included).
shapecheck = jaxtyped(typechecker=_beartype(conf=BeartypeConf(is_pep484_tower=True)))
```

Then `@shapecheck` on a boundary makes a wrong-shape/dtype array raise at the call. It's O(1) per call
(reads `.shape`/`.dtype`), independent of tensor size — negligible at coarse seams. A genuinely hot call
that wants the annotation as docs only just leaves `@shapecheck` off. `shape_contracts` (in the package) is
the static gate that fails on a public boundary still missing a jaxtyping shape.


## Directional layer contracts — import-linter

`graph`'s structural checks are layer-AGNOSTIC and catch *cycles*, but not a one-way *forbidden*
import (`core -> trainer` is no cycle). That directional axis is import-linter's.

Shipped: `[tool.importlinter]` in `pyproject.toml` carries `root_packages` + a kernel-independence
starter contract (the first package imports none of the others) inside a `# >>> LOCAL-SLOT: import-contracts`
region — add your viewer/trainer or domain contracts there. Enforced in nox/CI/pre-commit via `lint-imports`.

