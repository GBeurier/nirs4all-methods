# Contributing to nirs4all-methods (`n4m`)

Thanks for considering a contribution. `nirs4all-methods` is a
research-grade library going to production, so the bar is intentionally
high — but the rules are simple and the test infrastructure does most
of the enforcement automatically.

## The extensibility model (read this first)

`n4m` is a **closed library** in the sense that **methods, references,
bindings, and packaging subsets are extensible only via Pull Request**
to this repository. There is no plugin API, no user-side method
registration, no runtime injection of custom operators. This is a
deliberate choice — see [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)
§1.1bis.

User-driven requests go through structured GitHub issue templates so
the maintainer can triage cleanly. Pick the form that matches your
change:

| Want to … | File this issue | Then submit this PR |
|-----------|-----------------|---------------------|
| Add a new method | `method-request.yml` | `method-add.md` |
| Update an existing method's behaviour | `method-update.yml` | `method-update.md` |
| Add / pin an external reference | `external-reference-request.yml` | `external-reference.md` |
| Add a new language binding | `binding-request.yml` | `binding-add.md` |
| Add a new idiomatic profile to an existing binding | `binding-update.yml` | `binding-update.md` |
| Add a new packaging subset (`pls4all`, `nirs4all-selection`, ...) | `subset-request.yml` | `subset.md` |
| Report a numerical parity discrepancy | `parity-discrepancy.yml` | `parity-fix.md` |
| Report a regular bug | `bug-report.yml` | (the generic PR template) |

Each issue template links its matching PR template via a
`?template=…` URL so the request → PR flow lands you on the right
form automatically. Each PR template ends with the §2.10 invariants
checklist that the reviewer (human + Codex) verifies.

## Ground rules

1. **The canonical specs are [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
   and [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).** Any PR that
   violates one of their decisions (C ABI, zero mandatory deps, no
   exceptions across the boundary, no STL in the public API, single
   `libn4m` binary, etc.) will be rejected without further discussion.
   Open an issue first if you think a spec itself needs to change.
2. **The parity gate is non-negotiable.** Any PR that changes a
   numerical path MUST update the relevant snapshots and keep the
   tolerances declared in `catalog/methods/<id>.yaml` green. For
   `status: paper_only` methods, the `self_consistency` block stands
   in for the parity gate.
3. **The ABI surface is monitored.** Any PR that adds, removes or
   renames an exported `n4m_*` symbol must regenerate
   `cpp/abi/expected_symbols_*.txt` for all three platforms and
   document the change in `docs/abi/changes_log.md`.

## Development setup

The supported workflow is the devcontainer (or GitHub Codespaces) — it
ships the full C++ / Python / R / Octave / Node / Emscripten / Docker
toolchain at the version `n4m` targets, so you don't fight environment
drift.

```bash
# Easiest: VS Code + Docker (or Codespaces): "Reopen in Container"
make bootstrap         # detects VS Code + Docker; falls back to bare metal

# Equivalent without VS Code
docker compose -f .devcontainer/docker-compose.yml run --rm dev bash
# (or simply: `make shell`)

# Bare metal (no Docker): best-effort installer per OS
scripts/bootstrap-bare-metal-linux.sh      # Ubuntu 22.04 / 24.04 (apt)
scripts/bootstrap-bare-metal-macos.sh      # macOS (brew)
scripts/bootstrap-bare-metal-windows.ps1   # Windows (Chocolatey)

# Verify all required tools resolve
make doctor
```

Inside the dev shell (or once bare-metal install completes):

```bash
# C++ — fast inner loop
cmake --preset dev-debug
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure

# Production build (BLAS + OpenMP)
cmake --preset blas-omp && cmake --build --preset blas-omp -j
```

## Workflow

We follow the **roadmap → Codex review → implement → Codex review →
tag** loop documented in [`docs/dev/workflow.md`](docs/dev/workflow.md).
In short:

1. Pick or open the relevant issue template (see the table above).
2. Implement in small atomic commits. Each commit must keep CI green.
3. After every PR, Codex reviews the diff. If Codex disagrees with the
   implementation, the author adds context once; if Codex still
   disagrees, **Codex wins** and the author revises.
4. Codex transcripts are checked into `docs/reviews/phase-N/` for audit.

## Coding style

- **C++** — C++17, no STL in `cpp/include/n4m/*.h`, no exceptions
  crossing the C ABI. `clang-format` and `clang-tidy` configs are in
  the repo root; CI enforces both.
- **C** — public headers in `cpp/include/n4m/*.h` are `extern "C"`
  clean and compile under `-std=c11 -Wpedantic`.
- **Python** — type-hinted, formatted with `ruff format`, linted with
  `ruff check` (rules: `E`, `F`, `I`, `B`, `UP`, `SIM`).
- **R** — tidyverse style for the parity generators; base R for the
  in-tree binding.
- **MATLAB / Octave** — CI exercises Octave only (per
  `docs/REFACTOR_PLAN.md` §1.1ter). Differences between MATLAB and
  Octave are declared per-symbol in `bindings/matlab/COMPAT.md`.
- **Commit messages** — Conventional Commits (`feat:`, `fix:`, `chore:`,
  `docs:`, `test:`, `build:`, `ci:`, `refactor:`). Scope when relevant
  (e.g. `feat(abi):`, `fix(parity):`).

## Adding a new PLS variant

The full procedure is encoded in
[`.github/PULL_REQUEST_TEMPLATE/method-add.md`](.github/PULL_REQUEST_TEMPLATE/method-add.md).
At a glance:

1. **Catalog**: create `catalog/methods/<id>.yaml` (one file per method;
   schema in [`catalog/README.md`](catalog/README.md)).
2. **C++ impl** under `cpp/src/core/<category>/<id>.{cpp,hpp}` + the
   `extern "C"` wrapper under `cpp/src/c_api/c_api_<category>.cpp`.
3. **ABI snapshot** regen (automatic on CI; verify locally with `nm`).
4. **Reference & parity**: pin the external reference's environment
   under `parity/environments/<env_id>/`, write
   `parity/scenarios/<id>.yaml`, regenerate snapshots with
   `make snapshot METHOD=<id>`, calibrate tolerances in
   `catalog/methods/<id>.yaml`, get `make parity METHOD=<id>` green.
   For `paper_only` methods, populate the `self_consistency:` block
   instead.
5. **Bindings**: declare `bindings.<lang>.raw` and
   `bindings.<lang>.idiomatic[]` in the catalog YAML; run
   `python catalog/scripts/render_api.py` to regenerate wrappers
   across every active language. **No hand-written per-method
   per-binding code.**
6. **Docs**: write `examples/canonical/<id>.yaml` (per-binding examples
   are auto-generated); the Sphinx method page picks up the parity
   badge and timing mini-table from `dashboard.json`.

## Adding a new language binding

The full procedure is encoded in
[`.github/PULL_REQUEST_TEMPLATE/binding-add.md`](.github/PULL_REQUEST_TEMPLATE/binding-add.md).
At a glance:

```bash
make new-binding LANG=<lang>     # copies bindings/skeletons/<lang>/
```

Bindings are intentionally thin and **must**:

- Call into `libn4m` via the public C ABI only (never any internal
  C++ symbol).
- Pass the `bindings/conformance/` suite (language-agnostic JSON
  fixtures covering happy path + error path).
- Declare at least one framework profile in the catalog (e.g.
  `python_sklearn`, `r_formula`, `matlab_classdef`); per-profile
  wrappers are emitted from `bindings/profiles/templates/<id>.j2` by
  `render_api.py`.
- Inter-binding parity vs C++ raw: `rmse_rel < 1e-12`.

See [`bindings/SPEC.md`](bindings/SPEC.md) for the normative contract
(library loading, ABI version probe, matrix marshalling, error
handling, memory ownership).

## Reporting bugs

Please use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`.
For numerical-correctness reports, file a
[`parity-discrepancy.yml`](.github/ISSUE_TEMPLATE/parity-discrepancy.yml)
issue with a minimal reproducer (X, Y, configuration), the expected
reference values, and the observed `n4m` output. Without that we can't
triage.

## Releases

Maintainers only. See [`docs/dev/release_process.md`](docs/dev/release_process.md).
