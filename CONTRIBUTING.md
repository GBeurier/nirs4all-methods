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
   numerical path MUST update the relevant fixtures/snapshots and keep
   the per-method tolerance (declared alongside its `MethodSpec` in
   `benchmarks/parity_timing/registry.py`) green — verify with
   `parity/scripts/per_method_parity.py <id>`. Paper-only methods use
   the self-consistency gate (`make parity-paper-only METHOD=<id>`)
   instead of an external reference.
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

## Adding a new method

A method's facts live in **~6 hand-edited surfaces**, and they are not generated
from one source — you edit each and keep them consistent. The catalog is a
*manifest* (ABI symbols, translation units, publications); the **registry**
(`benchmarks/parity_timing/registry.py`) is what actually drives parity and
timing; binding wrappers are hand-written. The PR checklist is
[`.github/PULL_REQUEST_TEMPLATE/method-add.md`](.github/PULL_REQUEST_TEMPLATE/method-add.md).

1. **C++ core + C ABI.** Implement under
   `cpp/src/core/<category>/<id>.{c,cpp,h}` (C++17 internal; nothing from STL /
   Eigen / no exceptions cross the boundary). Add the `extern "C"` wrapper in
   `cpp/src/c_api/c_api_<category>.cpp` — or, for a `MethodResult`-style op, a
   case in the dispatcher `cpp/src/c_api/c_api_method_result.cpp`. Declare the
   public `N4M_API` symbols in `cpp/include/n4m/<category>.h`. The build picks up
   new sources via `CONFIGURE_DEPENDS` globs; only touch
   `cpp/src/n4m_targets.cmake` for a new vendored/optional dependency.
2. **ABI snapshot (release blocker).** Regenerate
   `cpp/abi/expected_symbols_{linux,macos,windows}.txt` for **all three**
   platforms and update `docs/abi/changes_log.md`. CI fails closed on any
   undocumented `n4m_*` symbol. Verify locally:
   `nm -D --defined-only build/dev-release/cpp/src/libn4m.so.<ver> | awk '{print $3}' | sort -u | diff -u cpp/abi/expected_symbols_linux.txt -`.
3. **Tests.** doctest cases in `cpp/tests/test_<category>.cpp` covering
   create/apply + null/shape errors **and that a non-trivial input is actually
   transformed** (determinism alone passes a silent no-op). For
   deterministic-given-seed methods, freeze a self-fixture under
   `parity/fixtures/<id>_v1.json` (IEEE-754 big-endian hex) and replay it with
   `assert_close`. Run `ctest --preset dev-release`; one method:
   `./build/dev-release/cpp/tests/n4m_tests --test-case="*<id>*"`.
4. **Reference & parity.** Add a `MethodSpec` to
   `benchmarks/parity_timing/registry.py` — this is where the external
   reference, `adapted_params`, and `rmse_rel_tol` actually live. The reference
   must be an installable, documented donor (`parity/REFERENCES.md`; confirm with
   `parity/scripts/check_references.py`). Verify per-method:
   `python parity/scripts/per_method_parity.py <id> --reference <ref> --tol 1e-8`
   (Gate A = each binding vs the C++ core at 1e-12; Gate B = C++ core vs the
   reference at 1e-8). Paper-only methods use
   `make parity-paper-only METHOD=<id>` (self-consistency) instead.
5. **Catalog manifest.** Add the row to `catalog/methods.yaml`, regenerate the
   split file (`python catalog/scripts/split_legacy_methods.py` →
   `catalog/methods/<id>.yaml`), and keep `abi_symbols` / `tu` / `headers` /
   `parity` / `publications` matching the code. `python catalog/scripts/validate.py`
   must pass. (Schema keys are `catalog_version`, `method_id`, `family`,
   `category`, `abi_symbols`, `tu`, `headers`, `parity`, `bench`, `bindings`,
   `publications`, `notes` — there is no `status` / `math` / `parameters` /
   `self_consistency` / `scenarios` field.)
6. **Bindings & docs.** Hand-write the Python wrapper in
   `bindings/python/src/n4m/python.py` (+ argtypes in `_ffi_decls.py`; a
   sklearn-style estimator in `bindings/python/src/n4m/sklearn/` if an idiomatic
   profile is wanted), and other language bindings only if in scope.
   `render_api.py` renders metadata, **not** package APIs, so wrappers are not
   auto-generated. Add a `CHANGELOG.md` entry; for dashboard methods, register
   the op in `benchmarks/cross_binding/donor_ops.py` and refresh
   `docs/_static/bench-data.json`.

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
