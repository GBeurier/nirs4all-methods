# Contributing to pls4all

Thanks for considering a contribution. `pls4all` is a research-grade library going to production, so the bar is intentionally high — but the rules are simple and the test infrastructure does most of the enforcement automatically.

## Ground rules

1. **The canonical spec is [`Direction_Technique.md`](Direction_Technique.md).** Any PR that violates one of its decisions (C ABI, zero mandatory deps, no exceptions across the boundary, no STL in the public API, etc.) will be rejected without further discussion. Open an issue first if you think the spec itself needs to change.
2. **The parity gate is non-negotiable.** Any PR that changes a numerical path MUST update the relevant fixtures and keep the tolerances in [`parity/tolerances.md`](parity/tolerances.md) green.
3. **The ABI surface is monitored.** Any PR that adds, removes or renames an exported `p4a_*` symbol must regenerate `cpp/abi/expected_symbols_*.txt` and explain the change in `docs/abi/changes_log.md`.

## Development setup

Prerequisites: a C++17 compiler (GCC ≥ 12, Clang ≥ 16, or MSVC 2022), CMake ≥ 3.21, Ninja, Python ≥ 3.10 (for the parity generator), R ≥ 4.3 (optional, for the R parity generator), and `gh` CLI authenticated.

```bash
git clone https://github.com/GBeurier/pls4all.git
cd pls4all

# C++ — fast inner loop.
cmake --preset dev-debug
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure

# Python parity generator (optional unless you change fixtures).
python -m venv .venv && . .venv/bin/activate
pip install -e parity/python_generator/

# R parity generator (optional unless you change fixtures).
Rscript -e 'install.packages(c("pls", "ropls", "mixOmics", "plsVarSel"))'
```

## Workflow

We follow the **roadmap → Codex review → implement → Codex review → tag** loop documented in [`docs/dev/workflow.md`](docs/dev/workflow.md). In short:

1. Pick (or open) a phase under [`roadmap/`](roadmap/).
2. Get the phase roadmap reviewed by Codex and addressed before any code lands.
3. Implement in small atomic commits. Each commit must keep CI green.
4. After every PR, Codex reviews the diff. If Codex disagrees with the implementation, the author adds context once; if Codex still disagrees, **Codex wins** and the author revises.
5. Codex transcripts are checked into `docs/reviews/phase-N/` for audit.

## Coding style

- **C++** — C++17, no STL in `cpp/include/pls4all/*.h`, no exceptions crossing the C ABI. `clang-format` and `clang-tidy` configs are in the repo root; CI enforces both.
- **C** — public headers in `cpp/include/pls4all/*.h` are `extern "C"` clean and compile under `-std=c11 -Wpedantic`.
- **Python** — type-hinted, formatted with `ruff format`, linted with `ruff check` (rules: `E`, `F`, `I`, `B`, `UP`, `SIM`).
- **R** — tidyverse style for the parity generator scripts; base R for the binding.
- **Commit messages** — Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `build:`, `ci:`, `refactor:`). Scope when relevant (e.g. `feat(abi):`, `fix(parity):`).

## Adding a new PLS variant (Phases 1+)

1. Add a Python adapter in `parity/python_generator/adapters/` that produces the canonical fixture fields for the variant (coefficients, predictions, scores, loadings, …).
2. Add an R adapter in `parity/r_generator/adapters/` using `pls`, `ropls`, `mixOmics` or `plsVarSel` as the reference, and document the cross-implementation tolerance in `parity/tolerances.md`.
3. Implement the algorithm in `cpp/src/core/pls/` against the new fixtures.
4. Surface it through the C ABI in `cpp/src/c_api/c_api_model.cpp` — no new functions, only new enum values in `p4a_algorithm_t` / `p4a_solver_t`.
5. Add a numerical unit test in `cpp/tests/numerical/` and a parity test in `cpp/tests/property/`.
6. Update `docs/abi/changes_log.md` and re-snapshot the symbol surface if needed.

## Adding a new language binding (Phase 2+)

Bindings are intentionally thin. They must:

- Call into `libp4a` via the public C ABI only (never any internal C++ symbol).
- Provide at minimum: `version()`, `Config` builder, `fit(X, Y, cfg)`, `predict(model, X)`, `save(model, path)`, `load(path)`.
- Ship a parity test that loads the JSON fixtures and asserts the same tolerances as the existing bindings.
- Document install + hello-version in `docs/bindings/<lang>.md`.

## Reporting bugs

Please use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`. Numerical-correctness reports must include a minimal reproducer (X, Y, configuration) and the expected reference values (sklearn / R `pls` / nirs4all) — without that we cannot triage.

## Releases

Maintainers only. See [`docs/dev/release_process.md`](docs/dev/release_process.md).
