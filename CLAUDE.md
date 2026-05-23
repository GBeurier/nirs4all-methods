# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`pls4all` / `nirs4all-methods` is a portable PLS/NIRS engine written in **C++17** with a **stable C ABI** (`libn4m` / `libp4a`) and **thin first-class bindings** for many languages. The same numerical core powers every binding: every binding only translates native objects into `p4a_matrix_view_t` / `n4m_*` calls — it never owns numerical logic.

The canonical spec is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The full design contract is [`Direction_Technique.md`] (referenced by `CONTRIBUTING.md`); when present, it overrides this file.

**Note on naming**: the project is mid-rename from `p4a` / `pls4all` to `n4m` / `nirs4all-methods`. Both prefixes coexist (`cpp/include/pls4all/p4a.h` defines `N4M_*` macros; `cpp/abi_n4m/` holds the current snapshots; `cpp/c_api/` and `cpp/c_api_n4m/` both exist). Treat `n4m` as the new canonical surface and `p4a` as the legacy alias.

## Build commands

This is a CMake project (≥ 3.21, Ninja). All builds go through **CMake presets** defined in `CMakePresets.json` — never hand-roll cmake flags.

```bash
# Local dev (fastest inner loop)
cmake --preset dev-debug          # or dev-release
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure

# Production (BLAS + OpenMP). Requires CONDA_PREFIX with libopenblas.
cmake --preset blas-omp
cmake --build --preset blas-omp -j

# CI parity (one of: ci-linux-{gcc12,gcc13,clang16}-{release,debug},
#                   ci-macos-clang-{release,debug}, ci-macos-universal2,
#                   ci-windows-{msvc,mingw}-release, ci-windows-msvc-debug,
#                   ci-asan, ci-ubsan, ci-tsan, ci-asan_ubsan, ci-coverage)
cmake --preset ci-linux-clang16-release
ctest --preset ci-linux-clang16-release --output-on-failure

# Cross-platform targets (require EMSDK / ANDROID_NDK in env)
cmake --preset emscripten
cmake --preset android-arm64
```

`build/<preset>/cpp/cli/pls4all_cli --selfcheck` runs an end-to-end smoke; `--abi-info` prints the ABI version triple.

## Test commands

```bash
# C++ unit + ABI + property tests (doctest binary `pls4all_tests`)
ctest --preset dev-release --output-on-failure

# Run a single C++ test file: doctest filter on the binary
./build/dev-release/cpp/tests/pls4all_tests --test-case="*aom*"

# Python binding tests
PYTHONPATH=bindings/python/src python -m pytest bindings/python/tests -q

# Narrower sklearn parity smoke
bindings/python/scripts/sklearn_parity_gate.sh

# R binding parity
Rscript bindings/r/test_parity.R

# MATLAB / Octave parity (after `addpath(genpath('bindings/matlab'))`)
# See bindings/matlab/test/ and bindings/matlab/build_mex.m

# Lockfile structural check (no live library imports)
python -m benchmarks.parity_timing.lockfile --check

# Cross-binding orchestrator (parity + timing). The big knob is registry-driven.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls pcr --registry-cells --threads 1 \
  --libp4a-build blas-omp --n-runs 2 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/audit.csv --force

# Overnight matrix (registry cells + registry-declared externals)
benchmarks/cross_binding/run_overnight.sh
# Env overrides: FORCE=1 RERUN_FAILED=1 FULL_MATRIX=1 LIBP4A_BUILD=all
#                REFERENCE_BACKENDS=all THREADS="1 3 10" N_RUNS=40
#                PUBLISH_WEB=1 DEPLOY_PAGES=0
```

Full operational checklist: [`docs/dev/testing.md`](docs/dev/testing.md).

## Release / ABI gates

Three checks are **release blockers**; CI enforces them and any change that touches them needs the snapshots regenerated in the same PR:

```bash
# Version sync (header → all downstream manifests)
scripts/bump_version.sh --check

# ABI symbol surface — must match exactly. If a new public p4a_*/n4m_*
# symbol is intentional, regenerate the snapshot AND document in
# docs/abi/changes_log.md.
nm -D --defined-only build/dev-release/cpp/src/libn4m.so.<ver> \
  | awk '{print $3}' | sort -u \
  | diff -u cpp/abi_n4m/expected_symbols_linux.txt -

# SONAME / linkage sanity
readelf -d build/dev-release/cpp/src/libn4m.so.<ver> \
  | grep -E 'SONAME|NEEDED|RPATH|RUNPATH'
```

ABI versioning (`N4M_ABI_VERSION_*` in `cpp/include/pls4all/p4a_version.h`) is **independent of project version** and bumps only on surface change. `n4m_check_abi_compatibility(header_major, header_minor)` is how bindings detect header/runtime skew.

## Architecture — load-bearing rules

These are the four hard constraints in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). PRs that violate them are rejected without further discussion:

1. **C++ internal, C ABI public.** `cpp/include/pls4all/p4a.h` is `extern "C"` only — no `std::*`, no `Eigen`, no smart pointers, no exceptions crossing the boundary. C++ may throw internally; `cpp/src/c_api/` catches and translates to `p4a_status_t` / `n4m_status_t`.
2. **Zero mandatory deps.** Required toolchain is C++17 compiler + CMake + standard library. BLAS, OpenMP, CUDA, Emscripten, Android NDK, MATLAB MEX, doctest, vendored JSON parser are all optional or test-only. The reference CPU linalg backend is implemented from scratch in `cpp/src/core/`.
3. **Never free across the language boundary.** Two output APIs: caller-allocated (`p4a_predict(model, X, out_buf, out_size)`) and core-allocated + `p4a_array_free(arr)`. Error strings live in a context-owned thread-local buffer invalidated by the next call; no `p4a_free_string` exists by design.
4. **Stride-aware everything.** `p4a_matrix_view_t` carries explicit `row_stride` / `col_stride` so NumPy row-major, R/MATLAB column-major, transposes and slice views never trigger a copy at the boundary.

## Repository layout

```
cpp/
  include/pls4all/p4a.h           # THE public C ABI header (~1.8k LOC)
  include/pls4all/p4a_version.h   # Canonical version source of truth — read by CMake
  src/core/                       # C++17 numerical core (PLS variants, AOM/POP,
                                  # preprocessing, splitters, diagnostics, selection,
                                  # CV, metrics, linalg). Never installed standalone.
  src/c_api/                      # extern "C" wrappers. c_api_method_result.cpp
                                  # alone is ~4k LOC — the registry dispatcher.
  cli/p4a_cli.cpp                 # `pls4all_cli` — --version / --abi-info / --selfcheck
  tests/                          # doctest binary `pls4all_tests`
  abi_n4m/expected_symbols_*.txt  # ABI snapshots (per-platform), diffed in CI

bindings/
  python/                         # `pls4all` package (tier-1 ABI + tier-2 sklearn classes)
  r/pls4all/                      # base R formula+S3, pls-compatible, mdatools-compatible
  matlab/                         # +pls4all classdefs + MEX dispatcher (build_mex.m)
  go|rust|ruby|dotnet|lua|nim|    # Tier-2 PoCs (SIMPLS via native FFI)
  julia|jni|android|js|...

benchmarks/
  parity_timing/registry.py       # **Canonical method catalog** (~10k LOC). Source of
                                  # truth for what each binding/reference exposes.
  parity_timing/lockfile.py       # Structural --check for registry consistency
  cross_binding/orchestrator.py   # Dual-gate (binding parity + reference parity) runner
  cross_binding/run_overnight.sh  # Full matrix sweep
  cross_binding/render_docs.py    # Renders dashboard markdown

parity/                           # Reference fixture generators + tolerances
  python_generator/, r_generator/ # Produce parity/fixtures/*.json (deterministic seeded)
  tolerances.md                   # Per-(algorithm, reference-pair) tolerance table
  schema/                         # Fixture schema v1

catalog/methods.yaml              # Human-readable method catalog (155k LOC, generated)
roadmap/phase-*.md                # Phase-by-phase implementation roadmaps
docs/                             # Sphinx site source (architecture, abi, parity,
                                  # methods, bindings, dev workflow)
.github/workflows/                # CI: ci.yml, abi-check.yml, parity-gate.yml,
                                  # sanitizers.yml, coverage.yml, release-*.yml,
                                  # version-sync.yml, docs.yml
```

## CMake target model

| Target            | Type   | Purpose                                                    |
| ----------------- | ------ | ---------------------------------------------------------- |
| `pls4all_core`    | STATIC | Internal C++17 implementation. Never installed.            |
| `pls4all_c`       | SHARED | Public C ABI — `libn4m.{so,dll,dylib}` (legacy `libp4a`).  |
| `pls4all_c_static`| STATIC | Same surface, statically linked (Android predict-only AAR).|
| `pls4all_tests`   | EXE    | doctest binary, links core + c_api.                        |
| `pls4all_cli`     | EXE    | `--version`, `--abi-info`, `--selfcheck`.                  |

Symbol visibility is hidden by default; only `P4A_API`-decorated declarations are exported. On MSVC a `.def` file additionally drives exports.

## Development workflow (Codex review loop)

From [`docs/dev/workflow.md`](docs/dev/workflow.md) and `CONTRIBUTING.md`:

```
1. Author drafts roadmap/phase-N.md or a PR diff.
2. Codex reviews via `codex exec` (or the project-scoped MCP server in .mcp.json).
3. Author applies corrections. On disagreement, author gives Codex more context
   once; if Codex still disagrees, **Codex wins** and the author revises.
4. The corrected diff lands on `main`.
5. Codex transcripts are checked into docs/reviews/phase-N/.
```

Codex review invocation (manual, until the MCP server is wired in):
```bash
codex exec -C $REPO -s read-only --skip-git-repo-check \
    -c reasoning_effort=medium \
    -o /tmp/review.md "$(cat review_prompt.txt)"
```

## Adding a new PLS variant

1. Python adapter in `parity/python_generator/adapters/` (canonical fixture).
2. R adapter in `parity/r_generator/adapters/` against `pls` / `ropls` / `mixOmics` / `plsVarSel`; record tolerances in `parity/tolerances.md`.
3. Implementation in `cpp/src/core/` (organised by family: `pls/`, `aom_pop/`, `selection/`, `preprocessing/`, `diagnostics/`, etc.).
4. Surface through C ABI in `cpp/src/c_api/c_api_method_result.cpp` — prefer new **enum values** on `p4a_algorithm_t` / `p4a_solver_t` over new functions.
5. Numerical unit test in `cpp/tests/` (e.g. `test_extra_pls.cpp`) and parity fixture in `parity/fixtures/`.
6. Update `docs/abi/changes_log.md` and re-snapshot `cpp/abi_n4m/expected_symbols_*.txt` if the symbol surface changed.
7. Update the canonical catalog in `benchmarks/parity_timing/registry.py` and `catalog/methods.yaml`.

## Adding a new language binding

Bindings must:
- Call `libn4m` via the public C ABI only — never any internal C++ symbol.
- Implement at minimum: `version()`, `Config` builder, `fit(X, Y, cfg)`, `predict(model, X)`, `save(model, path)`, `load(path)`.
- Ship a parity test that loads the JSON fixtures from `parity/fixtures/` and asserts the same tolerances as the existing bindings.
- Document install + hello-version in `docs/bindings/<lang>.md`.

## Style and conventions

- **C++**: C++17, no STL in `cpp/include/pls4all/*.h`, no exceptions crossing the C ABI. `clang-format` and `clang-tidy` configs at repo root; CI enforces both.
- **C**: public headers `extern "C"`-clean, compile under `-std=c11 -Wpedantic`.
- **Python**: type-hinted, `ruff format`, `ruff check` (rules: `E`, `F`, `I`, `B`, `UP`, `SIM`).
- **R**: tidyverse style for the parity generators; base R for the binding.
- **Commits**: Conventional Commits with scope when relevant (`feat(abi):`, `fix(parity):`).

## Things that will trip up a fresh agent

- Don't grep for `p4a_` only — many newer symbols are `n4m_*`. The rename is in-flight; both are valid.
- `benchmarks/parity_timing/registry.py` is the single source of truth for "what does each binding expose against which external reference". Touching it has knock-on effects on the dashboard renderer (`benchmarks/cross_binding/render_docs.py`) and the lockfile check.
- The dashboard distinguishes **binding parity** (internal pls4all bindings) from **reference parity** (vs external libs). They aren't interchangeable — see `docs/dev/stabilisation_plan.md` for the gate semantics that landed.
- `parity/fixtures/` regeneration historically depended on an out-of-tree `AOM_v0` oracle. Clean clones can't reproduce some AOM/POP fixtures yet; check `docs/dev/stabilisation_plan.md` before assuming a fixture diff is a real regression.
- ABI symbol diff fails closed. If you add a public `p4a_*` / `n4m_*` symbol, regenerate `cpp/abi_n4m/expected_symbols_*.txt` for **all three platforms** in the same PR and update `docs/abi/changes_log.md`.
