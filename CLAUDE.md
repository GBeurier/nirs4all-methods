# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`nirs4all-methods` (`n4m`) is a portable PLS/NIRS engine written in **C++17** with a **stable C ABI** (`libn4m`) and **thin first-class bindings** for many languages. The same numerical core powers every binding: every binding only translates native objects into `n4m_matrix_view_t` / `n4m_*` calls — it never owns numerical logic.

The founding refactor document is [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md). The canonical architecture spec is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); when present, it overrides this file.

**Note on naming**: the project completed its `p4a`/`pls4all` → `n4m`/`nirs4all-methods` token-level rename in Phase A. `pls4all` remains as a **packaging subset name** (the slim PLS-only re-export shipped on PyPI/CRAN) — the underlying binary is always `libn4m`. The historical project URL `github.com/GBeurier/pls4all` is preserved until the GitHub repo itself is renamed.

## Build commands

This is a CMake project (≥ 3.21, Ninja). All builds go through **CMake presets** defined in `CMakePresets.json` — never hand-roll cmake flags.

```bash
# Local dev (fastest inner loop)
cmake --preset dev-debug          # or dev-release
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure

# Production (BLAS + OpenMP)
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

`build/<preset>/cpp/cli/n4m_cli --selfcheck` runs an end-to-end smoke; `--abi-info` prints the ABI version triple.

## Dev shell (Phase A20-A24)

```bash
# Easiest: VS Code + Docker (or Codespaces): "Reopen in Container"
make bootstrap         # detects VS Code + Docker; falls back to bare-metal

# Compose-based shell (no VS Code)
make shell             # docker compose -f .devcontainer/docker-compose.yml run --rm dev bash

# Verify required tools
make doctor

# Bare-metal best-effort install
scripts/bootstrap-bare-metal-linux.sh    # Ubuntu 22.04 / 24.04 (apt)
scripts/bootstrap-bare-metal-macos.sh    # macOS (brew)
scripts/bootstrap-bare-metal-windows.ps1 # Windows (Chocolatey)
```

The `Makefile` is a **thin convenience layer over CMake** — `make build PRESET=<p>`
and `make test PRESET=<p>` (default `dev-debug`) wrap the preset commands above.
The `parity`, `snapshot`, `dashboard-*`, and `new-binding` targets are **Phase-gated
stubs** that print a "not yet wired" message and exit 0 — they are placeholders
documenting the future interface (Phases C/D/F-prep), not working commands yet.

## Test commands

```bash
# C++ unit + ABI + property tests (doctest binary `n4m_tests`)
ctest --preset dev-release --output-on-failure

# Run a single C++ test file: doctest filter on the binary
./build/dev-release/cpp/tests/n4m_tests --test-case="*aom*"

# Python binding tests
PYTHONPATH=bindings/python/src python -m pytest bindings/python/tests -q

# R binding parity
Rscript bindings/r/test_parity.R

# Octave / MATLAB parity (CI runs Octave; see bindings/matlab/COMPAT.md)
cd bindings/matlab && octave --no-gui --eval "addpath(genpath('.')); n4m_smoke_test"

# Cross-binding orchestrator (parity + timing). Registry-driven.
python benchmarks/cross_binding/orchestrator.py \
  --algorithms pls pcr --registry-cells --threads 1 \
  --libn4m-build blas-omp --n-runs 2 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/audit.csv --force

# Full registry sweep, sequential cells, 8 native threads per cell.
# BENCH_SKLEARN_N_JOBS overrides sklearn-style references that expose n_jobs.
BENCH_SKLEARN_N_JOBS=8 python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --registry-cells --threads 8 --workers 1 \
  --libn4m-build blas-omp --n-runs 2 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/n4m_full_parity.csv --force

# Overnight matrix (registry cells + registry-declared externals)
benchmarks/cross_binding/run_overnight.sh
```

Full operational checklist: [`docs/dev/testing.md`](docs/dev/testing.md).

## Release / ABI gates

Three checks are **release blockers**; CI enforces them and any change that touches them needs the snapshots regenerated in the same PR:

```bash
# Version sync (header → all downstream manifests)
scripts/bump_version.sh --check

# ABI symbol surface — must match exactly. If a new public n4m_* symbol is
# intentional, regenerate the snapshot AND document in docs/abi/changes_log.md.
nm -D --defined-only build/dev-release/cpp/src/libn4m.so.<ver> \
  | awk '{print $3}' | sort -u \
  | diff -u cpp/abi/expected_symbols_linux.txt -

# SONAME / linkage sanity
readelf -d build/dev-release/cpp/src/libn4m.so.<ver> \
  | grep -E 'SONAME|NEEDED|RPATH|RUNPATH'
```

ABI versioning (`N4M_ABI_VERSION_*` in `cpp/include/n4m/n4m_version.h`) is **independent of project version** and bumps only on surface change. `n4m_check_abi_compatibility(header_major, header_minor)` is how bindings detect header/runtime skew.

## Architecture — load-bearing rules

These are the four hard constraints in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). PRs that violate them are rejected without further discussion:

1. **C++ internal, C ABI public.** `cpp/include/n4m/n4m.h` is `extern "C"` only — no `std::*`, no `Eigen`, no smart pointers, no exceptions crossing the boundary. C++ may throw internally; `cpp/src/c_api/` catches and translates to `n4m_status_t`.
2. **Zero mandatory deps.** Required toolchain is C++17 compiler + CMake + standard library. BLAS, OpenMP, CUDA, Emscripten, Android NDK, MATLAB MEX, doctest, vendored JSON parser are all optional or test-only. The reference CPU linalg backend is implemented from scratch in `cpp/src/core/`.
3. **Never free across the language boundary.** Two output APIs: caller-allocated (`n4m_predict(model, X, out_buf, out_size)`) and core-allocated + `n4m_array_free(arr)`. Error strings live in a context-owned thread-local buffer invalidated by the next call.
4. **Stride-aware everything.** `n4m_matrix_view_t` carries explicit `row_stride` / `col_stride` so NumPy row-major, R/MATLAB column-major, transposes and slice views never trigger a copy at the boundary.

## Python binding threading

Both Python packages (`n4m` and the slim `pls4all` subset) load libn4m with
`ctypes.CDLL`, not `ctypes.PyDLL`. `ctypes` releases the Python GIL during
native calls through that handle, so long-running C calls can run while other
Python threads make progress. The libn4m `Context` object itself is not
thread-safe: create one context per thread and do not share model/context
handles across threads unless the C ABI documents that exact use.

`n4m.Context.num_threads` and `pls4all.Context.num_threads` wrap
`n4m_context_get/set_num_threads`. The cross-binding benchmark also sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`BLIS_NUM_THREADS`, and `BENCH_THREADS` from `--threads`; sklearn-compatible
references use `BENCH_SKLEARN_N_JOBS`, falling back to `BENCH_THREADS`.
Use `--workers 1` for high `--threads` values to avoid `workers * threads`
oversubscription.

## Repository layout (post-Phase-A)

```
cpp/
  include/n4m/                    # THE public C ABI: n4m.h (umbrella) + pls.h
                                  # (PLS-domain umbrella, restored in A27) + 11
                                  # per-category headers (preprocessing, models,
                                  # selection, splitters, aom_pop, filters,
                                  # augmentation, diagnostics, transfer, context,
                                  # utilities).
  include/n4m/n4m_version.h       # Canonical version source of truth — read by CMake
  src/core/                       # C++17 numerical core (PLS variants, AOM/POP,
                                  # preprocessing, splitters, diagnostics, selection,
                                  # CV, metrics, linalg). Never installed standalone.
  src/c_api/                      # extern "C" wrappers (~30 .cpp files, grouped by
                                  # category). c_api_method_result.cpp is the ~4k LOC
                                  # registry dispatcher (the single biggest source file).
                                  # Single surface — no more c_api / c_api_n4m split.
  cli/n4m_cli.cpp                 # `n4m_cli` — --version / --abi-info / --selfcheck
  tests/                          # doctest binary `n4m_tests` (post-A5: ex tests_n4m/)
  abi/expected_symbols_*.txt      # ABI snapshots (per-platform), diffed in CI

bindings/
  python/src/n4m/                 # Full n4m Python binding (Phase F-bootstrap target)
  python/src/pls4all/             # Mature pls4all subset (slim PLS-only re-export)
  r/n4m/                          # CRAN n4m package (post-A10 rename from bindings/r/pls4all/)
  matlab/                         # +pls4all classdef package + MEX dispatcher
                                  # (n4m_*_mex entry points). COMPAT.md documents
                                  # MATLAB-vs-Octave divergences (CI runs Octave only)
  js/                             # JS / WASM binding (Emscripten)
  julia/, jni/, android/, octave/ # Active bindings
  _archive/                       # Frozen PoC bindings (go, rust, dotnet, lua, nim, ruby)

benchmarks/
  parity_timing/registry.py       # Legacy canonical method catalog (~10k LOC).
                                  # Still drives full parity/timing until Phase B
                                  # migrates registry data into YAML.
  cross_binding/orchestrator.py   # Dual-gate (binding parity + reference parity) runner

catalog/                          # Phase B catalog source. catalog/methods/<id>.yaml
                                  # now exists for all 160 legacy method rows.
                                  # During transition, edit catalog/methods.yaml and
                                  # regenerate split files with split_legacy_methods.py.
parity/                           # Reference fixture generators + comparator
roadmap/phase-*.md                # Phase-by-phase implementation roadmaps
docs/REFACTOR_PLAN.md             # Post-merge refactor founding document
docs/                             # Sphinx site source
.github/ISSUE_TEMPLATE/           # 8 issue forms (method-request, binding-request, ...)
.github/PULL_REQUEST_TEMPLATE/    # 7 PR templates (method-add, binding-add, parity-fix, ...)
.github/workflows/                # CI: ci.yml, abi-check.yml, parity-gate.yml,
                                  # sanitizers.yml, coverage.yml, release-*.yml,
                                  # version-sync.yml, docs.yml
.devcontainer/                    # devcontainer.json + Dockerfile + docker-compose +
                                  # post-create.sh — Phase A20 dev shell
.github/codespaces/               # Codespaces mirror of the devcontainer
```

## CMake target model

| Target            | Type   | Purpose                                                    |
| ----------------- | ------ | ---------------------------------------------------------- |
| `n4m_core`        | OBJECT | Internal C++17 implementation. Never installed.            |
| `n4m_c`           | SHARED | Public C ABI — `libn4m.{so,dll,dylib}`.                    |
| `n4m_c_static`    | STATIC | Same surface, statically linked (Android predict-only AAR).|
| `n4m_tests`       | EXE    | doctest binary, links core + c_api.                        |
| `n4m_cli`         | EXE    | `--version`, `--abi-info`, `--selfcheck`.                  |

Symbol visibility is hidden by default; only `N4M_API`-decorated declarations are exported. On MSVC a `.def` file additionally drives exports. The Linux version script `cpp/src/c_api/n4m_linux.map` carries the `N4M_1` ABI version tag.

## Development workflow (Codex review loop)

From [`docs/dev/workflow.md`](docs/dev/workflow.md) and `CONTRIBUTING.md`:

```
1. Author opens the matching issue template under .github/ISSUE_TEMPLATE/.
2. Maintainer triages; if accepted, the work lands via the matching PR
   template under .github/PULL_REQUEST_TEMPLATE/.
3. Codex reviews via `codex exec` (or the project-scoped MCP server in .mcp.json).
4. Author applies corrections. On disagreement, author gives Codex more context
   once; if Codex still disagrees, **Codex wins** and the author revises.
5. The corrected diff lands on `main`.
6. Codex transcripts are checked into docs/reviews/phase-N/.
```

Codex review invocation (manual, until the MCP server is wired in):
```bash
codex exec -C $REPO -s read-only --skip-git-repo-check \
    -c reasoning_effort=medium \
    -o /tmp/review.md "$(cat review_prompt.txt)"
```

## Adding a new method / binding / subset

See the matching PR template under `.github/PULL_REQUEST_TEMPLATE/`:
- `method-add.md` — new method
- `method-update.md` — behavioural change to existing method
- `external-reference.md` — alternate / pinned version of an external reference
- `binding-add.md` — new language binding
- `binding-update.md` — new idiomatic profile / wrapper update
- `subset.md` — new packaging subset
- `parity-fix.md` — numerical parity discrepancy fix

Each template embeds the §2.10 invariants checklist the reviewer (human + Codex) verifies.

## Style and conventions

- **C++**: C++17, no STL in `cpp/include/n4m/*.h`, no exceptions crossing the C ABI. `clang-format` and `clang-tidy` configs at repo root; CI enforces both.
- **C**: public headers `extern "C"`-clean, compile under `-std=c11 -Wpedantic`.
- **Python**: type-hinted, `ruff format`, `ruff check` (rules: `E`, `F`, `I`, `B`, `UP`, `SIM`).
- **R**: tidyverse style for the parity generators; base R for the binding.
- **MATLAB / Octave**: target the intersection; CI runs Octave; divergences declared in `bindings/matlab/COMPAT.md`.
- **Commits**: Conventional Commits with scope when relevant (`feat(abi):`, `fix(parity):`).

## Things that will trip up a fresh agent

- The token-level `p4a` → `n4m` rename is **complete**. If you see `p4a_*` anywhere outside `bindings/_archive/`, `CHANGELOG.md`, or historical merge logs under `docs/merge/` / `docs/reviews/`, that's a regression — flag it.
- `pls4all` references still exist intentionally in: (a) the slim subset package name (`bindings/python/src/pls4all/`, `catalog/subsets/pls4all.yaml`), (b) the github.com URL, (c) historical Authors@R contributor tag. Other references are bugs.
- `benchmarks/parity_timing/registry.py` is the legacy method catalog (~10k LOC). Phase B replaces it with `catalog/methods/<id>.yaml` (one YAML per method). When touching the registry, check whether the same fact already exists in the catalog or vice versa.
- `catalog/scripts/validate.py` currently reports ABI symbol mismatches as warnings unless `--strict-abi` is passed. The split catalog still contains many auto-discovered symbol guesses; reconcile them against `cpp/abi/expected_symbols_*.txt` before enabling strict ABI in CI.
- `catalog/scripts/render_api.py` renders generated metadata under `build/catalog/rendered_api/`; it does not overwrite hand-written binding APIs yet.
- The dashboard distinguishes **binding parity** (internal n4m bindings) from **reference parity** (vs external libs). They aren't interchangeable.
- `parity/fixtures/` will be migrated to `parity/snapshots/current/<method_id>/<scenario_id>.json` in Phase C; both layouts may briefly coexist.
- ABI symbol diff fails closed. If you add a public `n4m_*` symbol, regenerate `cpp/abi/expected_symbols_*.txt` for **all three platforms** in the same PR and update `docs/abi/changes_log.md`.
- The build wires `n4m_core` once in `cpp/src/CMakeLists.txt`; `cpp/src/n4m_targets.cmake` extends it via `target_sources()` with the donor-side sources and defines `n4m_c` (the single shared library) — don't reintroduce a second `add_library(n4m_core ...)`.
