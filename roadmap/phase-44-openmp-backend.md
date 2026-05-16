# Phase 44 — Optional OpenMP Backend

**Status**: shipped 2026-05-16 · tag `phase-44-openmp-backend` ·
release `0.95.0+abi.1.13.0`. Default is **OFF**.

## Goal

Add `PLS4ALL_WITH_OPENMP=ON` as an additive build option that
parallelises the hot inner loops in `model.cpp` via `#pragma omp
parallel for`. The OFF build is bit-identical to pre-Phase-44.
Public ABI stays at 1.13.0. All four configurations (OFF/OFF,
BLAS-only, OMP-only, BLAS+OMP) must build and pass the 262 C++
tests plus the 10 cross-binding parity gates.

## Why inner loops only

The code-explorer audit identified two classes of OpenMP win:

1. **Inner-loop parallelism**: rank-1 deflations, dot-product
   loading computations, `fill_prediction`/`fill_transform`. Each
   loop has disjoint writes per iteration, no `ctx` access, no
   shared mutable state — clean `parallel for` candidates.
2. **Outer-loop parallelism**: CV folds, EMCUVE ensembles, AOM
   operator sweeps. These multiply available parallelism by ~5-100×
   on top of inner-loop wins, but require per-thread `Model`
   allocation and accumulator reductions. Higher implementation
   risk.

Phase 44 ships **class 1 only** — 21 pragma sites in `model.cpp`
covering every solver's deflation and loading kernels plus the two
fill helpers. Class 2 is deferred to Phase 44b after benchmark
data confirms it is worth the per-thread Model machinery.

## Architecture

```
cpp/src/core/
├── parallel.hpp   (new) — P4A_PARALLEL_FOR_STATIC macro
├── model.cpp      21 pragma sites
└── linalg.hpp     (Phase 43) — left untouched; BLAS already
                    parallelises internally, no overlap
```

The macro expands to:
* `_Pragma("omp parallel for schedule(static)")` on GCC / Clang /
  AppleClang.
* `__pragma(omp parallel for schedule(static))` on MSVC.
* Empty token sequence when `P4A_USE_OPENMP` is not defined — no
  `<omp.h>` include, no preprocessor noise, no symbol changes.

`schedule(static)` is used uniformly: every annotated loop has
equal work per iteration so static partitioning has zero runtime
overhead vs dynamic.

## Annotated sites in `model.cpp`

| Pattern | Outer loop | Sites |
|---------|-----------|-------|
| X deflation `xk[row*p+f] -= x_score[row] * x_loadings[f]` | `for (row)` | 8 |
| Y deflation `yk[row*q+t] -= x_score[row] * y_loadings[t]` | `for (row)` | 7 (incl. canonical NIPALS `y_scores` variant) |
| Fused X+Y deflation (orthogonal-scores, OPLS) | `for (row)` | 2 |
| X-loadings `for (f) { sum = sum X x_score; x_loadings[f] = sum / x_score_ss; }` | `for (feature)` | 2 |
| `fill_prediction` / `fill_transform` | `for (i)` | 2 |

All 21 sites verified by the Codex reviewer to have thread-disjoint
writes and zero `ctx` access inside the parallel region.

## CMake wiring

```cmake
# Top-level: hoisted so OpenMP::OpenMP_CXX is visible in both
# cpp/src/ and cpp/tests/.
if(PLS4ALL_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# cpp/src/CMakeLists.txt: propagate define + link into pls4all_core,
# pls4all_c, pls4all_c_static. Hard-fail diagnostic if find_package
# fails. Warning when BLAS+OMP both on (libgomp deadlock advisory).

# cpp/tests/CMakeLists.txt: mirror define + link into pls4all_tests
# because the test embeds pls4all_core OBJECTs directly.
```

The `pls4all_apply_c_target()` helper handles both BLAS and OMP
propagation uniformly — same pattern as Phase 43.

## C ABI hook

```c
// cpp/src/c_api/c_api_version.cpp
case P4A_BACKEND_OPENMP:
#if defined(P4A_USE_OPENMP)
    return 1;
#else
    return 0;
#endif
```

No new exported symbols. ABI version stays at 1.13.0.

## Ship gate

All four configurations build and pass 262/262 C++ tests:

| Config | CMake flags | Status |
|--------|-------------|--------|
| OFF/OFF | (default) | 262/262 |
| BLAS only | `-DPLS4ALL_WITH_BLAS=ON -DBLAS_LIBRARIES=...` | 262/262 |
| OMP only | `-DPLS4ALL_WITH_OPENMP=ON` | 262/262 |
| BLAS+OMP | both + `OPENBLAS_NUM_THREADS=1` runtime | 262/262 |

All 10 cross-binding parity gates pass against the OMP-only and
BLAS+OMP libp4a (fixture regenerated per config; bindings compare
against that config's fixture). ABI symbol count unchanged at 160
across all four builds.

## Codex review findings

The Codex reviewer surfaced three items; the actionable ones are
addressed in this ship:

* **MSVC `_Pragma` portability** — fixed in `parallel.hpp` with a
  `_MSC_VER` branch using `__pragma`.
* **OpenBLAS+libgomp deadlock** — addressed with a CMake
  `message(WARNING ...)` when both options are on, plus this
  documentation. The runtime advice is to set
  `OPENBLAS_NUM_THREADS=1` (or the MKL equivalent) in the
  consumer's environment.
* **TSAN incompatibility** — CMake `WARNING` when
  `PLS4ALL_ENABLE_TSAN` and `PLS4ALL_WITH_OPENMP` are both on:
  libgomp is not TSAN-instrumented and will produce false
  positives. Use a separate non-TSAN build for OpenMP testing.

## Known limits (Phase 44b backlog)

* **`p4a_context_set_num_threads` is not honored**: the public ABI
  exposes a thread-count setter (`Context::num_threads_`) that the
  OpenMP pragmas currently ignore. Honoring it requires routing the
  context through every annotated function so the pragmas can
  emit `omp_set_num_threads(...)` at entry. Phase 44b will plumb
  this; meanwhile users control parallelism via `OMP_NUM_THREADS`
  in the environment.
* **No outer-loop parallelism**: CV folds, EMCUVE ensembles, AOM
  operator sweeps — each multiplies parallelism by 5-100× on top
  of inner pragmas. Phase 44b after benchmarks justify the design
  cost of per-thread `Model` instances.
* **`linalg.hpp` reference path is not OMP-annotated**: the audit
  confirmed the inner reductions inside `gemv`/`gemm` are too short
  to amortise the OMP fork overhead. Routing through BLAS gets that
  parallelism for free when BLAS is on; routing through OMP would
  oversubscribe.

## Release

* Library version: `0.94.0` → `0.95.0` (additive, no ABI change).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-44-openmp-backend`.
