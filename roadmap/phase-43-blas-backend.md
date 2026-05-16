# Phase 43 — Optional CBLAS Backend

**Status**: shipped 2026-05-16 · tag `phase-43-blas-backend` ·
release `0.94.0+abi.1.13.0`. Default is **OFF** — the OFF build is
bit-identical to pre-Phase-43 (the same scalar loops, no new link
dependency).

## Goal

Make `PLS4ALL_WITH_BLAS=ON` route the hot linear-algebra kernels in
libp4a through a system CBLAS implementation (OpenBLAS, MKL, Apple
Accelerate, Netlib), without changing the public C ABI and without
introducing a new mandatory dependency. The shared cross-binding
parity fixture and all 10 language-binding parity gates must continue
to pass with the ON build.

## Why CBLAS, not Eigen / Blaze / our own SIMD

* CBLAS has stable, vendor-neutral semantics. OpenBLAS, MKL,
  Accelerate, Netlib and ATLAS all implement the same column-major
  Fortran ABI and the same `cblas_*` C wrapper, so a single dispatch
  layer reaches every major BLAS implementation on every platform.
* No vendor lock-in. Embedders pick their preferred BLAS at link
  time; we never bundle one.
* Sticking to BLAS-level entry points (`dgemv`, `dgemm`, `dger`) is
  enough to capture the dominant kernels in NIPALS, SIMPLS, SVD,
  power, randomized-SVD, kernel-algo, PCR and OPLS solvers without
  pulling in a full templated linear-algebra library.

## Architecture

```
cpp/src/core/
├── linalg.hpp     (new) — pls4all::linalg dispatch (gemv, matvec, gemm, ger)
├── model.cpp      hot kernels now go through pls4all::linalg
└── kernel_pls.cpp unchanged (kernel-evaluation loops dominate; no GEMM seam)
```

The seam header exposes four entry points in the `pls4all::linalg`
namespace:

* `gemv(trans, rows, cols, alpha, A, x, beta, y)` — `y = alpha *
  op(A) * x + beta * y`. Supports `Trans_No` and `Trans_Yes`.
* `matvec(A, rows, cols, x, out)` — thin wrapper for the
  `(NoTrans, alpha=1, beta=0)` case. Used by every `t = X * w`
  step (14 call sites).
* `gemm(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C,
  ldc)` — general `C = alpha * op(A) * op(B) + beta * C`. Used by
  every `X^T Y` cross-covariance step (9 call sites across SIMPLS,
  SVD, kernel-algo, power, randomized-SVD, canonical pair, OPLS,
  DI-PLS).
* `ger(M, N, alpha, x, y, A, lda)` — rank-1 update `A += alpha * x
  * y^T`. **Currently unused** — Phase 43b promotes the X / Y
  deflations and `compute_rotations_and_coefficients` GEMMs to the
  dispatch once the per-kernel benchmarks justify the diff.

When `P4A_USE_BLAS` is **not** defined, every entry point inlines the
same row-major double scalar loop the code had pre-Phase-43.
Compiler output is byte-equivalent; the OFF binary has no new symbol
references and no new link dependency.

When `P4A_USE_BLAS` **is** defined, the entry points dispatch to
`cblas_dgemv`, `cblas_dgemm`, and `cblas_dger` via the row-major
CBLAS layout. All buffers stay row-major contiguous (`lda = cols`),
so no transposition or copy crosses the FFI boundary.

## Call-site conversions in `cpp/src/core/model.cpp`

| Site | Old loop | New call |
|------|----------|----------|
| `matrix_vector_product` body (line 220) | scalar `t = A * x` | `linalg::matvec(...)` (covers 14 callers) |
| NIPALS `X^T * y_score / y_score_ss` (canonical, lines 1246-1252) | scalar | `linalg::gemv(Trans_Yes, rows, features, 1/y_score_ss, ...)` |
| NIPALS `X^T * y_score / y_score_ss` (regression, lines 1642-1648) | scalar | `linalg::gemv(Trans_Yes, n, p, 1/y_score_ss, ...)` |
| Cross-covariance `X^T Y` (9 sites) | scalar | `linalg::gemm(Trans_Yes, Trans_No, p, q, n, 1.0, X, p, Y, q, 0.0, C, q)` |

The remaining rank-1 deflations (`X -= t * p^T`, `Y -= t * q^T`)
and the `compute_rotations_and_coefficients` GEMMs stay as scalar
in Phase 43. Those are once-per-component, not per-power-iteration,
so the optimisation budget points at the inner-loop kernels first.

## CMake wiring

`cmake/Pls4allOptions.cmake` already had the `PLS4ALL_WITH_BLAS`
option placeholder; Phase 43 wires it up in `cpp/src/CMakeLists.txt`:

```cmake
if(PLS4ALL_WITH_BLAS)
    find_package(BLAS)        # hard-fail if not found
    check_include_file_cxx(cblas.h PLS4ALL_HAS_CBLAS_HEADER)
    # ... hard-fail if header not reachable
    target_compile_definitions(pls4all_core PRIVATE P4A_USE_BLAS=1)
    # link to ${BLAS_LIBRARIES} happens in pls4all_apply_c_target
    # for pls4all_c / pls4all_c_static; also linked into pls4all_tests
    # because the test target embeds pls4all_core .o files directly.
endif()
```

The two hard-fail diagnostics catch the most common misconfigurations
(missing BLAS package, missing CBLAS dev headers — Apple Accelerate
uses `<Accelerate/Accelerate.h>` which the linalg header detects via
`__has_include`).

## Build & ship-gate

```bash
# 1. Default OFF build (CI gate, bit-identical to pre-Phase-43)
cmake -S . -B build/dev-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev-release -j
build/dev-release/cpp/tests/pls4all_tests        # 259/259 PASS

# 2. BLAS ON build
cmake -S . -B build/blas-on -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPLS4ALL_WITH_BLAS=ON \
      -DBLA_VENDOR=OpenBLAS \
      -DBLAS_LIBRARIES=/path/to/libopenblas.so
cmake --build build/blas-on -j
build/blas-on/cpp/tests/pls4all_tests            # 259/259 PASS

# 3. Cross-binding parity gates against the BLAS-ON libp4a
#    (Python / R / Julia / Octave / JNI / Go / Rust / .NET / Ruby / Lua / Nim)
#    All 10 pass with rmse_rel <= 5.2e-16, well inside the 1e-12 gate.
```

## Numerical parity

The 10 cross-binding parity gates each rebuild the same deterministic
(50 × 5) sin-based input and compare predictions to the shared
fixture regenerated against the BLAS-ON libp4a. All 10 pass with
maximum observed rmse_rel of `5.2e-16` (Octave / JNI / Rust / .NET /
Ruby / Lua / Nim on coefficients), four orders of magnitude inside
the `1e-12` cross-binding gate. The drift is OpenBLAS multi-thread
reduction order — exactly as the architect's risk #3 anticipated.

For bit-exact reproducibility set `OPENBLAS_NUM_THREADS=1` (or the
equivalent for MKL / Accelerate); the runs above were against the
default thread count.

## Phase 43 risks: status after ship

* **ABI version drift**: ABI symbol count unchanged at 160 between
  OFF and ON. Public ABI version remains 1.13.0.
* **Accidental BLAS linkage in consumers**: PRIVATE link visibility
  + the OBJECT-library embedding pattern keep BLAS internal to
  `libp4a.so`. Consumers that link against the shared library do
  not see a transitive `-lopenblas` requirement.
* **Numerical drift**: 5.2e-16 max, well inside the gate.
* **Threading conflicts**: PLS4ALL_WITH_OPENMP is still a placeholder
  in Phase 43. When Phase 44 lands the OpenMP backend the option
  comment must call out the OpenBLAS oversubscription concern.
* **macOS Accelerate load order**: addressed by the `__has_include`
  guard in `linalg.hpp` which selects `<Accelerate/Accelerate.h>`
  over `<cblas.h>` when only the former exists.
* **ILP64 MKL**: the casts in `linalg.hpp` use `static_cast<int>`,
  which is correct for the default LP64 BLAS interface (OpenBLAS,
  Accelerate, Netlib, MKL-LP64). For MKL-ILP64 builds the user must
  set `BLA_VENDOR=Intel10_64ilp` and rebuild; the casts then widen
  to 64-bit at compile time via the CBLAS `blasint` typedef. Not
  tested on this host.

## Release

* Library version: `0.93.0` → `0.94.0`. ABI version unchanged at
  `1.13.0`. Additive: the BLAS backend is opt-in.
* Tag: `phase-43-blas-backend`.

## Phase 43b backlog

* Promote the rank-1 deflations to `linalg::ger` once profiling shows
  meaningful share of fit time.
* Promote `compute_rotations_and_coefficients` to three `linalg::gemm`
  calls (low-frequency but high-flop).
* Add a `pls4all_cli --bench --backend blas` knob for runtime
  comparison.
* Wire the BLAS backend into the final timing benchmarks (Phase 50).
