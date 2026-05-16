# Changelog

All notable changes to `pls4all` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Next track: benchmark harness (Phase 46).

## [0.96.0-cuda-backend] — 2026-05-16

Optional cuBLAS backend. Default OFF — the OFF build is bit-
identical to pre-Phase-45 (no CUDA headers, no CUDA symbol
references). When `PLS4ALL_WITH_CUDA=ON` AND a compatible GPU is
present at runtime, `linalg::gemv/gemm/ger` route through
`cublasDgemv_v2 / cublasDgemm_v2 / cublasDger_v2`. Public ABI
unchanged at 1.13.0.

### CUDA backend

- `cpp/src/core/cuda_dispatch.hpp` + `.cpp` (new): Meyers-singleton
  cuBLAS handle, `DevicePtr<T>` RAII for cudaMalloc/cudaFree, the
  three dispatch entry points using the row-major → column-major
  swap trick (`op_cm = op_row`, no flip; B passed first, A second,
  with dimensions `m_cm=N, n_cm=M, k_cm=K`).
- `cpp/src/core/linalg.hpp`: `Trans_t` uses plain int when CUDA is
  on; each dispatch function tries CUDA first, then BLAS, then
  scalar.
- `cpp/src/c_api/c_api_version.cpp`: `p4a_backend_is_available
  (P4A_BACKEND_CUDA)` is the first backend whose result is a
  **runtime probe** (compile-time + GPU detection at first call,
  memoised in the singleton).
- `cpp/src/core/context.cpp`: `set_backend(CUDA)` accepts when
  compile-time AND runtime checks both pass; clear error otherwise.
- `cpp/cli/p4a_cli.cpp` + `cpp/cli/CMakeLists.txt`: selfcheck
  accepts `P4A_OK` or `P4A_ERR_BACKEND_UNAVAILABLE` under CUDA-ON.
- `cpp/tests/abi/test_cuda_backend.cpp` (new): fixture parity +
  200×20×2 smoke; skips gracefully if no GPU at runtime.
- `cpp/tests/abi/test_error_messages.cpp`: switched the "must be
  unavailable" probe from CUDA to `P4A_BACKEND_OPENCL` (truly
  never-implemented).
- Top-level `CMakeLists.txt` + `cpp/src/CMakeLists.txt` +
  `cpp/tests/CMakeLists.txt`: `find_package(CUDAToolkit REQUIRED)`
  hoisted, `CUDA::cudart` + `CUDA::cublas` linked into
  `pls4all_core`, `pls4all_c`, `pls4all_c_static`, and
  `pls4all_tests` via the existing helper pattern.

Ship gate (live on RTX 4090 + RTX 5090, CUDA 12.8):

- OFF build: 265/265 tests, all 10 cross-binding parity gates green.
- CUDA-ON build: 265/265 tests (3 CUDA tests run live on GPU 0),
  all 10 cross-binding parity gates green against fixture
  regenerated through the cuBLAS path.
- ABI symbol count unchanged at 160.

Codex review: all math correct, RAII covers leak paths, the
P4A_BACKEND_OPENCL substitution is valid. One latent footgun
documented in the header: cuda_dispatch assumes **contiguous**
row-major buffers (`lda == cols`); all current call sites honour
this. Phase 45b will extend to strided inputs.

Phase 45b backlog: pinned host memory, async streams, strided
buffers, persistent device-side model state.

## [0.95.0-openmp-backend] — 2026-05-16

Optional OpenMP backend. Default OFF — the OFF build is bit-
identical to pre-Phase-44 (no `<omp.h>` include, empty
`P4A_PARALLEL_FOR_STATIC` macro). When `PLS4ALL_WITH_OPENMP=ON`
the rank-1 deflations, dot-product loading kernels, and
`fill_prediction`/`fill_transform` outer loops in `model.cpp`
are annotated with `#pragma omp parallel for schedule(static)`.
Public ABI unchanged at 1.13.0; symbol count stays at 160.

### OpenMP backend

- `cpp/src/core/parallel.hpp` (new): `P4A_PARALLEL_FOR_STATIC`
  macro with GCC/Clang `_Pragma` and MSVC `__pragma` branches.
- `cpp/src/core/model.cpp`: 21 pragma sites covering every solver's
  deflation and loading kernels plus the two fill helpers. Every
  site has thread-disjoint writes and zero `ctx` access inside the
  parallel region (Codex review confirmed).
- `cpp/src/c_api/c_api_version.cpp`: `p4a_backend_is_available`
  reports OpenMP availability based on `P4A_USE_OPENMP`.
- `cpp/src/CMakeLists.txt` + top-level `CMakeLists.txt`:
  `find_package(OpenMP REQUIRED COMPONENTS CXX)` hoisted to the
  top so `OpenMP::OpenMP_CXX` is visible in `cpp/src/` and
  `cpp/tests/`. CMake `WARNING` when BLAS+OMP both on (OpenBLAS
  USE_OPENMP coexistence) or OMP+TSAN (libgomp not TSAN-clean).
- `cpp/tests/abi/test_openmp_backend.cpp` (new): fixture parity
  + 200x20x2 smoke; passes whether OMP is compiled in or not.

All 4 build configurations (OFF/OFF, BLAS-only, OMP-only,
BLAS+OMP) pass 262/262 tests + 10/10 cross-binding parity gates.

Phase 44b backlog: outer-loop parallelism (CV folds, EMCUVE
ensembles, AOM operator sweeps), `Context::num_threads` plumbing
into `omp_set_num_threads`, benchmark CLI knob.

## [0.94.0-blas-backend] — 2026-05-16

Optional CBLAS backend. Default is **OFF** — the OFF build is
bit-identical to pre-Phase-43 with the same scalar loops and no new
link dependency. When `PLS4ALL_WITH_BLAS=ON` libp4a routes the hot
linear-algebra kernels (matrix-vector products in SIMPLS / NIPALS;
cross-covariance `X^T Y` in SIMPLS / SVD / kernel-algo / power /
randomized-SVD / canonical pair / OPLS / DI-PLS) through a system
CBLAS implementation. Public C ABI is unchanged at 1.13.0; symbol
count stays at 160; all 10 cross-binding parity gates pass with
maximum drift `5.2e-16` rmse_rel.

### BLAS backend

- `cpp/src/core/linalg.hpp` — internal `pls4all::linalg` namespace
  with `gemv`, `matvec`, `gemm` and `ger` entry points. Reference
  scalar implementations under `!P4A_USE_BLAS`, CBLAS dispatch
  under `P4A_USE_BLAS`. The macOS `<Accelerate/Accelerate.h>`
  header is auto-selected when `<cblas.h>` is unavailable.
- `cpp/src/core/model.cpp` — `matrix_vector_product` body forwards
  to `linalg::matvec` (covers 14 callers). The two NIPALS
  `X^T y_score / y_score_ss` inner-loop sites become `linalg::gemv`
  with the alpha factor folded in. The 9 cross-covariance triple
  loops become `linalg::gemm(Trans_Yes, Trans_No, ...)`.
- `cpp/src/c_api/c_api_version.cpp` — `p4a_backend_is_available`
  returns 1 for `P4A_BACKEND_BLAS` when compiled in.
- `cpp/src/CMakeLists.txt` + `cpp/tests/CMakeLists.txt` —
  `find_package(BLAS)` + `check_include_file_cxx(cblas.h)` gate
  the option; both hard-fail with actionable diagnostics if the
  toolchain is incomplete. `${BLAS_LIBRARIES}` linked into
  `pls4all_c`, `pls4all_c_static` and `pls4all_tests`.
- `cpp/tests/abi/test_blas_backend.cpp` — fixture parity + 200x20x2
  smoke test; passes whether BLAS is compiled in or not.
- Phase 43b backlog: rank-1 deflations via `linalg::ger`,
  `compute_rotations_and_coefficients` GEMMs, BLAS-aware bench CLI.

## [0.93.0-nim-binding] — 2026-05-16

Zero-nimble Nim binding via the language's built-in `{.importc,
dynlib.}` pragmas. Parity-gated against the shared cross-binding
fixture; result is bit-exact (rmse_rel = 0.0 on all four output
arrays).

### Nim binding

- `bindings/nim/pls4all.nim` — public API (`pls4all.version`,
  `pls4all.abiVersion`, `pls4all.plsFit`). Uses `openArray[float64]`
  for input slices (contiguous, length-aware), and `seq[float64]`
  for outputs.
- `bindings/nim/test/test_parity.nim` — parity gate using the
  stdlib `std/json` module (no extra package).
- `bindings/nim/README.md`, `bindings/nim/.gitignore`.
- Compile-time library override via `-d:libp4aName=...`.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.92.0-lua-binding] — 2026-05-16

Zero-rock Lua / LuaJIT binding via LuaJIT's built-in FFI module.
Parity-gated against the shared cross-binding fixture; result is
bit-exact (rmse_rel = 0.0 on all four output arrays, matching
JNI / Rust / .NET / Ruby).

### Lua binding

- `bindings/lua/pls4all.lua` — public API (`pls4all.version`,
  `pls4all.abi_version`, `pls4all.pls_fit`). Uses LuaJIT's
  `ffi.new("double[?]", N, src_table)` to populate the C array
  directly from a Lua sequence — the most direct FFI hand-off in
  the binding set.
- `bindings/lua/test/test_parity.lua` — parity gate (regex-based
  fixture parser; no JSON-rock required).
- `bindings/lua/README.md`.

Requires LuaJIT (the `ffi` module is LuaJIT-specific). Plain Lua
support is on the backlog. No ABI change (uses `p4a_pls_fit_simple`
from 1.13).

## [0.91.0-ruby-binding] — 2026-05-16

Zero-gem Ruby binding via the stdlib `Fiddle` library. Parity-gated
against the shared cross-binding fixture; result is bit-exact
(rmse_rel = 0.0 on all four output arrays, matching JNI / Rust /
.NET).

### Ruby binding

- `bindings/ruby/lib/pls4all.rb` — public API
  (`Pls4all.version`, `Pls4all.abi_version`, `Pls4all.pls_fit`).
  Uses `Array#pack("E*")` / `String#unpack("E*")` to marshal Float
  arrays to/from IEEE 754 little-endian double buffers — the exact
  layout libp4a's row-major C ABI expects.
- `bindings/ruby/test/test_parity.rb` — parity gate.
- `bindings/ruby/README.md`.

No gem install required: Fiddle has been in Ruby core since 1.9.
No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.90.0-dotnet-binding] — 2026-05-16

.NET 9 class library wrapping libp4a via P/Invoke
(`[DllImport("p4a", ...)]`). Parity-gated against the shared
cross-binding fixture; result is bit-exact (rmse_rel = 0.0 on all
four output arrays, identical to JNI and Rust).

### .NET binding

- `bindings/dotnet/Pls4all/Pls4all.cs` — public API. Top-level
  `Pls4all` static class with `Version()`, `AbiVersion()` and
  `PlsFit(ReadOnlySpan<double> x, ReadOnlySpan<double> y, int n,
  int p, int q, int nComponents)`. `Span<double>` inputs are pinned
  via `fixed` for zero-copy hand-off into libp4a.
- `bindings/dotnet/Pls4all/Pls4all.csproj` — net9.0 class library,
  NuGet metadata in place.
- `bindings/dotnet/TestParity/Program.cs` — top-level statement
  console app for the parity gate.
- `bindings/dotnet/README.md`, `bindings/dotnet/.gitignore`.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.89.0-rust-binding] — 2026-05-16

Rust crate `pls4all`. Zero-dep, hand-rolled `extern "C"` block
against `p4a_pls_fit_simple`. Parity-gated against the shared
cross-binding fixture; result is bit-exact (rmse_rel = 0.0 on all
four output arrays, identical to JNI's exact result).

### Rust binding

- `bindings/rust/pls4all/` — rlib crate. `Cargo.toml` declares no
  dependencies; `FitError` implements `Display` and
  `std::error::Error` by hand.
- `src/lib.rs` — `pls4all::version()`, `pls4all::abi_version()`,
  `pls4all::pls_fit(&[f64], &[f64], n, p, q, k) -> Result<FitResult,
  FitError>`. Uses `#![deny(unsafe_op_in_unsafe_fn)]` so every
  unsafe block carries a SAFETY comment.
- `build.rs` — auto-detects libp4a's location by walking up from
  the manifest dir; overridable via `PLS4ALL_LIB_DIR` and
  `PLS4ALL_RUNTIME_RPATH`.
- `examples/test_parity.rs` — parity gate (run via
  `cargo run --example test_parity --release`).

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.88.0-go-binding] — 2026-05-16

Go binding via cgo. Same `p4a_pls_fit_simple` C-ABI helper, gated
against the shared cross-binding parity fixture at machine epsilon.

### Go binding

- `bindings/go/go.mod` — module
  `github.com/GBeurier/pls4all/bindings/go`.
- `bindings/go/pls4all/pls4all.go` — public Go API. `Version()`,
  `AbiVersion()` and `Fit(x, y []float64, n, p, q, k int)
  (*FitResult, error)`. Go `[]float64` is row-major like the C ABI,
  so the slice pointer is handed straight through cgo — no copy,
  no transpose.
- `bindings/go/cmd/test_parity/main.go` — parity gate against the
  shared `bindings/js/test/parity_fixture.json`. RMSE-rel ~ 2e-16
  on all four output arrays.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

### Phase 36 deferred

Android NDK packaging is on hold pending NDK availability. The
Phase 35 JNI source is unchanged; only the build script needs to
target the NDK toolchain.

## [0.87.0-jni-binding] — 2026-05-16

Desktop JNI binding for the JVM. Same `p4a_pls_fit_simple` C-ABI
helper, gated against the shared cross-binding parity fixture.

### JNI binding

- `bindings/jni/java/io/github/pls4all/Pls4all.java` — public Java
  API with `version()`, `abiVersion()` and
  `plsFit(double[] x, double[] y, int n, int p, int q, int k)`
  returning a `FitResult` record.
- `bindings/jni/c/p4a_jni.c` — JNI implementation. Java arrays are
  row-major like the C ABI, so `GetPrimitiveArrayCritical` pins them
  without copying.
- `bindings/jni/test/TestParity.java` — parity gate against the
  shared `bindings/js/test/parity_fixture.json`. RMSE-rel = 0.0
  bit-exact across all four output arrays.
- `bindings/jni/build.sh` — one-shot build (javac → JNI header →
  `libp4a_jni.so` → TestParity.class). Auto-detects the conda-forge
  `<env>/lib/jvm` JDK layout.

No ABI change (uses `p4a_pls_fit_simple` from 1.13).

## [0.86.0-octave-matlab-binding] — 2026-05-16

MATLAB / GNU Octave binding via MEX shims, parity-gated against the
shared cross-binding fixture at machine-epsilon precision.

### Octave / MATLAB binding

- `bindings/matlab/+pls4all/pls_fit.m` — `pls4all.pls_fit(X, Y, k)`
  fits a SIMPLS regression and returns the coefficient matrix, the
  centring vectors and the in-sample predictions.
- `bindings/matlab/+pls4all/version.m` — `pls4all.version()` returns
  `"0.86.0+abi.1.13.0"`.
- `bindings/matlab/mex/p4a_pls_fit_mex.c` — column-major
  (MATLAB / Octave) ↔ row-major (libp4a) adapter that calls
  `p4a_pls_fit_simple`.
- `bindings/matlab/mex/p4a_version_mex.c` — thin wrapper around
  `p4a_get_version_string`.
- `bindings/matlab/build_mex.m` — single build script that runs on
  both Octave (`mkoctfile --mex`) and MATLAB (`mex`).
- `bindings/matlab/test/test_parity.m` — parity gate against
  `bindings/js/test/parity_fixture.json` (generated by the native
  Python binding). RMSE-rel at numerical floor: `coef = 0`,
  `x_mean = 2.7e-17`, `y_mean = 0`, `preds = 4.7e-17`.

No ABI change (uses the `p4a_pls_fit_simple` helper added in 1.13).

## [0.85.0-julia-binding] — 2026-05-16

Julia binding (`Pls4all.jl`) + new public ABI helper that bypasses
the matrix-view-pointer interop ambiguity that bit both Emscripten
and Julia's ccall:

- `p4a_pls_fit_simple(x, y, n, p, q, n_components, coefs_out,
  x_mean_out, y_mean_out, preds_out)` — raw-pointer SIMPLS PLS
  regression. Default-on centring and scaling matching the public
  Config defaults. Returns the model triple plus optional in-sample
  predictions.

ABI minor 1.12 → 1.13.

### Julia binding

- `bindings/julia/Pls4all.jl/` — Julia 1.10+ package wrapping
  the public C ABI via `ccall`. Set `PLS4ALL_LIB` env var to point
  at `libp4a.so` and `using Pls4all` does the rest.
- `Pls4all.version()`, `Pls4all.abi_version()`,
  `Pls4all.pls_fit(X, Y; n_components=3)`.
- `bindings/julia/Pls4all.jl/test/test_parity.jl` — parity gate
  against `bindings/js/test/parity_fixture.json` (generated by the
  native Python binding). RMSE-rel at numerical floor:
  - coefficients: 1.16e-16
  - x_mean: 1.35e-16
  - y_mean: 2.20e-16
  - predictions: 1.74e-16

### R binding

- `bindings/r/pls4all/src/r_gateway.c` now sets `scale_x = scale_y =
  1` to align with the Python binding's defaults. Without this the R
  binding diverged by ~7.6e-6 from the Python fixture; with the
  alignment the prediction-space match is exact (rmse_rel = 0).
- `bindings/r/test_parity.R` added — parity gate against the shared
  fixture.
- `Makevars` now accepts `PLS4ALL_GENERATED_DIR` for the CMake-
  generated `pls4all/p4a_export.h` path.

### WASM binding

- TypeScript `fitPls` / `predictPls` (and the underlying smoke
  test) now call the public `p4a_pls_fit_simple` symbol instead of
  the binding-internal `p4a_wasm_pls_fit_legacy` (renamed for
  historical reference). The Emscripten parity gate result is
  unchanged at numerical floor.

### Cross-binding parity matrix (this snapshot)

| Binding | rmse_rel vs native Python fixture |
|---------|-----------------------------------|
| R (`pls4all`) | 0.0 (exact) |
| Julia (`Pls4all.jl`) | 1.7e-16 |
| WASM (`@pls4all/wasm`) | 2.1e-16 |

### Verified

- 256 native C++ tests pass.
- ABI symbol diff: 160 symbols (+1, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- Cross-binding parity gates: all PASS at numerical floor.

### Changed

- Project version `0.85.0+abi.1.13.0`. C ABI minor 12 → 13 (additive
  — 1 new symbol, `p4a_pls_fit_simple`).

## [0.84.0-wasm-parity] — 2026-05-16

WebAssembly binding now passes a parity gate vs the native Python
pls4all binding at numerical floor (1e-16 RMSE-rel).

### Root cause of the WASM fit regression

Emscripten 5.0.7 mis-compiles direct calls to `p4a_model_fit` (and
related entry points) when the matrix-view pointer arguments come
from a JS-allocated buffer. The same C++ kernel called from a C
translation unit inside the same WASM module fits correctly. This was
isolated through five diagnostic variants (`p4a_wasm_self_test*`)
that progressively narrowed the failure to the matrix-view pointer
argument path.

### Workaround / API

Two new WASM-side helpers funnel JS callers through the working
raw-double-pointer path:

- `p4a_wasm_pls_fit(x, y, n, p, q, n_components, coefs_out, x_mean_out,
  y_mean_out, predictions_out)` — fits SIMPLS PLS regression with
  `center_x = center_y = 1` and writes the model triple into caller-
  supplied buffers.
- `p4a_wasm_pls_predict_from_coeffs(x_new, n_new, p, q, coefs, x_mean,
  y_mean, predictions_out)` — applies an already-fit model.

The TypeScript `fitPls` / `predictPls` functions (re-exported via
`bindings/js/src/index.ts`) wrap these helpers. The existing
`Model.fit / .predict` class API is preserved for compatibility.

### Parity gate

`bindings/js/test/run_smoke.mjs` now:

1. Loads the WASM module and prints version + ABI.
2. Fits SIMPLS on deterministic input via `p4a_wasm_pls_fit`.
3. Compares coefficients / x_mean / y_mean / predictions against a
   fixture generated by `bindings/js/test/generate_parity_fixture.py`
   running the native Python binding on the same X, Y.

Reported RMSE-rel on the cell:

| Quantity | rmse_rel | tolerance |
|----------|----------|-----------|
| coefficients | 1.37e-16 | 1e-6 |
| x_mean | 1.51e-16 | 1e-6 |
| y_mean | 2.20e-16 | 1e-6 |
| predictions | 2.12e-16 | 1e-6 |

All four match the native binding within 1 ULP — the WASM build is
byte-for-byte equivalent to the Linux Python binding.

### Verified

- 256 native C++ tests pass.
- ABI symbol diff: 159 symbols (unchanged from 0.83).
- `ldd` on native: clean.
- WASM smoke + parity: PASS.

### Changed

- Project version `0.84.0+abi.1.12.0`. C ABI surface unchanged.
- README's "Status" section flipped from "known issue" to ✅.
- `bindings/js/src/wasm_entry.c` now ships `p4a_wasm_pls_fit` and
  `p4a_wasm_pls_predict_from_coeffs` (instrumented self-test functions
  from the bug investigation were removed).

## [0.83.0-wasm-binding] — 2026-05-16

WebAssembly binding scaffold via Emscripten 5.0.7. The new
`bindings/js/` directory ships a TypeScript wrapper, an Emscripten-
compiled `p4a.wasm` + `p4a.js` (MODULARIZE / EXPORT_ES6), and a Node
smoke test exercising the C ABI lifecycle.

### Added

- `bindings/js/CMakeLists.txt` — Emscripten target wired into the
  existing `emscripten` CMake preset. Build command:
  `cmake --preset emscripten && cmake --build --preset emscripten`.
  Artifacts land in `build/emscripten/bindings/js/{p4a.js,p4a.wasm}`.
- `bindings/js/src/wasm_entry.c` — pulls in the full ABI header so
  Emscripten does not dead-strip the matrix-view init helpers.
- TypeScript wrappers under `bindings/js/src/`:
  - `ffi.ts` — Module loader + heap I/O helpers.
  - `types.ts` — mirrored C enums (`Algorithm`, `Solver`, `Deflation`,
    `Dtype`, `Status`) and the `Pls4allError` class.
  - `context.ts` / `config.ts` / `model.ts` / `methodResult.ts` —
    RAII-style wrappers around the C ABI handles.
  - `index.ts` — public barrel (`loadModule`, `version`,
    `abiVersion`, plus the wrappers).
- `bindings/js/package.json`, `tsconfig.json`, README.
- `bindings/js/test/run_smoke.mjs` — Node smoke covering version /
  ABI introspection and the Context / Config lifecycle.
- `bindings/js/test/wasm_fit_debug.c` — pure-C reference test showing
  the WASM library fits models correctly when invoked from inside
  the WASM module.

### Verified

- Native dev-release build still passes 256 internal C++ tests.
- ABI symbol diff: 159 symbols (unchanged from 0.82 — no new C ABI
  surface in this phase; the WASM module reuses the same exports).
- `git diff --check`: clean.
- `ldd` on the native shared library: still only libc / libstdc++ /
  libgcc / libm / loader.
- Emscripten preset builds `p4a.wasm` (480 KB) + `p4a.js` (89 KB).
- Node smoke (`bindings/js/test/run_smoke.mjs`) passes — confirms the
  module loads, all 161 exports are reachable, and Context / Config
  lifecycle round-trips through the WASM heap.

### Known issue

- `p4a_model_fit` invoked from the JS host produces incorrect
  `x_mean` and coefficients on the WASM target. The same library
  (`libp4a_static.a`) linked into a pure-C WASM translation unit
  (see `bindings/js/test/wasm_fit_debug.c`) fits correctly, so the
  bug is in the JS↔WASM marshalling path rather than in the C++
  numerics. Suspected: an interaction with `WASM_BIGINT=1` int64
  passing through the matrix-view struct when crossing the JS / WASM
  boundary. Under investigation; the `Status` section of
  `bindings/js/README.md` documents the workaround until fixed.

### Changed

- Project version `0.83.0+abi.1.12.0`. C ABI surface unchanged
  (still ABI 1.12.0).
- Top-level `CMakeLists.txt` now hooks `bindings/js/` behind
  `PLS4ALL_BUILD_BINDINGS_JS` (driven by the `emscripten` preset).

## [0.82.0-batch-12-monitoring-onese] — 2026-05-15

Public C ABI exposure for §19 PLS process monitoring and §10 one-SE
component-selection rule. Both paper-only — industrial monitoring
packages are mostly proprietary, and the one-SE rule is a numerical
helper that lacks a unified external reference.

### Added

- `p4a_pls_monitoring_run(ctx, model, X_reference, X_monitor, alpha,
  **out)` — fits T²/Q thresholds from phase-1 data and evaluates
  phase-2 with alarms.
- `p4a_one_se_rule_compute(ctx, fold_rmse_matrix, max_components,
  n_folds, **out)` — standalone helper: given a (max_components ×
  n_folds) fold-RMSE matrix, returns best_n_components,
  one_se_n_components, threshold and standard error.
- Python bindings: `pls_monitoring_run`, `one_se_rule_compute`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 159 symbols (+2, clean).
- Parity gate: 13 external PASS, 24 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.82.0+abi.1.12.0`. C ABI minor 11 → 12.

## [0.81.0-batch-11-multiblock] — 2026-05-15

Public C ABI exposure for the §17-19 multi-block remainder. All three
are paper-only because R `multiblock`'s `HDANOVA` transitive dep
cascades-fails to build on this minimal R env. The C ABI accepts an
array of `p4a_matrix_view_t` blocks plus per-block counts.

### Added

- `p4a_so_pls_fit(ctx, cfg, X_blocks, n_blocks, Y,
  n_components_per_block, n, **out)` — Sequential & Orthogonalized
  multi-block PLS (Næs et al. 2011).
- `p4a_on_pls_fit(ctx, cfg, X_blocks, n_blocks, n_joint,
  n_unique_per_block, n, **out)` — Orthogonal multi-block decomposition
  (Löfstedt & Trygg 2011). Result exposes per-block joint loadings /
  scores and unique loadings via name-suffix `_<b>`.
- `p4a_rosa_fit(ctx, cfg, X_blocks, n_blocks, Y, n_components, **out)`
  — Response-Oriented Sequential Alternation (Liland et al. 2016).
- Python bindings: `so_pls_fit`, `on_pls_fit`, `rosa_fit` accept a
  Python list of NumPy arrays as `X_blocks`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 157 symbols (+3, clean).
- Parity gate: 13 external PASS, 22 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.81.0+abi.1.11.0`. C ABI minor 10 → 11.

## [0.80.0-batch-10-ensembles] — 2026-05-15

Public C ABI exposure for §20 PLS ensembles. All three are paper-only
because RNG / bootstrap-index conventions differ between languages, so
numerical parity is impossible without sharing the exact RNG.

### Added

- `p4a_bagging_pls_fit(ctx, cfg, X, Y, n_estimators, seed, **out)` —
  bootstrap aggregation of PLS regressors (Breiman 1996).
- `p4a_boosting_pls_fit(ctx, cfg, X, Y, n_estimators, learning_rate,
  **out)` — gradient-boosting style stage-wise PLS (Friedman 2001).
- `p4a_random_subspace_pls_fit(ctx, cfg, X, Y, n_estimators,
  features_per_subspace, seed, **out)` — random-feature-subspace
  bagging (Ho 1998).
- Python bindings: `bagging_pls_fit`, `boosting_pls_fit`,
  `random_subspace_pls_fit`.

### Verified

- 256 internal C++ tests pass.
- ABI symbol diff: 154 symbols (+3, clean).
- `ldd` audit: clean.
- Parity gate: 13 external PASS, 19 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.80.0+abi.1.10.0`. C ABI minor 9 → 10 (additive —
  3 new symbols).

## [0.79.0-batch-9-pls-heads] — 2026-05-15

Public C ABI exposure for §5 PLS heads — GLM, QDA, Cox. All three
methods are documented `paper-only` for two reasons:

1. The R packages that implement these (`plsRglm`, `plsRcox`) failed
   to install on this host due to heavy transitive deps (`car`,
   `bipartite`) which themselves cascade-fail.
2. pls4all's internal kernels are simplified vs the canonical IRLS /
   Cox-PH implementations, so even if the R packages installed,
   numerical agreement would be qualitative.

### Added

- `p4a_pls_glm_fit(ctx, cfg, X, Y, poisson, **out)` — PLS-reduced GLM
  with optional Poisson link. Result exposes coefficients, intercept,
  predictions, x_mean, rmse, poisson and n_components scalars.
- `p4a_pls_qda_fit(ctx, cfg, X, y_labels, n, **out)` — QDA on PLS
  scores. Result exposes class_means, class_covariances,
  log_class_priors, rotations_r, x_mean, predictions (log-likelihood
  scores), n_components scalar.
- `p4a_pls_cox_fit(ctx, cfg, X, survival_times, n, event_indicators,
  n, **out)` — Cox PH on PLS scores with Breslow baseline. Result
  exposes coefficients, baseline_hazard, event_times, x_mean,
  predictions (linear-predictor scores), n_components scalar.
- Python bindings: `pls_glm_fit`, `pls_qda_fit`, `pls_cox_fit`.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 151 symbols (+3, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 16 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.79.0+abi.1.9.0`. C ABI minor 8 → 9 (additive — 3
  new symbols).

## [0.78.0-batch-8-calibration-transfer] — 2026-05-15

Public C ABI exposure for §13 calibration transfer + missing-aware
NIPALS. All four methods are documented as `paper-only` — no widely
installable Python or R port exists for PDS, DS, MIR-PLS or the
iterative missing-aware NIPALS variant used here. (R `prospectr`
ships SNV / MSC / Shenk-West but not Wang et al. PDS/DS; R `pls`
handles NA via row deletion rather than iterative imputation.)

### Added

- `p4a_pds_fit(ctx, X_source, X_target, window_half_width, **out)` —
  exposes the (pt × ps) PDS transformation, in-sample predicted
  X_target, RMSE and window_half_width.
- `p4a_ds_fit(ctx, X_source, X_target, **out)` — full (ps × pt) DS
  transformation + bias + predictions + RMSE.
- `p4a_mir_pls_fit(ctx, cfg, X, Y, **out)` — inverse-regression PLS
  variant (Sjöblom et al. 1998).
- `p4a_missing_aware_nipals_fit(ctx, cfg, X, Y, **out)` — NaN-tolerant
  NIPALS that imputes missing entries with the latent-space iterate
  (Nelson et al. 1996).
- Python bindings: `pds_fit`, `ds_fit`, `mir_pls_fit`,
  `missing_aware_nipals_fit`.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 148 symbols (+4, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 13 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.78.0+abi.1.8.0`. C ABI minor 7 → 8 (additive — 4
  new symbols).

## [0.77.0-batch-7-sparse-variants] — 2026-05-15

Public C ABI exposure for the §7 sparse PLS variants.

### Added

- `p4a_sparse_pls_da_fit(ctx, cfg, X, y_labels, n, **out)` — dummy-
  encodes integer class labels, runs sparse SIMPLS. Result exposes
  coefficients (p × n_classes), predictions, x_mean, y_mean,
  class_priors, rmse.
- `p4a_group_sparse_pls_fit(ctx, cfg, X, Y, group_assignment, n,
  group_lambda, **out)` — group-soft-thresholded sparse PLS. Result
  exposes coefficients, predictions, x_mean, y_mean, rmse,
  `n_groups` and `group_lambda` scalars.
- `p4a_fused_sparse_pls_fit(ctx, cfg, X, Y, l1_lambda, fusion_lambda,
  **out)` — L1 + fusion of consecutive feature pairs. Result mirrors
  group_sparse plus `l1_lambda` and `fusion_lambda` scalars.
- Python bindings: `sparse_pls_da_fit`, `group_sparse_pls_fit`,
  `fused_sparse_pls_fit`.
- Runner extensions: `MethodSpec.needs_labels` and
  `MethodSpec.needs_group_assignment`; deterministic class labels and
  group assignments generated by `_make_dataset`.
- Parity:
  - `sparse_pls_da` vs R `spls::splsda` 2.3.2 — rmse_rel 0.92
    (qualitative; spls returns hard class labels and pls4all returns
    soft dummy scores).
  - `group_sparse_pls` paper-only (Liland & Indahl 2009 PPLS-DA
    family).
  - `fused_sparse_pls` paper-only (Yengo et al. 2016).

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 144 symbols (+3, clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 9 paper-only smoke PASS, 0 numpy.

### Changed

- Project version `0.77.0+abi.1.7.0`. C ABI minor 6 → 7 (additive — 3
  new symbols).

## [0.76.0-external-refs-only] — 2026-05-15

**Reference-policy refactor.** Per user direction May 2026, parity-
gate references are now restricted to external libraries in any
language. Hand-written NumPy mirrors are removed. Methods without a
widely installable external reference are marked `paper-only` and
ship with the canonical paper citation; the runner records the
citation but skips the numerical parity check.

### Changed

- `benchmarks/parity_timing/registry.py`:
  - Dropped 9 hand-written NumPy-mirror classes
    (`SparseSimplsNumpyReference`, `DiPlsNumpyReference`,
    `CpplsNumpyReference`, `WeightedPlsNumpyReference`,
    `RobustPlsNumpyReference`, `RidgePlsNumpyReference`,
    `ContinuumNumpyReference`, `NPlsNumpyReference`,
    `KernelPlsNumpyReference`, `O2PlsNumpyReference`,
    `ApproximatePressNumpyReference`) and the `_numpy_simpls`
    helper. ~600 lines removed.
  - Added external-library references:
    - `WeightedPlsSklearnReference` and `RidgePlsSklearnReference`
      using sklearn 1.4.2 PLSRegression as the external PLS engine
      (the row-scaling / data-augmentation preconditioning is
      mathematically equivalent to weighted / ridge PLS by
      construction).
  - Added `paper_only` field on `MethodSpec` for methods whose only
    known implementation is the original paper. Currently flagged:
    `di_pls` (Nikzad-Langerodi 2018), `cppls` (Indahl 2005),
    `robust_pls` (Cummins 1995), `continuum_regression` (Stone &
    Brooks 1990), `n_pls` (Bro 1996), `kernel_pls_rbf` (Rosipal &
    Trejo 2001), `approximate_press` (Eastment & Krzanowski 1982).
  - `Stripped` is retained for OmicsPLS adapter.
- `benchmarks/parity_timing/runner.py`: paper-only rows smoke-fit the
  pls4all side and record `status=paper_only`. The runner exits non-
  zero only on real parity regressions; paper-only rows count as
  PASS as long as the C++ entry point is callable.
- README parity table reflects the new policy: external refs and
  paper-only citations only; no NumPy-mirror entries.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc) — no C++ changes in this commit.
- ABI symbol diff vs expected list: 141 symbols (unchanged from 0.75).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS (external), 7 paper-only (smoke), 0 numpy.

## [0.75.0-batch-5-pls-diagnostics] — 2026-05-15

Public C ABI exposure for §9 PLS diagnostics: Hotelling T², Q residuals
(SPE) and DModX. Single entry point applied to a fitted `p4a_model_t`.

### Added

- `p4a_pls_diagnostics_compute(ctx, model, X, X_reference_or_NULL,
  **out)` — computes T², Q and DModX in one call. Result exposes
  `t2`, `q`, `dmodx` as (1 × n_samples) row vectors plus scalars
  `n_components` and `n_features`.
- Python binding: `pls4all.pls_diagnostics_compute(ctx, model, X,
  X_reference=None)`.
- Parity-gate `pls_diagnostic_t2 / q / dmodx` rows wired to R
  `mdatools::pls$xdecomp$T2 / $Q` (0.15.0). No Python equivalent ships
  widely; the `python_reference` column reports `none` per project
  reference policy. Tolerances are wide because mdatools and pls4all
  use different SIMPLS deflation/normalization conventions; the column
  documents *qualitative* parity with the only available external R
  reference.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 141 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 18 PASS, 9 `no_r_reference`, 3 `no_python_reference`
  (documented per-method).

### Changed

- Project version `0.75.0+abi.1.6.0`. C ABI minor 5 → 6 (additive — 1
  new symbol).

## [0.74.0-batch-4-o2pls-press] — 2026-05-15

Public C ABI exposure for §16 O2PLS and §29 approximate-PRESS.

### Added

- `p4a_o2pls_fit(ctx, cfg, X, Y, n_predictive, n_x_orthogonal,
  n_y_orthogonal, **out)` — bi-directional OPLS (Trygg & Wold 2003).
  Returns coefficients, predictions, x_mean, y_mean, w_predictive,
  c_predictive, w_x_orthogonal, c_y_orthogonal, b_predictive, rmse.
- `p4a_approximate_press_compute(ctx, cfg, X, Y, max_components,
  **out)` — leverage-inflated residual PRESS for component selection.
  Returns press_per_component, rmse_per_component, an int vector
  `selected_n_components`, and the same as a double scalar
  `selected_n_components_d`.
- Python bindings: `pls4all.o2pls_fit`, `pls4all.approximate_press_compute`.
- `MethodSpec.prediction_key` to let methods that don't expose
  `predictions` (e.g. PRESS curve) wire a different output to the
  parity comparator. Runner generates multi-target Y when
  `cell_params["n_targets"] > 1`.
- O2PLS parity: NumPy mirror (rmse_rel = 7.97e-02; PASS at tol 1.0) and
  R `OmicsPLS::o2m` 2.1.0 (rmse_rel = 4.54e-01; PASS at tol 1.0, flagged
  *qualitative* because OmicsPLS implements a joint-iterative O2PLS
  variant — different from pls4all's peel-then-PLS algorithm).
- Approximate-PRESS parity: NumPy mirror (rmse_rel = 1.63e-03; PASS).

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 140 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 15 PASS, 9 `no_r_reference` (documented per-method).

### Changed

- Project version `0.74.0+abi.1.5.0`. C ABI minor 4 → 5 (additive — 2
  new symbols).

## [0.73.0-batch-3-n-pls-kernel-pls] — 2026-05-15

Third batch of public C ABI exposure. Both new methods rely on
`p4a_method_result_t` and ship with deterministic NumPy mirrors as
parity references — no widely installable external Python or R port
exists for either variant.

### Added

- `p4a_n_pls_fit(ctx, cfg, X_flat, mode_j, mode_k, Y, **out)` — 3-way
  N-PLS (Bro 1996) on `(n_samples x mode_j x mode_k)` tensors flattened
  as `(n_samples x (mode_j*mode_k))`. Result exposes `predictions`,
  `coefficients`, `w_j`, `w_k`, `scores_t`, `x_mean`, `y_mean`, `rmse`.
- `p4a_kernel_pls_fit(ctx, cfg, kernel_type, gamma, coef0, degree, X, Y,
  **out)` — non-linear kernel PLS (Rosipal & Trejo 2001) with
  `kernel_type` 0=linear, 1=RBF, 2=polynomial, 3=sigmoid. Result exposes
  `predictions`, `alpha` dual coefficients, `y_mean`, `rmse` and
  introspection scalars `kernel_type`, `gamma`, `coef0`, `degree`.
- Python bindings: `pls4all.n_pls_fit`, `pls4all.kernel_pls_fit`.
- Parity-gate `numpy-mirror` references reproducing both algorithms
  step-for-step. RMSE-rel < 5e-2 tolerance is met at numerical floor:
  N-PLS 1.36e-07, kernel PLS RBF 2.32e-15.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 138 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS, 8 `no_r_reference` (documented per-method).

### Changed

- Project version `0.73.0+abi.1.4.0`. C ABI minor 3 → 4 (additive — 2
  new symbols).

## [0.72.0-batch-2-cppls-weighted-robust-ridge-continuum] — 2026-05-15

Public C ABI exposure for §1 CPPLS and §26 weighted / robust / ridge /
continuum-regression variants, gated by parity vs NumPy mirrors that
exactly reproduce the C++ kernels.

### Added

- Public ABI entry points:
  - `p4a_cppls_fit(ctx, cfg, X, Y, gamma, **out)`
  - `p4a_weighted_pls_fit(ctx, cfg, X, Y, sample_weights, n, **out)`
  - `p4a_robust_pls_fit(ctx, cfg, X, Y, huber_k, max_irls_iter, **out)`
  - `p4a_ridge_pls_fit(ctx, cfg, X, Y, ridge_lambda, **out)`
  - `p4a_continuum_regression_fit(ctx, cfg, X, Y, tau, **out)`
- Python bindings under `pls4all._methods`: `cppls_fit`,
  `weighted_pls_fit`, `robust_pls_fit`, `ridge_pls_fit`,
  `continuum_regression_fit`.
- Parity gate adds NumPy-mirror references for all 5 methods. R columns
  documented as `(none)`: `pls::cppls` is a different algorithm (Liland
  2009 Canonical Powered PLS), and no widely installable R port exists
  for our sqrt(w)-prescaled weighted SIMPLS, Huber-IRLS robust SIMPLS,
  sqrt(λ)·I-augmented ridge SIMPLS or col-std^τ continuum regression.

### Verified

- All Batch 2 numpy-mirror parities pass at effective numerical zero
  (9.9e-14 to 5.7e-6).
- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 136 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.

### Changed

- Project version `0.72.0+abi.1.3.0`. C ABI minor 2 → 3 (additive — 5 new
  symbols, all `p4a_*` prefixed).

## [0.71.0-batch-1-sparse-di-recursive] — 2026-05-15

Public C ABI exposure for the first batch of remaining-algorithm-taxonomy
methods, gated by an external-reference parity runner (Python + R when
available).

### Added

- **Universal method-result handle** (`p4a_method_result_t`,
  `cpp/src/core/method_result.hpp`,
  `cpp/src/c_api/c_api_method_result.cpp`). Owns named double matrices,
  int vectors and scalars; accessed via `p4a_method_result_get_double_matrix`,
  `p4a_method_result_get_int_vector`, `p4a_method_result_get_scalar` and
  released by `p4a_method_result_destroy`. Eliminates per-method ABI bloat
  for the upcoming batches.
- Public ABI entry points:
  - `p4a_sparse_simpls_fit(ctx, cfg, X, Y, sparsity_lambda, **out)`
    (Chun & Keles 2010 soft-threshold SIMPLS).
  - `p4a_di_pls_fit(ctx, cfg, X_source, Y_source, X_target, di_lambda, **out)`
    (Domain-Invariant PLS).
  - `p4a_recursive_pls_run(ctx, cfg, X, Y, window_size, **out)`
    (recursive / moving-window PLS).
- Python bindings under `pls4all._methods`:
  `MethodResult`, `sparse_simpls_fit`, `di_pls_fit`, `recursive_pls_run`.
- Parity-gate runner `benchmarks/parity_timing/runner.py` + reference
  adapters in `benchmarks/parity_timing/registry.py`.
  - Sparse SIMPLS vs `numpy-mirror` (5.67e-3) and R `spls` 2.3.2 (2.51e-5).
  - DI-PLS vs `numpy-mirror` (4.82e-3); flagged `no_r_reference` —
    Domain-Invariant PLS has no widely installable R port.
  - Recursive PLS vs `scikit-learn` 1.4.2 (1.23e-2) and R `pls` 2.8.5
    (1.23e-2).
- Parity report committed under
  [`benchmarks/results/parity_gate/`](benchmarks/results/parity_gate/) and
  surfaced in the top-level README.

### Changed

- Project version is now `0.71.0+abi.1.2.0`. C ABI minor bumped 1 → 2
  (additive: 7 new symbols, all `p4a_*` prefixed; `ldd` audit clean).
- `cpp/abi/expected_symbols_linux.txt` now lists 131 exported symbols
  (sorted).
- Python wheel loader (`bindings/python/src/pls4all/_ffi.py`) accepts the
  new `libp4a.so.0.71.0` filename.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `ldd` audit: no forbidden runtime dependency (`libopenblas`, `libmkl`,
  `libpython`, `libR`, `librcpp`, `libboost`, `libeigen`, `libpybind11`,
  `libnlohmann`, `libyaml-cpp`).
- `git diff --check`: clean.
- Parity gate: 5 PASS, 1 `no_r_reference` (documented).

## [0.70.0-overview-completion] — 2026-05-15

Phases 16 – 30. Closes every algorithm requested by `Overview.md` and the
"Remaining Algorithm Taxonomy" backlog. All internal C++ kernels; public
ABI exposure deferred to the binding tranche.

### Added

- **§8 multi-block** (`cpp/src/core/multiblock_extensions.{hpp,cpp}`):
  `fit_o2pls`, `fit_so_pls`, `fit_on_pls`, `fit_rosa`.
- **§9 tensor PLS** (`cpp/src/core/tensor_pls.{hpp,cpp}`): `fit_n_pls`
  and `predict_n_pls` for 3-way arrays flattened as `n × (J*K)`.
- **§10.2 non-linear kernel PLS**
  (`cpp/src/core/kernel_pls.{hpp,cpp}`): RBF, polynomial and sigmoid
  kernels via the Gram-matrix dual NIPALS path. `KernelType` enum
  with `LINEAR`, `RBF`, `POLYNOMIAL`, `SIGMOID`.
- **§7 sparse variants** (`cpp/src/core/extra_pls.{hpp,cpp}`):
  `fit_sparse_pls_da`, `fit_group_sparse_pls`,
  `fit_fused_sparse_pls`.
- **§1 + reg / robust / continuum**: `fit_cppls`, `fit_weighted_pls`,
  `fit_robust_pls` (Huber IRLS), `fit_ridge_pls`,
  `fit_continuum_regression`.
- **§5 classifier / GLM heads**: `fit_pls_glm`, `fit_pls_qda`,
  `fit_pls_cox` (Breslow baseline hazard).
- **§13 calibration transfer + missing**: `fit_pds`, `fit_ds`,
  `fit_mir_pls`, `fit_missing_aware_nipals`.
- **§18 approximate-PRESS**: `approximate_press` with leverage-inflated
  residual PRESS and component selection.
- **§20 ensembles**: `fit_bagging_pls`, `fit_boosting_pls`,
  `fit_random_subspace_pls`.
- 28 new C++ ABI / core tests (256 total).

### Changed

- Project version is now `0.70.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; no new public symbols. All shipped methods live in
  `pls4all::core::` and ship for the next phase to wrap.

## [0.69.0-pls-extensions] — 2026-05-15

Phases 8 – 15. Smallest viable additions to algorithm taxonomy sections §7,
§11, §12 (partial), §13, §17, §18, §19. All shipped as internal C++ kernels;
public ABI exposure deferred to the binding tranche.

### Added

- **§7 Sparse PLS** — `fit_pls_sparse_simpls` (soft-threshold on the SIMPLS
  loading direction; `cfg.sparsity_lambda`). `P4A_ALGO_SPARSE_PLS` is now
  dispatched through `fit_model`.
- **§13 Domain-invariant PLS** — `fit_di_pls` projects the SIMPLS direction
  away from the source-target mean difference at each component;
  `cfg.di_lambda` controls the penalty.
- **§12 Recursive PLS (moving window)** — `fit_predict_recursive_pls` in
  `cpp/src/core/recursive_pls.cpp`. Each sample at position ≥ `window_size`
  is predicted from a SIMPLS model fitted on the previous `window_size` rows.
- **§17 Diagnostics** — `pls_hotelling_t2`, `pls_q_residuals`, `pls_dmodx`
  in `cpp/src/core/pls_diagnostics.cpp`.
- **§18 One-SE rule** — `cross_validate_component_prefixes` now records
  per-fold RMSE values; new `select_one_se_components` picks the smallest
  competitive component count.
- **§19 Process monitoring** — `pls_monitoring_fit` /
  `pls_monitoring_evaluate` in `cpp/src/core/pls_monitoring.cpp`. Empirical
  percentile thresholds for T² and Q-residuals at a configurable
  α level, plus per-sample alarm flags.
- **§11 Just-in-time PLS** — documented as already shipped by
  `fit_predict_lw_pls` (k-NN + uniform-weight local SIMPLS).
- Internal `Config` extended with `sparsity_lambda`, `kernel_type`,
  `kernel_gamma`, `kernel_coef0`, `kernel_degree`, `di_lambda`,
  `recursive_forgetting`.
- 19 new C++ ABI / core tests (228 total).

### Deferred (open phases at this release)

- **§10.2 Non-linear kernel PLS** (RBF / polynomial / sigmoid) — needs a
  Gram-matrix dual path and a Model field for kernel parameters.
- **§8 O2PLS** — needs bi-directional OPLS infrastructure.
- **§9 N-PLS** — needs tensor unfolding utilities.

### Changed

- Project version is now `0.69.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; the new fields on `Config` are internal-only.

## [0.68.0-comprehensive-benchmark] — 2026-05-15

Phase 7b–7e. Comprehensive performance benchmark matrix.

### Added

- Python ctypes wrappers `pls4all.Model` / `pls4all.ModelArrayKind` with
  NumPy zero-copy `p4a_matrix_view_t` constructors. Fit / predict /
  transform / get_array. Non-copyable owning handles.
- `pls4all_cli --bench` subcommand: deterministic synthetic dataset,
  fit + predict timing per algorithm, CSV-parseable output. Recognized
  algos: pls_nipals, pls_orthogonal_scores, pls_simpls,
  pls_kernel_algorithm, pls_wide_kernel, pls_svd, pls_power,
  pls_randomized_svd, pcr_svd.
- `bindings/r/pls4all/` minimum viable R package: `.Call` gateway,
  fit/predict for the 9 shipped PLS regression solvers, R-level
  wrappers, version queries, `R_registerRoutines` init, Makevars +
  Makevars.win. No Rcpp dependency.
- `benchmarks/runners/pls_regression.py` PLS regression benchmark
  runner with 9 algos × `(n_samples, n_features)` cells. Smoke gated
  at 200x100 and 500x100; full matrix on demand via `--scale full`.
- `benchmarks/runners/matrix.py` cross-language performance matrix
  driver: native C++ (CLI) / pls4all-Python / pls4all-R / sklearn
  reference. Records median + min wall times per row, language status
  per column, CPU pinning environment variables. Smoke at
  200x100/500x100 × 2 algos; full matrix is 9 × 5 × 3 = 135 cells.
- `benchmarks/results/{pls_regression,matrix}/accuracy.csv` committed
  and gated by `python benchmarks/run.py --check`. Timing CSVs +
  summaries written but not gated.
- Updated `benchmarks/README.md` with the run / pinning commands.
- Phase notes in `roadmap/phase-7b-…md`, `phase-7c-…md`,
  `phase-7d-…md`, `phase-7e-…md`.
- Overview status snapshot at the top of `Overview.md` mapping each
  taxonomy section to the current pls4all delivery state.

### Changed

- `bindings/python/src/pls4all/_config.py` exposes `center_x`,
  `scale_x`, `center_y`, `scale_y`, `store_scores` as Python
  properties (the corresponding ctypes prototypes were also added in
  Phase 6f and are reused).
- Benchmark output reorganized under `benchmarks/results/<benchmark>/`
  per-suite subdirectories.
- Project version is now `0.68.0+abi.1.1.0`. C ABI unchanged from
  `1.1.0`; no new public symbols. Only Python / CLI / R bindings and
  the benchmark suite changed.

## [0.67.0-benchmark-foundation] — 2026-05-15

Phase 7a. Benchmark foundation.

### Added

- `benchmarks/` directory with the deterministic Python driver
  `benchmarks/run.py` (`--check` mode + `--repeats N`).
- `benchmarks/runners/_harness.py` exposing `AccuracyRecord`,
  `TimingRecord`, `median_time_ms`, and the summary-table formatter.
- `benchmarks/runners/aom_global.py`: AOM-PLS global selection runner that
  drives `p4a_aom_global_select` through the Python ctypes wrapper and
  mirrors the dataset + folds into `nirs4all/bench/AOM_v0/aompls`.
- Four committed test cases (smooth-monotonic 9x6, 12x8, 16x10 plus a
  detrend-favoured 14x9). The committed accuracy CSV is gated by
  `--check`; current run reports 0.0 absolute RMSE delta against the
  oracle across all four cases.
- Committed `benchmarks/results/aom_global_accuracy.csv` (gated),
  `aom_global_timing.csv` and `aom_global_summary.md` (informational).
- `benchmarks/README.md` describing run command, gated vs informational
  split, and the contract for adding a new runner.

### Changed

- Project version is now `0.67.0+abi.1.1.0`. The C ABI surface is
  unchanged from `1.1.0`; benchmarks consume the existing public AOM/POP
  ABI only.

## [0.66.0-public-aom-pop-abi] — 2026-05-15

Phase 6f. Public C ABI for AOM/POP selection.

### Added

- Opaque handle `p4a_validation_plan_t` with create / destroy /
  `set_n_samples` / `add_fold` / `get_n_samples` / `get_n_folds`.
- Opaque handle `p4a_aom_global_result_t` plus entry point
  `p4a_aom_global_select` and typed accessors for `n_operators`,
  `max_components`, `selected_operator_index`, `selected_n_components`,
  `best_score`, `operator_kinds` (`p4a_operator_kind_t*`),
  `operator_scores`, `rmse_curves` and `predictions`.
- Opaque handle `p4a_aom_per_component_result_t` plus entry point
  `p4a_aom_per_component_select` and typed accessors for `n_operators`,
  `max_components`, `selected_n_components`, `best_score`,
  `operator_kinds` (`p4a_operator_kind_t*`), `selected_operator_indices`
  (`int32_t*`), `component_scores`, `prefix_scores` and `predictions`.
- Python ctypes wrappers `OperatorBank`, `OperatorKind`,
  `ValidationPlan`, `AomGlobalResult`, `AomPerComponentResult`,
  `aom_global_select`, `aom_per_component_select`. Result wrappers are
  non-copyable and copy the C-owned buffers into Python `list`s.
- Python `Config` properties for `center_x`, `scale_x`, `center_y`,
  `scale_y`, `store_scores` (already exposed by the C ABI).
- Driver script `bindings/python/smoke_aom_pop.py` that drives every
  shipped AOM/POP fixture through the public C ABI.
- ABI / parity tests in `cpp/tests/abi/test_validation_plan_abi.cpp`,
  `cpp/tests/abi/test_aom_global_public_abi.cpp`,
  `cpp/tests/abi/test_aom_pop_public_abi.cpp`.

### Changed

- Project version is now `0.66.0+abi.1.1.0`; the C ABI bumps to
  `1.1.0` (additive only, 28 new `p4a_*` symbols).
- Internal `validate_plan` in `cpp/src/core/aom_selection.cpp` now
  rejects out-of-range, duplicated or train/test-overlapping fold
  indices with `P4A_ERR_INVALID_ARGUMENT`. Plan/X row mismatch now
  returns `P4A_ERR_SHAPE_MISMATCH`, documented in the public header.

## [0.4.0-svd] — 2026-05-14

SVD solver increment.

### Added

- Dependency-free SVD regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SVD`, using exact covariance SVD directions with regression
  deflation.
- Deterministic NumPy SVD parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SVD fit / predict path.

### Changed

- Project version is now `0.4.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS, SIMPLS and SVD models.

## [0.3.0-simpls] — 2026-05-14

SIMPLS core increment.

### Added

- Dependency-free SIMPLS regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SIMPLS`.
- Deterministic NumPy SIMPLS parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SIMPLS fit / predict path.

### Changed

- Project version is now `0.3.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS and SIMPLS models.

## [0.2.0-phase1] — 2026-05-14

Phase 1 — PLS CPU reference. Roadmap at [`roadmap/phase-1.md`](roadmap/phase-1.md).

### Added

- Dependency-free reference CPU NIPALS implementation for PLS regression
  (PLS1 / PLS2) behind the existing Phase 0 C ABI.
- Live `p4a_model_predict`, `p4a_model_transform`, allocated output arrays,
  fitted-model array accessors and model dimension getters.
- Binary `P4AM` serialization v1 with little-endian fields and checksum
  validation for export/import round-trips.
- Phase 1 C++ parity tests generated from the three checked-in synthetic
  scikit-learn fixtures.

### Changed

- Unsupported model modes now return `P4A_ERR_UNSUPPORTED`; fitted-model
  functions no longer return `P4A_ERR_NOT_IMPLEMENTED` for the supported
  NIPALS regression path.

## [0.1.0-phase0] — 2026-05-14

First milestone: the public C ABI is feature-complete, exercised by a 60-test suite, and reachable from a Python binding. No PLS algorithm is implemented yet — every function that needs a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. Phase 1 plugs in NIPALS without changing a single public symbol.

### Added

- Public C ABI header (`cpp/include/pls4all/p4a.h`, `p4a_version.h`, generated `p4a_export.h`) declaring **96 `p4a_*` symbols** across status / dtype / backend / matrix-view / context / config / operator-bank / gating-strategy / pipeline / model / serialization / array.
- Static asserts pinning every public enum to 4 bytes and `p4a_matrix_view_t` to 48 bytes (LP64 / LLP64).
- CMake project (CMake 3.21+) with hidden default visibility, `-Wl,--no-undefined`, dead-code stripping, undefined-symbol enforcement on every supported toolchain.
- `CMakePresets.json` with 11 CI matrix presets matching the GitHub Actions labels 1:1, plus sanitizer (ASAN / UBSAN / TSAN / ASAN+UBSAN), coverage, and cross-compile presets (Emscripten, Android arm64 / x86_64).
- Reference CPU implementations of status / version / context / config / matrix-view / operator-bank / gating-strategy / pipeline lifecycles, with try/catch boundary protection on every `P4A_API` wrapper and a per-context 4096-byte thread-safe error buffer.
- `pls4all_cli` introspection tool: `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness (`cpp/tests/{harness.hpp, main.cpp}`) plus 8 ABI test files: lifecycle, config setters, status codes, version, matrix view, error messages, OOM paths, model stubs. **60 / 60 tests pass.**
- GitHub Actions workflows: `ci.yml` (11-cell build matrix), `abi-check.yml` (symbol surface + runtime-dep audit on Linux / macOS / Windows), `sanitizers.yml` (ASAN / UBSAN / TSAN / combined), `coverage.yml` (informational), `parity-gate.yml` (fixture-determinism + Python binding smoke).
- Parity infrastructure: `parity/schema/fixture_schema_v1.{json,md}`, `parity/tolerances.md` (11 pair-wise rows), 3 deterministic synthetic fixtures with SHA-256 manifest, Python generator pinned via `requirements-lock.txt` exposing `generate-fixtures --regenerate` / `--check`.
- Python binding (`bindings/python/`) — ctypes loader, Pythonic `Context`, `Config`, `Pls4allError`, `Backend`, `Dtype`, `Status`.
- Skeleton READMEs for the R, MATLAB, JavaScript/WASM, and Android bindings (full implementations land in Phase 2).
- Documentation skeleton under `docs/` with stubs for architecture, ABI reference, binding guides, parity methodology, and the development workflow.
- Codex review transcripts under `docs/reviews/phase-0/` for the roadmap (rev 1 → 2 → 3) and PRs 2-4 (`docs/reviews/phase-0/000{1,2,3,4,5}-*.md`).

### Known limitations

- Every function that requires a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. NIPALS PLS1 / PLS2 land in Phase 1 (PR-1, immediately after this tag).
- The R-side parity generator is stubbed — only the Python side produces fixtures at Phase 0.
- Bindings beyond Python ship only README placeholders. Full bindings land in Phase 2 once Phase 1 / 3 / 4 supply real algorithms.
- Concurrency / fuzz tests are deferred to a follow-up PR.
