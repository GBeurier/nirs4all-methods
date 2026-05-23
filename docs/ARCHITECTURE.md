# Architecture

This document distils the rationale behind the design choices laid out in [`Direction_Technique.md`](Direction_Technique.md). Read the full spec for the canonical detail; this file is what a new contributor reads first.

## One sentence

> **pls4all is a portable PLS / NIRS engine in C++17, exposed as a small stable C ABI, with thin bindings that just translate native language objects to that ABI.**

## The four load-bearing decisions

### 1. C++ internal, stable C ABI public

C++ ABIs are fragile and toolchain-coupled. The GCC `libstdc++` "dual ABI" history alone (since GCC 5.1, for `std::string` and `std::list`) is enough reason not to expose any C++ type across a shared-library boundary. A C ABI is decades-stable, supported natively by every language we care about (Python ctypes, R `.Call`, MATLAB MEX, Emscripten `cwrap`, Android JNI, Julia `ccall`, .NET P/Invoke, Rust FFI, Go cgo), and lets us ship one set of binaries for many ecosystems.

Concretely: the public header `cpp/include/pls4all/p4a.h` contains only `extern "C"` declarations, opaque handle typedefs, plain enums, plain structs and primitive types. No `std::vector`, no `std::string`, no `Eigen::MatrixXd`, no `std::unique_ptr`.

### 2. Zero mandatory external dependencies

A library that wants to ship to PyPI, CRAN, npm, MATLAB Toolbox Exchange, Maven Central and conda-forge cannot afford to depend on Eigen, Boost, OpenBLAS, MKL, pybind11, Rcpp, embind or nanobind. Each one multiplies packaging pain.

The required toolchain for `pls4all` is: a C++17 compiler, CMake ≥ 3.21, and the standard C/C++ library. Everything else (BLAS, OpenMP, CUDA, Emscripten, Android NDK, MATLAB MEX, doctest, vendored JSON parser) is either optional or test-only.

The reference CPU backend is implemented from scratch in `cpp/src/core/linalg/` (dot, axpy, scal, norm2, gemv, gemm-small, QR, power iteration). Optional accelerated backends (BLAS, CUDA) are alternative implementations of the same internal interface, selectable at runtime via `p4a_context_set_backend`.

### 3. CMake as the only build system

CMake covers C++, static and shared libraries, Android NDK, Emscripten and MATLAB MEX through one configuration model and supports proper target export/import for downstream `find_package(pls4all)`. Trying to maintain Make + Autotools + Bazel + Meson in parallel for a research library would be a permanent distraction. CMake is what we have.

### 4. Bindings are translators, not reimplementations

Every binding's job is exactly:

1. Accept native objects (NumPy array, R matrix, MATLAB `mxArray`, `TypedArray`, Kotlin `FloatArray`).
2. Construct a `p4a_matrix_view_t` (stride-aware — no layout-imposed copies).
3. Call `p4a_*` functions.
4. Translate the result back to the native object.

A binding never owns numerical logic. This is what guarantees that a model fit in Python predicts identically in MATLAB.

## Build-time targets

| CMake target           | Type    | Linked into                                | Purpose                                                    |
| ---------------------- | ------- | ------------------------------------------ | ---------------------------------------------------------- |
| `pls4all_core`         | STATIC  | only `pls4all_c` and tests                 | Internal C++17 implementation. Never installed.            |
| `pls4all_c`            | SHARED  | binding glue, CLI, examples                | Public C ABI shared library `libp4a.{so,dll,dylib}`.       |
| `pls4all_c_static`     | STATIC  | Android predict-only AAR, embedders        | Same surface as `pls4all_c`, statically linked.            |
| `pls4all_tests`        | EXE     | —                                          | doctest binary, links `pls4all_c` + `pls4all_core`.        |
| `pls4all_cli`          | EXE     | —                                          | `--version`, `--abi-info`, `--selfcheck`.                   |

Visibility is hidden by default (`-fvisibility=hidden`, `__declspec(dllimport/export)` on Windows). Only the macros `P4A_API`-decorated declarations escape, and on MSVC we additionally drive a `.def` export file.

## Memory model

Three load-bearing rules:

1. **Never free across the language boundary.** The wrapper never frees a buffer the core allocated; the core never frees a buffer the wrapper allocated. Concrete consequence: we do **not** have a `p4a_free_string` (a free-across-runtime defect waiting to happen — see the `aompls` reference). Error messages live in a context-owned thread-safe buffer; the pointer returned by `p4a_context_last_error` is invalidated by the next call on the same context.
2. **Two output APIs.** For high-performance paths, the caller pre-allocates and passes a buffer (`p4a_predict(model, X, out_buf, out_size)`). For convenience paths, the core allocates and the core frees via `p4a_array_free(arr)`. Both, for different use cases. Never mixed.
3. **Stride-aware everything.** `p4a_matrix_view_t` has explicit `row_stride` and `col_stride` (in element counts). This accepts NumPy row-major, R / MATLAB / Fortran column-major, transposed views, slice views, sub-matrix views — without copying. Core canonicalises internally only when an algorithm needs it.

## Error model

- Every fallible `p4a_*` function returns `p4a_status_t`. `P4A_OK == 0`. Negative-look codes do not exist; the error space is a flat enum.
- Functions that take a context write a human-readable message into the context's 4 KiB thread-safe error buffer on failure. `p4a_context_last_error(ctx)` returns the latest message; it is invalidated by the next call on the same context.
- No exception ever crosses the C ABI. Internal C++ may throw; the `extern "C"` wrappers in `cpp/src/c_api/` catch and translate.

## Versioning

- **ABI version** (`P4A_ABI_VERSION_{MAJOR,MINOR,PATCH}`) is independent of the **project version** (`P4A_PROJECT_VERSION_*`). The ABI bumps less often than the project. A major ABI bump is a hard break — old binaries cannot load new clients.
- `p4a_check_abi_compatibility(header_major, header_minor)` lets bindings detect mismatched headers vs runtime library.
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` are checked-in golden lists. CI diffs them on every PR; any drift fails the build until the snapshots are regenerated *and* `docs/abi/changes_log.md` is updated.

## Parity model

The library is only as good as its agreement with the established references. We pin specific reference implementations and document, per pair of implementations and per algorithm, the achievable absolute and relative tolerances.

- **Python references**: scikit-learn `PLSRegression` / `PLSCanonical` / `PLSSVD` / `CCA`, and the 21 PLS variants under `nirs4all.operators.models.sklearn.*`.
- **R references**: `pls` (PLSR, PCR, CPPLS, NIPALS-OS, SIMPLS, kernel PLS, wide-kernel PLS), `ropls` (PLS / OPLS / PLS-DA / OPLS-DA with diagnostics and permutation tests), `mixOmics` (sPLS, MB-sPLS, DIABLO-like), `plsVarSel` (BVE, GA, IPW, MCUVE, REP, SPA, ST, T2, WVC, …).
- **Fixture format**: see [`parity/schema/fixture_schema_v1.json`](parity/schema/fixture_schema_v1.json). Generators in `parity/python_generator/` and `parity/r_generator/` produce deterministic seeded synthetic data plus real datasets (BEER / CORN / ALPINE).
- **Tolerances**: [`parity/tolerances.md`](parity/tolerances.md). No two reference implementations agree at machine precision; the tolerance table is pair-specific because of legitimate algorithmic drift (e.g. `pls::plsr(method='kernelpls')` vs `sklearn.PLSRegression` NIPALS).

## Phased rollout

See [`ROADMAP.md`](ROADMAP.md). Phases 0 → 1 → 3 → 4 → 2 → 5 → 6 → 7. The "depth-first core" ordering reflects the priority decision captured 2026-05-14: we'd rather build a small correct multi-algorithm core before exposing it through five bindings. Binding skeletons ship at Phase 0 only so the FFI plumbing is already CI-green when Phase 2 lights it up for real.

## Review model

Every roadmap and every implementation PR is reviewed by Codex via `codex exec` (or via the `codex mcp-server` MCP integration, once registered). Codex transcripts are checked into `docs/reviews/phase-N/` for audit. On disagreement, the contributor adds context once; if Codex still disagrees, **Codex wins** and the contributor revises.
