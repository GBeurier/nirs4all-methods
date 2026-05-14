# Phase 1 — PLS CPU Reference

> Status: implementation roadmap · 2026-05-14
>
> Baseline tag: `phase-0`.
> Canonical spec: [`Direction_Technique.md`](../Direction_Technique.md).
> Parity fixtures: [`parity/fixtures/`](../parity/fixtures/).

## 1. Objective

Replace the Phase 0 fitted-model stubs with a portable, dependency-free CPU
reference implementation of NIPALS PLS regression. Phase 1 must not add, remove
or rename any public `p4a_*` symbol.

Acceptance criteria:

1. `p4a_model_fit` supports `P4A_ALGO_PLS_REGRESSION` +
   `P4A_SOLVER_NIPALS` + `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
2. `p4a_model_predict`, `p4a_model_transform`,
   `p4a_model_predict_alloc`, `p4a_model_transform_alloc` and
   `p4a_model_get_array` return real values for fitted models.
3. The Phase 0 synthetic fixtures match the checked-in scikit-learn references
   within the Phase 1 gate tolerance (`<= 1e-9` absolute / relative for
   well-conditioned fields).
4. Invalid shapes, unsupported algorithms/solvers/deflation modes, integer
   dtypes, NaN/Inf data and under-sized output buffers return deterministic
   status codes with context error messages.
5. `p4a_model_export_size`, `p4a_model_export_to_buffer`,
   `p4a_model_import_from_buffer` and `p4a_serialization_inspect` implement
   the v1 little-endian `P4AM` model format.
6. `p4a_array_*` helpers expose core-allocated `f64` arrays and remain
   NULL-safe.
7. The ABI symbol snapshot is unchanged except for implementation behavior.
8. `cmake --preset dev-release && cmake --build --preset dev-release &&
   ctest --preset dev-release --output-on-failure` exits 0, and
   `pls4all_cli --selfcheck` exits 0.

## 2. Supported Numerical Surface

Phase 1 implements the same NIPALS regression path used by
`sklearn.cross_decomposition.PLSRegression`:

- column centering and sample-standard-deviation scaling (`ddof=1`),
- two-block power iteration for the first singular vector of `X.T @ Y`,
- regression deflation of `X` and `Y` by the `X` score,
- sign normalization by the largest absolute `X` weight,
- rotations `R = W @ inv(P.T @ W)`,
- coefficients in public ABI shape `(n_features, n_targets)`,
- intercept equal to the fitted `y_mean`, with prediction performed as
  `(X - x_mean) @ coefficients + intercept`.

Phase 1 intentionally does not implement:

- PLS canonical, PLS-SVD, SIMPLS, OPLS, PLS-DA or PCR,
- preprocessing pipelines,
- operator banks or gating strategies,
- BLAS/OpenMP/CUDA backends,
- language-level model classes in Python/R/MATLAB/JS/Android.

Unsupported requested modes return `P4A_ERR_UNSUPPORTED`, not
`P4A_ERR_NOT_IMPLEMENTED`, because the model subsystem itself is now live.

## 3. Internal Layout

New internal code lives under `cpp/src/core/`:

- `model.hpp/.cpp`: fitted model state, NIPALS fit, prediction,
  transformation and model-array extraction helpers.
- Serialization remains behind the C ABI functions in the model C-API
  translation unit; the format is fixed and self-contained.

The implementation remains zero-mandatory-dependency C++17. The small linear
algebra needed by NIPALS is implemented with explicit loops over contiguous
`std::vector<double>` buffers.

## 4. Binary Format v1

The v1 format starts with the public header promised in `p4a.h`:

```text
magic[4] = "P4AM"
u32 format_version = 1
u32 writer_abi_major
u32 writer_abi_minor
u32 writer_abi_patch
```

The payload then stores the fitted-model metadata and all arrays required for
predict/transform/get-array. Multi-byte fields are little-endian. Doubles are
IEEE-754 binary64 little-endian.

Imports reject:

- bad magic or truncated buffers with `P4A_ERR_CORRUPT_BUFFER`,
- unsupported format versions with `P4A_ERR_VERSION_INCOMPATIBLE`,
- impossible dimensions or array lengths with `P4A_ERR_CORRUPT_BUFFER`.

## 5. Tests

Phase 1 replaces `abi/test_model_stubs.cpp` with:

- lifecycle and null-pointer checks for live model handles,
- PLS1 / PLS2 fixture parity checks,
- caller-provided and core-allocated predict/transform checks,
- model-array shape/value checks,
- serialization inspect/export/import round-trip checks,
- invalid input checks for shape, dtype and NaN/Inf behavior.

The test harness remains dependency-free. Fixture expectations used by the C++
tests are generated mechanically from the checked-in JSON fixtures.
