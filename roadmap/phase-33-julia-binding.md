# Phase 33 — Julia binding + cross-binding parity helper

**Status:** shipped — local tag `phase-33-julia-binding`
(`0.85.0+abi.1.13.0`).

## Summary

Adds the Julia binding plus the public ABI helper that the WASM
binding already needed: `p4a_pls_fit_simple`. Both the new Julia
binding and the existing WASM binding now go through that helper,
sidestepping a `Ref{matrix_view_t}` interop ambiguity that bit both
Emscripten 5.0.7 and Julia 1.12 ccall.

## New public C ABI symbol

```c
p4a_status_t p4a_pls_fit_simple(
    const double* x, const double* y,
    int32_t n, int32_t p, int32_t q, int32_t n_components,
    double* coefficients_out,
    double* x_mean_out,
    double* y_mean_out,
    double* predictions_out);
```

- Row-major X (n × p), Y (n × q).
- Defaults: SIMPLS, `center_x = center_y = scale_x = scale_y = 1`.
- Optional `predictions_out` (NULL skips in-sample predict).
- ABI minor bumped 1.12 → 1.13.

## Julia binding

```
bindings/julia/Pls4all.jl/
├── Project.toml
├── src/Pls4all.jl       # ccall wrappers
└── test/test_parity.jl  # parity vs bindings/js/test/parity_fixture.json
```

Usage:

```julia
ENV["PLS4ALL_LIB"] = "/path/to/libp4a.so"
using Pls4all

X = ...; Y = ...                       # Julia column-major Float64
m = Pls4all.pls_fit(X, Y; n_components = 3)
m.coefficients   # (p × q) Array{Float64,2}
m.x_mean         # (p,) Vector{Float64}
m.y_mean         # (q,) Vector{Float64}
m.predictions    # (n × q) Array{Float64,2}
```

## R binding alignment

`bindings/r/pls4all/src/r_gateway.c` now sets `scale_x = scale_y =
1` to align with the Python binding's defaults. Before this change
the R binding diverged from the Python fixture by ~7.6e-6 because R
was hard-coding `scale = 0`. With the alignment R matches Python
exactly (rmse_rel = 0).

Also added: `bindings/r/test_parity.R` running the same parity gate
as the Julia / WASM bindings.

## Cross-binding parity matrix

| Binding | reference | rmse_rel |
|---------|-----------|----------|
| Python | (self) | — |
| R (`pls4all`) | Python fixture | 0.0 (exact) |
| Julia (`Pls4all.jl`) | Python fixture | 1.7e-16 |
| WASM (`@pls4all/wasm`) | Python fixture | 2.1e-16 |

All bindings match the native Python at numerical floor on the same
deterministic 50 × 5 SIMPLS regression cell.

## Local gate

- 256 native C++ tests pass.
- ABI symbol diff: 160 (+1, clean).
- `ldd`: only libc/libstdc++/libgcc/libm/loader.
- All four binding-level parity gates: PASS at numerical floor.
- `git diff --check`: clean.
