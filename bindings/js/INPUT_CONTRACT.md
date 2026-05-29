# @nirs4all/methods-wasm — input contract

The WASM binding is a **non-idiomatic function library**: it accepts raw typed
arrays and returns typed arrays. Aggregation / estimator ergonomics live in the
separate `nirs4all-lite` repo, not here.

## Matrix input (`Matrix`)

```ts
interface Matrix {
  data: Float64Array;  // ROW-MAJOR, length === rows * cols
  rows: number;        // n_samples
  cols: number;        // n_features (a.k.a. wavelengths/bands)
}
```

- `data` is **row-major**: element (i, j) is at `data[i * cols + j]`.
- `data.length` MUST equal `rows * cols` (the binding throws otherwise).
- Values are IEEE-754 doubles (the engine is f64 end-to-end).

## X (spectra) — required
The spectral matrix. `rows` = samples, `cols` = wavelengths. Pass as a `Matrix`.

## wavelengths — optional
A `Float64Array` of length `X.cols` giving the wavelength axis (nm). Only a few
methods use it (edge/spline augmenters, some preprocessing). Omit (or pass
`undefined`) when the method does not need a wavelength axis; the binding does
not synthesize one.

## y (targets) — optional
Supervised methods (PLS/PCR/selectors) need targets. Pass as a `Matrix` with
`cols >= 1` (multi-target supported). A 1-D target is `{data: y, rows: n, cols: 1}`.
Unsupervised/transform methods (SNV, MSC, SG, ...) take no `y`.

## Output
- `fitPls` / `predictPls` return plain JS data (nothing to free).
- The generic `MethodResult`-based path returns the universal
  `n4m_method_result_t` container; read named outputs with
  `result.matrix("coefficients")`, `result.vectorInt("selected")`,
  `result.scalar("rmse")`, then `result.destroy()`.

## Memory
WASM memory is manual. Every handle (`Context`, `Config`, `MethodResult`) has a
`destroy()`; call it. `fitPls`/`predictPls` return plain JS data (nothing to free).

## Generic method access — current status
`fitPls`/`predictPls` (and the other raw-pointer shims in
`bindings/js/src/wasm_entry.c`) are callable end-to-end today. The **generic
"call any of the 188 methods by id"** path is **not yet enabled**: Emscripten
miscompiles JS-built `n4m_matrix_view_t*` parameters to the deep numeric
entrypoints (documented in `wasm_entry.c`), so methods that take a matrix-view
argument must be reached through a C raw-pointer shim. Closing that gap (an
Emscripten-codegen fix, or per-family raw-pointer shims) is a tracked follow-up;
see the README "Generic method path" section.
