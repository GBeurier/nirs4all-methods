# `moments` — n4m.lowlevel.moments.moments

_Namespace_: **`n4m.lowlevel.moments`** · _Fully-qualified_: `n4m.lowlevel.moments.moments` · _Catalog id_: `utilities.moments`

## API surface

**C ABI (ABI 2):** [`n4m_lowlevel_moments_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/lowlevel.h#L13) · [`n4m_lowlevel_moments_subset_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/lowlevel.h#L13) · [`n4m_lowlevel_moments_subtract`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/lowlevel.h#L14). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.lowlevel.moments import moments`

**Signature:** [`moments(X, y, *, row_indices = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L769)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `row_indices` | `—` | `None` |

## Explanations

### Bibliographic source

No single canonical paper defines the ABI utility surface; it implements standard first- and second-moment sufficient statistics used by linear-model updates.

### Mathematical principle

Accumulate quantities such as sample count, feature sums, cross-products, and response cross-products so compatible linear fits can be computed or updated without retaining every original row.

### Appropriate uses

Auditable aggregation and replay of moment-compatible linear calibration calculations.

### Limits and validation

Moments do not retain row identities, nonlinear transformations, or arbitrary fold structure. Merging statistics is only valid for identically defined feature spaces and preprocessing; it cannot recover information discarded upstream.

### Implementation

`n4m.lowlevel.moments`; C ABI symbols in the `n4m_lowlevel_moments_*` family.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/lowlevel/moments.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/lowlevel.h

## Catalog note

Row-additive sufficient-statistics substrate returning raw and centered X'X, X'Y and Y'Y moments. Fold subtraction is exact for train-from-heldout moment screens.

_Timing benchmark_: `benchmarks/cross_binding/bench_moment_sweep_timing.py`


_See also_: [methods index](index.md).