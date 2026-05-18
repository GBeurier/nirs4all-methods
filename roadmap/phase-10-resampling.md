# Phase 10 — Resampling, cropping, and target discretization

## Goal

Implement five additional preprocessing operators completing the spectral-
grid manipulation and target-discretization toolkit:

  * **Resampler**               — wavelength-grid interpolation between a
                                  fitted source axis and a configured target
                                  axis (linear / nearest / cubic methods).
                                  Stateful (`_fit` learns source/target grid
                                  mapping).
  * **CropTransformer**         — slice columns `[start, end)` from X.
                                  Stateless.
  * **ResampleTransformer**     — resample every row to a fixed target column
                                  count via linear interpolation on the
                                  `linspace(0, 1, n)` axis. Stateless.
  * **IntegerKBinsDiscretizer** — sklearn KBinsDiscretizer with integer
                                  output (uniform / quantile strategy).
                                  Stateful (`_fit` learns bin edges).
  * **RangeDiscretizer**        — `np.digitize(X, bins, right=False)` over a
                                  fixed edge vector. Stateless.

## Operators

| Operator | Algorithm | Type | Symbols |
|----------|-----------|------|--------:|
| `c4a_pp_resampler_*` | scipy interp1d (linear / nearest / cubic) | stateful | 6 |
| `c4a_pp_crop_*` | column slice `[start, end)` | stateless | 4 |
| `c4a_pp_resample_*` | linear interp to fixed N on `[0, 1]` axis | stateless | 4 |
| `c4a_pp_kbins_disc_*` | sklearn KBinsDiscretizer (ordinal, int32) | stateful | 5 |
| `c4a_pp_range_disc_*` | np.digitize (int32) | stateless | 3 |

Total **22 new ABI symbols** (126 → 148). The Resampler also exposes
`_output_cols(handle)` (post-fit) so callers can pre-allocate the output
matrix view.

## Algorithm details

### Resampler

Per-row interpolation between two strictly-ascending wavelength axes. The
fit step caches the source axis (post-crop) and pre-computes per-target
bracketing indices. For `method == cubic` we also build the segment widths
`h[i]` and a pre-allocated 4*n scratch buffer for the per-row Thomas-algorithm
solution of the not-a-knot tridiagonal system.

Per-row hot path (no allocation):
- **linear**: `slope = (y[hi] - y[lo]) / (x[hi] - x[lo])`,
              `out = slope * (q - x[lo]) + y[lo]`
- **nearest**: scipy's bin-midpoint search `(x[i] + x[i+1]) / 2`,
               `side='left'` rounding (replicates `interp1d(kind='nearest')`).
- **cubic**: not-a-knot cubic spline. Per-row Thomas sweep over a 3-banded
             system reduced to a strictly-tridiagonal interior via Gaussian
             elimination of the boundary rows. Spline evaluation uses
             Horner-form per query.

Bounds handling: `bounds_error == 1` rejects out-of-range queries with
C4A_ERR_NUMERICAL_FAILURE; `bounds_error == 0` either fills with `fill_value`
(if `extrapolate == 0`) or extrapolates the boundary segment.

### CropTransformer

`end == -1` mirrors nirs4all's `end is None` sentinel (slice to end of row).
The per-call memcpy is byte-exact (no arithmetic).

### ResampleTransformer

`num_samples == -1` is the identity passthrough (nirs4all's `num_samples is
None`). When `cols == out_cols` we also passthrough verbatim — nirs4all's
fast-path that skips interp1d. The interpolation axis is the
`np.linspace(0, 1, n)` arithmetic progression with the endpoint pinned to
`1.0` (numpy convention).

### IntegerKBinsDiscretizer

Per-column edges computed at fit time:
- **uniform**: `linspace(col_min, col_max, n_bins + 1)`.
- **quantile**: `np.percentile(col, linspace(0, 100, n_bins + 1))` with
  linear interpolation.

sklearn drops bins with width <= 1e-8 (per column); we mirror this dedup
step so the inner-edge search is stable on degenerate columns (e.g. constant
features).

Transform per column: `searchsorted(edges[1:-1], col, side='right')`. Output
is row-major int32.

### RangeDiscretizer

`np.digitize(X, bins, right=False)` ≡ `searchsorted(bins, x, side='right')`.
Bins must be strictly ascending. Output labels in `[0, len(bins)]` (length
`len(bins) + 1` virtual buckets). Output is row-major int32.

## Files created

### Engines (`cpp/src/core/preprocessing/resampling/` — new subdir)
- `resampler.{c,h}`
- `crop.{c,h}`
- `resample_transformer.{c,h}`
- `kbins_discretizer.{c,h}`
- `range_discretizer.{c,h}`

### C ABI
- `cpp/src/c_api/c_api_resampling.cpp`  — 22 wrappers.

### Tests
- `cpp/tests/test_preprocessing_resampling.cpp` — 5 smoke + 5 parity + 2
  not-fitted = **12 tests**.

### Fixtures
- `parity/fixtures/{resampler, crop, resample, kbins_disc, range_disc}_v1.json`

### Fixture generator
- `parity/python_generator/scripts/generate_phase10_fixtures.py`

### Brief
- `roadmap/phase-10-resampling.md` (this file).

## Files modified

- `cpp/src/CMakeLists.txt` — add 5 new core sources + `c_api_resampling.cpp`.
- `cpp/tests/CMakeLists.txt` — add `test_preprocessing_resampling.cpp` to
  the test executable.

## Files NOT modified (central integration)

- `cpp/include/chemometrics4all/c4a.h` — integrator appends §14 banner +
  22 ABI declarations.
- `cpp/include/chemometrics4all/c4a_version.h` — integrator bumps MINOR
  `6 → 7` (1.6.0 → 1.7.0).
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` — integrator
  regenerates (126 → 148).
- `cpp/tests/main.cpp` — integrator appends the
  `register_preprocessing_resampling_tests(r)` call.

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol | Notes |
|----------|-----------|---------|---------|-------|
| Resampler (linear / nearest) | scipy interp1d | 1e-12 | 1e-13 | Closed-form arithmetic, IEEE-ordered. |
| Resampler (cubic)            | scipy CubicSpline `not-a-knot` | 1e-10 | 1e-11 | Thomas-vs-scipy-LAPACK gap on tridiagonal system; few ULPs in practice. |
| CropTransformer              | nirs4all CropTransformer | 0 | 0 | Byte-exact memcpy. |
| ResampleTransformer          | nirs4all ResampleTransformer | 1e-12 | 1e-13 | scipy interp1d linear path. |
| IntegerKBinsDiscretizer      | sklearn KBinsDiscretizer | int eq | int eq | Exact integer equality. |
| RangeDiscretizer             | nirs4all RangeDiscretizer | int eq | int eq | Exact integer equality. |

## Verification

```bash
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests
# After central integration: 82 + 12 = 94 tests pass.
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.7.0 \
    | awk '$2=="T" {print $3}' | sort -u | wc -l
# 148
```

Local validation in this worktree confirmed 94/94 tests pass with a
temporary main.cpp patch (subsequently reverted). The integrator will
re-register the suite as part of the c4a.h / main.cpp updates.
