# YOutlierFilter (Y-based outlier detection)

First member of the `c4a_filter_*` ABI category (Phase 12). Identifies samples
whose 1-D target (`y`) values are statistical outliers using one of four
detection methods, and returns a per-sample keep/exclude mask together with a
stats record.

Mirrors `nirs4all.operators.filters.YOutlierFilter` at
`/home/delete/nirs4all/nirs4all/nirs4all/operators/filters/y_outlier.py`.

## Methods

| `method`     | Bound formula                                          | Default `threshold` |
| ------------ | ------------------------------------------------------ | ------------------- |
| `iqr`        | `[Q1 - t * IQR, Q3 + t * IQR]`                         | 1.5 (mild outliers) |
| `zscore`     | `[mu - t * sigma, mu + t * sigma]`                     | 3.0 (3-sigma rule)  |
| `percentile` | `[pct(y, lower_pct), pct(y, upper_pct)]`               | -                   |
| `mad`        | `[median +/- t * MAD * 1.4826]`                        | 3.5                 |

- Quantiles use NumPy 1.26.4's default `"linear"` interpolation.
- `np.std` uses `ddof=0` (biased estimator, NumPy default).
- The `mad` method's `1.4826` factor is the Gaussian-consistency normaliser
  (`sigma ~= 1.4826 * MAD` on normal data).

## Mask semantics

- `mask[i] == 1` keeps sample `i` (within bounds and not NaN).
- `mask[i] == 0` excludes sample `i` (outside bounds OR `y[i]` is NaN).

## Degenerate inputs

- `n == 0`: returns empty mask and `stats = {0, 0, 0, 0.0}`.
- `y` is all-NaN: every sample is excluded.
- exactly one non-NaN value: bounds collapse to `[v - 1e-10, v + 1e-10]`
  (matches the Python reference).
- zero scale (constant `y` for `zscore` / `mad`): same `+/- 1e-10` collapse,
  no division by zero.

## Parameters

- `method`: one of `iqr`, `zscore`, `percentile`, `mad`.
- `threshold`: > 0. Multiplier for `iqr`, `zscore`, `mad`. Ignored for
  `percentile`.
- `lower_percentile`, `upper_percentile`: only used by `percentile`. Must
  satisfy `0 <= lower < upper <= 100`.

## ABI

```c
typedef enum c4a_y_outlier_method_t {
    C4A_Y_OUTLIER_IQR        = 0,
    C4A_Y_OUTLIER_ZSCORE     = 1,
    C4A_Y_OUTLIER_PERCENTILE = 2,
    C4A_Y_OUTLIER_MAD        = 3
} c4a_y_outlier_method_t;

typedef struct c4a_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} c4a_filter_stats_t;

typedef struct c4a_filter_y_outlier_handle_t c4a_filter_y_outlier_handle_t;

c4a_status_t c4a_filter_y_outlier_create(c4a_filter_y_outlier_handle_t** out,
                                          c4a_y_outlier_method_t method,
                                          double threshold,
                                          double lower_pct, double upper_pct);
void         c4a_filter_y_outlier_destroy(c4a_filter_y_outlier_handle_t* h);

/* Stateful fit/apply split (matches sklearn). */
c4a_status_t c4a_filter_y_outlier_fit(
    c4a_filter_y_outlier_handle_t* h,
    const double* y, int64_t n);
c4a_status_t c4a_filter_y_outlier_apply(
    const c4a_filter_y_outlier_handle_t* h,
    const double* y, int64_t n,
    uint8_t* mask_out,
    c4a_filter_stats_t* stats_out);
c4a_status_t c4a_filter_y_outlier_is_fitted(
    const c4a_filter_y_outlier_handle_t* h, int* out);
```

`_fit` learns the per-method bounds from the supplied y vector; subsequent
calls overwrite the previously fitted bounds. `_apply` writes the keep-mask
+ stats using the previously learned bounds (returns `C4A_ERR_NOT_FITTED`
when called on an unfitted handle). `_is_fitted` reports whether `_fit` has
been called. The mask and stats buffers are caller-allocated.

## Numerical contract

- Bounds use NumPy linear-interpolation percentiles (closed-form).
- Z-score uses the biased (`ddof=0`) standard deviation.
- MAD uses the unbiased median of `|y - median(y)|`, scaled by `1.4826`.
- Parity tolerance vs the frozen NumPy reference: **exact** equality for the
  mask bytes and integer counts (`n_samples`, `n_kept`, `n_excluded`),
  **1e-12 abs** for the `exclusion_rate` (single division by `n`).
- Reference: `parity/python_generator/src/c4a_parity_filters_ref/y_outlier.py`.

## Reference

- `nirs4all.operators.filters.YOutlierFilter`
- See `roadmap/phase-12-y-filter.md` for the brief and acceptance criteria.
