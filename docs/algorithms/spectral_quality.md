# SpectralQualityFilter (multi-criteria per-row quality check)

Identifies samples whose spectra fail at least one of six quality checks. The filter is stateless: thresholds are constructor parameters and the same checks run on every `apply` call.

## Checks

A row is KEPT if and only if every enabled check passes:

1. **NaN ratio** `<= max_nan_ratio` — fraction of NaN entries per row, must be `<= max_nan_ratio` (range `[0, 1]`).
2. **Inf check** (when `check_inf` is non-zero) — no `+/- inf` is allowed.
3. **Zero ratio** `<= max_zero_ratio` — fraction of exact-zero entries, with NaNs treated as non-zero (mirrors `np.nan_to_num(X, nan=1.0)` in the reference).
4. **Variance** `>= min_variance` — `np.nanvar(row)`. Inf values in the row propagate to a NaN variance and fail the check.
5. **Max value** `<= max_value` (when `use_max != 0`) — `np.nanmax(row) <= max_value`.
6. **Min value** `>= min_value` (when `use_min != 0`) — `np.nanmin(row) >= min_value`.

The mask is exact (mask equality with the frozen reference, no float tolerance) because every check is a single threshold comparison on a deterministic statistic.

## Numerical contract

- Per-row evaluation in a single pass over the data (cache-friendly).
- Variance uses the `np.nanvar` semantics: NaN entries are skipped from the mean/sum-of-squares but Inf entries are included so that any row containing Inf has a NaN variance (which fails the check regardless of `min_variance`).
- Zero-cols input: rejects every row (consistent with the reference's behaviour when `n_features == 0`).

## ABI

```c
c4a_status_t c4a_filter_quality_create(c4a_filter_quality_handle_t** out,
                                        double max_nan_ratio,
                                        double max_zero_ratio,
                                        double min_variance,
                                        int    use_max, double max_value,
                                        int    use_min, double min_value,
                                        int    check_inf);
void         c4a_filter_quality_destroy(c4a_filter_quality_handle_t* h);
c4a_status_t c4a_filter_quality_apply(const c4a_filter_quality_handle_t* h,
                                       c4a_matrix_view_t X,
                                       uint8_t* mask_out,
                                       c4a_filter_stats_t* stats_out);
```

## Reference

- nirs4all 0.8.x — `nirs4all/operators/filters/spectral_quality.py`
- `parity/python_generator/src/c4a_parity_filters_ref/spectral_quality.py`
