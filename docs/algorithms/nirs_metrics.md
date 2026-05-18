# NIRS regression metrics

Eight closed-form regression metrics over `(y_true, y_pred)` pairs of
length `n`. Mirrors `nirs4all.core.metrics._eval_single`.

| Metric | Formula | ABI |
|--------|---------|-----|
| **RMSE**  | `sqrt(mean((y_true - y_pred)²))`                    | `c4a_metric_rmse(y_true, y_pred, n, *out)` |
| **MAE**   | `mean(|y_true - y_pred|)`                           | `c4a_metric_mae(y_true, y_pred, n, *out)` |
| **Bias**  | `mean(y_pred - y_true)`                             | `c4a_metric_bias(y_true, y_pred, n, *out)` |
| **SEP**   | `std(y_pred - y_true)` (ddof = 0)                   | `c4a_metric_sep(y_true, y_pred, n, *out)` |
| **RPD**   | `std(y_true) / SEP`                                  | `c4a_metric_rpd(y_true, y_pred, n, *out)` |
| **RPIQ**  | `IQR(y_true) / RMSE` (linear interp Q1, Q3)         | `c4a_metric_rpiq(y_true, y_pred, n, *out)` |
| **R²**    | `1 - SSE / SST` (SST = `Σ(y_true - mean(y_true))²`) | `c4a_metric_r2(y_true, y_pred, n, *out)` |
| **NRMSE** | `RMSE / mean(y_true)`                                | `c4a_metric_nrmse(y_true, y_pred, n, *out)` |

## Edge cases

- `n <= 0` returns `C4A_ERR_INVALID_ARGUMENT`.
- `NULL` for any of the three pointer arguments returns
  `C4A_ERR_NULL_POINTER`.
- RPD: when SEP is exactly zero, returns `+inf` (matches the Python
  reference which falls through to `float('inf')`).
- RPIQ: when RMSE is exactly zero, returns `+inf`.
- NRMSE: when `mean(y_true)` is exactly zero, returns `+inf`.
- R²: when SST is zero (constant `y_true`), returns 1.0 if SSE is also
  zero, else 0.0.

## Numerical contract

- One forward pass per metric (mean + variance accumulators where needed).
- Identical evaluation order to NumPy's vectorised reductions.
- Parity tolerance vs reference: **1e-13 absolute or relative** (closed-form
  arithmetic with no iteration). The fixture stores bit-exact hex doubles;
  the test passes whenever the relative gap is below the tolerance.

## ABI

```c
C4A_API c4a_status_t c4a_metric_rmse (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_mae  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_bias (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_sep  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_rpd  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_rpiq (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_r2   (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
```

## Reference

- `nirs4all.core.metrics._eval_single` for the canonical Python formulae.
- Fixture: `parity/fixtures/nirs_metrics_v1.json`.
