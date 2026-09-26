# `regression_metrics` — n4m.metrics.scoring.regression_metrics

_Namespace_: **`n4m.metrics.scoring`** · _Fully-qualified_: `n4m.metrics.scoring.regression_metrics` · _Catalog id_: `diagnostics.regression_metrics`

## API surface

**C ABI (ABI 2):** [`n4m_metrics_regression_metrics_bias`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L18) · [`n4m_metrics_regression_metrics_mae`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L15) · [`n4m_metrics_regression_metrics_nrmse`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L33) · [`n4m_metrics_regression_metrics_r2`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L30) · [`n4m_metrics_regression_metrics_rmse`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L12) · [`n4m_metrics_regression_metrics_rpd`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L24) · [`n4m_metrics_regression_metrics_rpiq`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L27) · [`n4m_metrics_regression_metrics_sep`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L21). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.metrics.scoring import nirs_metrics`

**Signature:** [`nirs_metrics(y_true, y_pred)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10641)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `y_true` | `—` | `required` |
| `y_pred` | `—` | `required` |

## Explanations

### Bibliographic source

No single paper defines this API bundle. RPD usage in NIR calibration is reviewed by Williams & Sobering (1993); RPIQ by Bellon-Maurel et al. (2010), https://doi.org/10.1016/j.trac.2010.07.005. RMSE, MAE, bias, SEP, R², and NRMSE use the explicit formulas in the implementation.

### Mathematical principle

For residuals $e_i=\hat y_i-y_i$, the API computes RMSE, MAE, mean bias, and population SEP $=\mathrm{sd}(e)$. It also returns $RPD=\mathrm{sd}(y)/SEP$, $RPIQ=IQR(y)/RMSE$, $R^2=1-SSE/SST$, and $NRMSE=RMSE/\bar y$; quartiles use linear interpolation.

### Appropriate uses

Reporting predictive error, systematic bias, variance-normalized performance, and robust range-normalized performance for calibration/validation results.

### Limits and validation

RPD/RPIQ depend on the response distribution and cannot be compared blindly across populations. NRMSE divides by the signed response mean: it can be negative when that mean is negative and its magnitude is unstable near zero (infinite only at exactly zero). RPD/RPIQ diverge for perfect predictions. This implementation uses population (`ddof=0`) SEP and SD.

### Implementation

Python functions `n4m.metrics.scoring.rmse`, `mae`, `bias`, `sep`, `rpd`, `rpiq`, `r2`, and `nrmse` wrap the eight `n4m_metrics_regression_metrics_<name>` ABI-2 symbols; formulas are in `cpp/src/core/utilities/nirs_metrics.c`.

### Sources and provenance

https://doi.org/10.1016/j.trac.2010.07.005; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/nirs_metrics.c


_See also_: [methods index](index.md).