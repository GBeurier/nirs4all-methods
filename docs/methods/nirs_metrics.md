# `nirs_metrics` — NIRS regression metrics

_Group_: **Metric** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/nirs_metrics.md`](../algorithms/nirs_metrics.md)

## Description

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

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `nirs4all.core.metrics._eval_single` for the canonical Python formulae.
- Fixture: `parity/fixtures/nirs_metrics_v1.json`.

### Mathematical principle

`nirs_metrics` follows the standard metric operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- One forward pass per metric (mean + variance accumulators where needed).
- Identical evaluation order to NumPy's vectorised reductions.
- Parity tolerance vs reference: **1e-13 absolute or relative** (closed-form
  arithmetic with no iteration). The fixture stores bit-exact hex doubles;
  the test passes whenever the relative gap is below the tolerance.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_metric_bias (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_mae (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_nrmse(const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_r2 (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rmse (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rpd (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rpiq (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_sep (const double* y_true, const double* y_pred, int64_t n, double* out);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_metric_` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_metric_bias (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_mae (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_nrmse(const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_r2 (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rmse (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rpd (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_rpiq (const double* y_true, const double* y_pred, int64_t n, double* out);
c4a_status_t c4a_metric_sep (const double* y_true, const double* y_pred, int64_t n, double* out);
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
