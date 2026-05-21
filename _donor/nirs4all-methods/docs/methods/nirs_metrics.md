# `nirs_metrics` — NIRS metrics

_Group_: **Metric** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/nirs_metrics.md`](../algorithms/nirs_metrics.md)

## Description

Eight closed-form regression metrics over `(y_true, y_pred)` pairs of
length `n`. Mirrors `nirs4all.core.metrics._eval_single`.

| Metric | Formula | ABI |
|--------|---------|-----|
| **RMSE**  | `sqrt(mean((y_true - y_pred)²))`                    | `n4m_metric_rmse(y_true, y_pred, n, *out)` |
| **MAE**   | `mean(|y_true - y_pred|)`                           | `n4m_metric_mae(y_true, y_pred, n, *out)` |
| **Bias**  | `mean(y_pred - y_true)`                             | `n4m_metric_bias(y_true, y_pred, n, *out)` |
| **SEP**   | `std(y_pred - y_true)` (ddof = 0)                   | `n4m_metric_sep(y_true, y_pred, n, *out)` |
| **RPD**   | `std(y_true) / SEP`                                  | `n4m_metric_rpd(y_true, y_pred, n, *out)` |
| **RPIQ**  | `IQR(y_true) / RMSE` (linear interp Q1, Q3)         | `n4m_metric_rpiq(y_true, y_pred, n, *out)` |
| **R²**    | `1 - SSE / SST` (SST = `Σ(y_true - mean(y_true))²`) | `n4m_metric_r2(y_true, y_pred, n, *out)` |
| **NRMSE** | `RMSE / mean(y_true)`                                | `n4m_metric_nrmse(y_true, y_pred, n, *out)` |

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
n4m_status_t n4m_metric_bias (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_mae (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_nrmse(const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_r2 (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rmse (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rpd (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rpiq (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_sep (const double* y_true, const double* y_pred, int64_t n, double* out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_metric_` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.nirs_metrics` | Python | ABI-close function backed by ctypes. |
| ref.numpy | `NumPy regression metric formulae` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_metric_bias (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_mae (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_nrmse(const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_r2 (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rmse (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rpd (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_rpiq (const double* y_true, const double* y_pred, int64_t n, double* out);
n4m_status_t n4m_metric_sep (const double* y_true, const double* y_pred, int64_t n, double* out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.nirs_metrics(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.numpy`** (Python · canonical) — `NumPy regression metric formulae` · numpy 2.3.5
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.numpy` | `NumPy regression metric formulae` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.1e-15</td><td class="ms ms-best">🏆 0.036 ms</td><td class="ms">0.040 ms</td><td class="ms">0.041 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.1e-15</td><td class="ms">0.038 ms</td><td class="ms ms-best">🏆 0.038 ms</td><td class="ms ms-best">🏆 0.040 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.1e-15</td><td class="ms">0.045 ms</td><td class="ms">0.045 ms</td><td class="ms">0.047 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): NumPy regression metric formulae · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.055 ms</td><td class="ms">0.062 ms</td><td class="ms">0.060 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
