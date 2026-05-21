# `ds` — Direct Standardisation

_Group_: **Calibration transfer** · _Registry tolerance_: `0.5`

## Description

DS — Direct Standardization (§13)

From the `pls4all.sklearn.DSTransformer` docstring:

> Direct Standardization — full-band cross-instrument regression.

> **Registry note** — Base R per-band `lm.fit` with window_half_width=0 — Direct Standardization is just per-band linear regression.

_No tunable parameters declared at the binding level._

## Explanations

### Bibliographic source

Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). *Multivariate instrument standardisation*. Analytical Chemistry 63(23), 2750–2756.

### Mathematical principle

Direct Standardisation (DS) learns a **single global transfer matrix** $\mathbf{F}$ that maps secondary-instrument spectra back onto the primary instrument: $\hat{\mathbf{X}}_{\mathrm{primary}} = \mathbf{X}_{\mathrm{secondary}} \mathbf{F}$. The matrix is fit by least squares on a small set of transfer-standard samples measured on both instruments, typically with a Tikhonov ridge for stability.

DS is the simplest member of the calibration-transfer family and works well when the inter-instrument response is approximately linear and stationary — i.e. when instrument differences amount to a constant linear transformation that is invariant across the spectrum. When the transformation varies across wavelength bands (common in dispersive vs FT spectrometers) the global transfer matrix produces band-mixing artefacts and PDS should be used instead.

A canonical workflow: fit DS on $\le 30$ transfer standards, apply $\mathbf{F}$ to all subsequent secondary-instrument spectra, then use a single PLS model fit only on primary data.

### Implementation

`p4a_ds_fit` (TransformerMixin in tier 2). Reference: R `chemometrics::stdize`.

R roxygen note (`methods_extra.R::ds_fit`):

> Direct Standardization (calibration transfer).
> @param X_source Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X_target Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @export

MATLAB header (`bindings/matlab/+pls4all/ds.m`):

```text
pls4all.ds  Direct Standardization (calibration transfer).
```

### Usage

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libp4a
:sync: c
:class-label: lang-c

```c
/* C ABI — libp4a */
p4a_context_t* ctx = p4a_context_create();
p4a_config_t*  cfg = p4a_config_create();
p4a_method_result_t* res = NULL;
p4a_ds_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* p4a_method_result_get_double_matrix / vector / scalar … */
p4a_method_result_destroy(res);
p4a_config_destroy(cfg);
p4a_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all._methods import ds_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = ds_fit(ctx, cfg, X, y, X_target=X_target)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import DSTransformer
mdl = DSTransformer(n_components=2)
mdl.fit(X, y, X_target=X_target)
y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("ds", X, y,
                      n_components = 2L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- ds_fit(X_source, X_target)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.ds(X, y, 2);
% see header of bindings/matlab/+pls4all/ds.m for full
% parameter surface:
%   res = ds(X_source, X_target)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("ds", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_base`** (R · r) — `base` R 4.3.3 · qualitative (rmse_rel ≤ 5e-01) — Base R `lm` per spectral band — closest installable analog to Wang 1991 Piecewise Direct Standardization. With window_half_width=0 this reduces to Direct Standardization.
:::

### Validation contract

Structural validation contract for `ds`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `external` |
| Canonical reference role | `r` |
| Declared references | R |
| Comparator | `reference_parity` (external-reference validity gate) |
| Comparator metric | `rmse_rel` |
| Reference tolerance | `5e-01` (qualitative) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | `X_target` |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_features` | `30` |
| `n_samples` | `200` |

**Validation suites**

- **`benchmark`** — Full cross-binding sweep - every MethodSpec across every default dataset size. Mirrors the cross-binding orchestrator's `DEFAULT_SIZES` surface. Datasets: `10000x50`, `10000x500`, `100x2500`, `100x50`, `100x500`, `2500x2500`, `2500x50`, `2500x500`, `500x2500`, `500x50`, `500x500`. Comparators: `binding_parity`, `reference_parity`.
- **`smoke`** — Fastest cross-binding cells for every MethodSpec; used by pre-commit / CI smoke gates. Datasets: `100x50`, `100x500`. Comparators: `binding_parity`, `reference_parity`.

**Dataset cells referenced by these suites**

| Dataset | n | p |
|---------|---|---|
| `100x50` | 100 | 50 |
| `100x500` | 100 | 500 |
| `100x2500` | 100 | 2500 |
| `500x50` | 500 | 50 |
| `500x500` | 500 | 500 |
| `500x2500` | 500 | 2500 |
| `2500x50` | 2500 | 50 |
| `2500x500` | 2500 | 500 |
| `2500x2500` | 2500 | 2500 |
| `10000x50` | 10000 | 50 |
| `10000x500` | 10000 | 500 |

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Divergence** is the worst finite value over the visible sizes for each backend: reference-gate rows show `rmse_rel`, binding-gate rows show `max_diff` against the C++ baseline. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.

**Reference gate**: non-release diagnostic — shape/smoke comparison only (`rmse_rel ≤ 5e-01`). Treat it as evidence that both implementations ran, not as numerical agreement.

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">1000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-08</td><td class="ms ms-best">1.29 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">92.8 ms</td><td class="ms">21.8 s</td><td class="ms">872.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-08</td><td class="ms">1.33 ms</td><td class="ms">91.4 ms</td><td class="ms">24.0 s</td><td class="ms">853.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-08</td><td class="ms">1.33 ms</td><td class="ms ms-best">90.3 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">24.2 s</td><td class="ms">837.5 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">2e-08</td><td class="ms">1.31 ms</td><td class="ms">92.2 ms</td><td class="ms">21.5 s</td><td class="ms">868.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">1.41 ms</td><td class="ms">90.4 ms</td><td class="ms">23.1 s</td><td class="ms">827.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">1.53 ms</td><td class="ms">92.4 ms</td><td class="ms">23.0 s</td><td class="ms ms-best">814.3 ms<span class="medal" title="fastest">🏆</span></td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">8.07 ms</td><td class="ms">195.4 ms</td><td class="ms">24.5 s</td><td class="ms">1.9 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">9.26 ms</td><td class="ms">211.6 ms</td><td class="ms">25.0 s</td><td class="ms">1.8 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-drift">2e-09</td><td class="ms">10.4 ms</td><td class="ms">234.8 ms</td><td class="ms">23.8 s</td><td class="ms">1.9 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-drift">2e-09</td><td class="ms">10.1 ms</td><td class="ms">227.1 ms</td><td class="ms">25.7 s</td><td class="ms">1.8 s</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">3.60 ms</td><td class="ms">113.7 ms</td><td class="ms">24.2 s</td><td class="ms">967.2 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-drift">2e-08</td><td class="ms">5.76 ms</td><td class="ms">112.6 ms</td><td class="ms">23.3 s</td><td class="ms">1.0 s</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): base R 4.3.3 — qualitative (rmse_rel ≤ 5e-01)">📐</span><code>ref.r_base</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">24.5 ms</td><td class="ms">227.7 ms</td><td class="ms ms-best">1.9 s<span class="medal" title="fastest">🏆</span></td><td class="ms">2.1 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)