# `pds` — Piecewise Direct Standardisation

_Group_: **Calibration transfer** · _Registry tolerance_: `0.5`

## Description

PDS — Piecewise Direct Standardization (§13)

From the `pls4all.sklearn.PDSTransformer` docstring:

> Piecewise Direct Standardization (Wang 1991).

> **Registry note** — Base R per-band `lm.fit` over a window of source bands — the canonical Wang 1991 PDS algorithm with no extra deps. Same algorithm as pls4all's pds_fit modulo CSV-roundtrip precision.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `window_half_width` | `int` | `5` | Half-width (in channels) of the PDS local regression window. |
| `window_size` | `int` | `5` | Length of the moving window for recursive / interval-random-frog models. |

## Explanations

### Bibliographic source

Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). *Multivariate instrument standardization*. Analytical Chemistry 63(23), 2750–2756. https://doi.org/10.1021/ac00023a016 — same paper as `ds`; PDS is introduced in §3 (piecewise local regression with a sliding window of width 2w+1).

### Mathematical principle

PDS generalises DS by allowing the transfer to vary across wavelength bands. For each wavelength $j$ on the primary instrument, a local regression maps a window of $\pm w$ wavelengths from the secondary instrument: $\hat{x}_{\mathrm{primary}, j} = \mathbf{x}_{\mathrm{secondary}, j-w:j+w} \cdot \mathbf{f}_j$. The full transfer matrix is then **banded**: only $\pm w$ off-diagonal columns per row are non-zero.

PDS handles wavelength-dependent inter-instrument behaviour — wavelength-axis drift, resolution differences, detector non-linearities — that DS cannot. The window half-width $w$ controls the locality: $w=0$ recovers a diagonal-only transfer, $w \to p/2$ recovers DS.

PDS is the de-facto standard in NIR / FT-IR calibration transfer; the `prospectr` R package's implementation is considered canonical.

### Implementation

`p4a_pds_fit` (TransformerMixin in tier 2). Reference: R `prospectr::pds`. Note: pls4all applies the transpose convention so that `transform(X_secondary)` returns the standardised primary-instrument estimate.

R roxygen note (`methods_extra.R::pds_fit`):

> Piecewise Direct Standardization (calibration transfer).
> @param X_source Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param X_target Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @param window_half_width Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
> @export

MATLAB header (`bindings/matlab/+pls4all/pds.m`):

```text
pls4all.pds  Piecewise Direct Standardization (calibration transfer).
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
p4a_pds_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import pds_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = pds_fit(ctx, cfg, X, y, X_target=X_target)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import PDSTransformer
mdl = PDSTransformer(window_half_width=5)
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
res <- pls4all_method("pds", X, y,
                      n_components = 2L, params = list(window_half_width = 2L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- pds_fit(X_source, X_target, window_half_width = 2L)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.pds(X, y, 2);
% see header of bindings/matlab/+pls4all/pds.m for full
% parameter surface:
%   res = pds(X_source, X_target, window_half_width)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("pds", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.r_base`** (R · r) — `base` R 4.3.3 · qualitative (rmse_rel ≤ 5e-01) — Base R `lm` per spectral band — closest installable analog to Wang 1991 Piecewise Direct Standardization. With window_half_width=0 this reduces to Direct Standardization.
:::

### Validation contract

Structural validation contract for `pds`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

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
| `window_half_width` | `2` |

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
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">1.12 ms</td><td class="ms ms-best">20.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">590.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">183.6 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">1.12 ms</td><td class="ms">20.6 ms</td><td class="ms">596.1 ms</td><td class="ms">189.6 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">1.20 ms</td><td class="ms">20.4 ms</td><td class="ms">596.0 ms</td><td class="ms">192.7 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms ms-best">1.11 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">20.5 ms</td><td class="ms">646.6 ms</td><td class="ms">192.1 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">1.16 ms</td><td class="ms">20.5 ms</td><td class="ms">667.3 ms</td><td class="ms">186.4 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">1.34 ms</td><td class="ms">22.2 ms</td><td class="ms">609.3 ms</td><td class="ms">204.6 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">7.90 ms</td><td class="ms">127.4 ms</td><td class="ms">1.6 s</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">9.49 ms</td><td class="ms">149.7 ms</td><td class="ms">1.8 s</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">9.18 ms</td><td class="ms">149.9 ms</td><td class="ms">1.8 s</td><td class="ms">1.3 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">9.46 ms</td><td class="ms">151.8 ms</td><td class="ms">1.8 s</td><td class="ms">1.3 s</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">3.54 ms</td><td class="ms">38.3 ms</td><td class="ms">704.4 ms</td><td class="ms">382.8 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-ref-qualitative">1e-08</td><td class="ms">4.03 ms</td><td class="ms">38.7 ms</td><td class="ms">803.5 ms</td><td class="ms">380.5 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="6" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-qualitative"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): base R 4.3.3 — qualitative (rmse_rel ≤ 5e-01)">📐</span><code>ref.r_base</code></td><td class="parity parity-divergence parity-ref-source">0</td><td class="ms">23.8 ms</td><td class="ms">219.3 ms</td><td class="ms">2.1 s</td><td class="ms">2.0 s</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)