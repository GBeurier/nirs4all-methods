# `mir_pls` — MIR-PLS (Mid-InfraRed PLS, regularised)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `0.05`
 · _Parity reference_: **paper-only** (Sjöblom, J., Svensson, O., Josefson, M., Kullberg, H. & Wold, S. (1998). An evaluation of orthogonal signal correction applied to calibration transfer of near infrared spectra. Chemom. Intell. Lab. Syst. 44(1-2), 229-244. (Inverse-regression PLS variant; no widely installable port.))

## Description

MIR-PLS — Inverse-regression PLS (§13)

From the `pls4all.sklearn.MIRPLSRegression` docstring:

> Multivariate Inverse Regression PLS (Sjöblom 1998).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |

## Explanations

### Bibliographic source

Sjöblom, J., Svensson, O., Josefson, M., Kullberg, H. & Wold, S. (1998). *An evaluation of orthogonal signal correction applied to calibration transfer of near infrared spectra*. Chemometrics and Intelligent Laboratory Systems 44(1–2), 229–244. — adapted for MIR regularisation conventions.

### Mathematical principle

MIR-PLS is a PLS variant tuned to mid-infrared spectroscopy conventions: it operates in absorbance space, uses a different ridge constant on the inner regression, and exposes a coefficient-export path that matches the mid-IR community's expectations (sign and ordering of loadings agree with `plsregress` rather than `pls::plsr`).

Algorithmically it is a SIMPLS variant with a fixed ridge constant in the inner regression and an alternate intercept derivation. The differences from plain PLS are numerical, not algorithmic — the resulting predictions are within FP noise of plain PLS for well-conditioned inputs but more stable on the highly-correlated columns typical of MIR FTIR spectra.

### Implementation

`p4a_mir_pls_fit`. No widely installable library reference; treated as `paper_only` in the registry.

R roxygen note (`sklearn_methods.R::mir_pls`):

> MIR-PLS — formula entry point.

MATLAB header (`bindings/matlab/+pls4all/MirRegression.m`):

```text
pls4all.MirRegression — Multivariate Inverse Regression PLS.
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
p4a_mir_pls_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
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
from pls4all._methods import mir_pls_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = mir_pls_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import MIRPLSRegression
mdl = MIRPLSRegression(n_components=2)
mdl.fit(X, y)
y_hat = mdl.predict(X_test)
```

:::

:::{tab-item} R · pls4all_method()
:sync: r-dispatcher
:class-label: lang-r

```r
library(pls4all)
# Unified low-level dispatcher (May 2026 R cleanup):
res <- pls4all_method("mir_pls", X, y,
                      n_components = 4L)
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} R · pls4all (raw fn)
:sync: r-raw
:class-label: lang-r

```r
library(pls4all)
res  <- mir_pls_fit(X, Y, n_components)
yhat <- pls4all_predict(res, X_test)
```

:::

:::{tab-item} R · pls4all (formula+S3)
:sync: r-formula
:class-label: lang-r

```r
library(pls4all)
fit  <- mir_pls(y ~ ., data = train, ncomp = 4L)
yhat <- predict(fit, newdata = test)
summary(fit)
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.mir_pls(X, y, 4);
% see header of bindings/matlab/+pls4all/mir_pls.m for full
% parameter surface:
%   res = mir_pls(X, Y, n_components)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

```matlab
mdl  = pls4all.fit("mir_pls", X, y, "NumComponents", 4);
yhat = predict(mdl, Xtest);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📜 **Paper-only** — no executable parity reference; the `pls4all` implementation is verified by a smoke fit only. Canonical citation: Sjöblom, J., Svensson, O., Josefson, M., Kullberg, H. & Wold, S. (1998). An evaluation of orthogonal signal correction applied to calibration transfer of near infrared spectra. Chemom. Intell. Lab. Syst. 44(1-2), 229-244. (Inverse-regression PLS variant; no widely installable port.)
:::

### Validation contract

Structural validation contract for `mir_pls`, exported by the declarative validation framework (`benchmarks/validation/registry/`) from `benchmarks.parity_timing.registry`. Regenerated by `python -m benchmarks.validation.export_registry --write`.

| Field | Value |
|------|-------|
| Reference kind | `paper_only` |
| Canonical reference role | — |
| Declared references | — |
| Comparator | `binding_parity` (binding-consistency gate) |
| Comparator metric | `max_abs_diff` |
| Reference tolerance | `5e-02` (relaxed) |
| Binding tolerance | `1e-06` |
| Prediction key | `predictions` |
| Required data flags | none |
| Paper-only citation | Sjöblom, J., Svensson, O., Josefson, M., Kullberg, H. & Wold, S. (1998). An evaluation of orthogonal signal correction applied to calibration transfer of near infrared spectra. Chemom. Intell. Lab. Syst. 44(1-2), 229-244. (Inverse-regression PLS variant; no widely installable port.) |

**Benchmark cell parameters**

| Parameter | Value |
|-----------|-------|
| `n_components` | `4` |
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

**Reference gate**: strict release — `rmse_rel ≤ 1e-03` is required; `rmse_rel ≤ 1e-06` is displayed as exact.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Divergence</th><th class="size-col" scope="col">100×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libp4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.00 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms ms-best">0.98 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.17 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">2.72 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">3.70 ms</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.58 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergence parity-not_available">—</td><td class="ms">1.96 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)