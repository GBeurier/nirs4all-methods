# `rosa` — ROSA (Response-Oriented Sequential Alternation)

_Group_: **Multi-block / cross-modal** · _Registry tolerance_: `1e-06`

## Description

ROSA — Response-Oriented Sequential Alternation (§19)

From the `pls4all.sklearn.ROSARegression` docstring:

> Response-Oriented Sequential Alternation (Liland & Næs 2016).

> **Registry note** — Canonical multiblock ROSA (Liland, Næs & Indahl 2016). Both references implement the canonical formulation; pls4all matches them bit-for-bit (max_abs < 1e-6) for single-target Y. The R reference centers multi-target Y incorrectly (recycles `colMeans(y)` rather than broadcasting), so multi-target parity is intentionally evaluated against the NumPy reference.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_components` | `int` | `2` | Number of latent components extracted (k). |
| `block_sizes` | `—` | `None` | Sequence of contiguous block widths defining the X-block partition (columns of X). |
| `n_targets` | `int` | `1` | registry benchmark cell value |
| `n_blocks` | `int` | `3` | registry benchmark cell value |

## Explanations

### Bibliographic source

Liland, K. H. & Næs, T. (2016). *Response-oriented sequential alternation: a fast multiblock regression algorithm*. Journal of Chemometrics 30(11), 651–662.

### Mathematical principle

ROSA is a forward-greedy multi-block PLS: at each component extraction step, it tries one new component from **every block** and keeps the block whose new component most reduces residual variance in $\mathbf{y}$. The component sequence is therefore data-driven rather than pre-specified by the user.

This auto-attribution makes ROSA a strong default when the analyst has no prior on which block matters most: the block budget is allocated dynamically. ROSA is also much faster than SO-PLS for large block counts because it adds one component per iteration rather than refitting an inner PLS per block.

Output includes the **block-attribution vector** — the sequence of block indices selected — which is the main interpretive artefact: it tells you which block contributed which component, in order.

### Implementation

`n4m_rosa_fit`. Reference: CRAN `multiblock 0.8.10`.

MATLAB header (`bindings/matlab/+pls4all/rosa.m`):

```text
pls4all.rosa  Response-Oriented Sequential Alternation (Liland & Næs 2016).
```

### Usage

Every pls4all binding tab dispatches into the same C kernel; the external libraries listed at the bottom of the page are the parity references registered in `benchmarks.parity_timing.registry`. Switch tabs to read the same fit in your language. The R package now ships drop-in-compatible facades for the CRAN `pls` package (`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` matrix idiom — those tabs appear only on the methods that have a meaningful equivalence.

**pls4all bindings**

::::{tab-set}
:class: pls4all-bindings

:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
/* C ABI — libn4m */
n4m_context_t* ctx = n4m_context_create();
n4m_config_t*  cfg = n4m_config_create();
n4m_method_result_t* res = NULL;
n4m_rosa_fit(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
/* … read coefficients / mask / scores via */
/* n4m_method_result_get_double_matrix / vector / scalar … */
n4m_method_result_destroy(res);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

:::

:::{tab-item} Python · pls4all (raw)
:sync: python-raw
:class-label: lang-python

```python
import pls4all
from pls4all._methods import rosa_fit
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    res = rosa_fit(ctx, cfg, X, y, n_components=4)
# then: res.matrix("predictions"), res.matrix("coefficients"),
# res.vector("mask"), res.scalar("intercept"), …
```

:::

:::{tab-item} Python · pls4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from pls4all.sklearn import ROSARegression
mdl = ROSARegression(n_components=2, block_sizes=None)
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
res <- pls4all_method("rosa", X, y,
                      n_components = 4L, params = list(n_targets = 1L, n_blocks = 3L))
# res is a named list with MethodResult arrays/scalars.
# selected_indices / top_k_intervals are 1-based.
```

:::

:::{tab-item} MATLAB · pls4all (MEX)
:sync: matlab-mex
:class-label: lang-matlab

```matlab
res = pls4all.rosa(X, y, 4);
% see header of bindings/matlab/+pls4all/rosa.m for full
% parameter surface:
%   res = rosa(X, Y, n_components, block_sizes)
yhat = predict(res, Xtest);
```

:::

:::{tab-item} MATLAB · pls4all (classdef)
:sync: matlab-classdef
:class-label: lang-matlab

_No idiomatic classdef wrapper — invoke `pls4all.fit("rosa", X, y, …)` directly from the unified MEX factory._

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_numpy`** (python · python) — `numpy` 2.3.5 · strict (rmse_rel ≤ 1e-06) — Canonical multiblock ROSA in NumPy (Liland, Næs & Indahl 2016). Reproduces R `multiblock::rosa(canonical=TRUE)` exactly for single-target Y; for q >= 2 it uses proper column-mean centering and diverges from R `multiblock` because that package recycles `colMeans(y)` incorrectly. pls4all matches this NumPy reference for q == 1 at IEEE round-off.
- 📐 **`ref.r_multiblock`** (R · r) — `multiblock` 0.8.10 · strict (rmse_rel ≤ 1e-06) — R `multiblock::rosa 0.8.10` (Liland, Næs & Indahl 2016). ROSA's greedy block-selection-per-component may diverge from pls4all's ordering — tolerance widened.
:::

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-06`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; pls4all language bindings show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-drift">≈ +7e-16</td><td class="ms">3.13 ms</td><td class="ms">2.12 ms</td><td class="ms ms-best">12.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">58.8 ms</td><td class="ms ms-best">1.23 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">2.90 ms</td><td class="ms">6.79 ms</td><td class="ms">60.4 ms</td><td class="ms ms-best">306.6 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">29.5 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">313.9 ms</td><td class="ms">1.8 s</td><td class="ms">130.3 ms</td><td class="ms ms-best">1.4 s<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-drift">≈ +7e-16</td><td class="ms">2.59 ms</td><td class="ms">1.29 ms</td><td class="ms">13.2 ms</td><td class="ms">59.0 ms</td><td class="ms">1.24 ms</td><td class="ms ms-best">2.61 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">5.64 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">60.2 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">311.5 ms</td><td class="ms">30.5 ms</td><td class="ms ms-best">293.8 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.8 s<span class="medal" title="fastest">🏆</span></td><td class="ms">130.2 ms</td><td class="ms">1.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-drift">≈ +7e-16</td><td class="ms ms-best">2.58 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-best">1.17 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">12.4 ms</td><td class="ms">57.3 ms</td><td class="ms">1.24 ms</td><td class="ms">2.70 ms</td><td class="ms">6.33 ms</td><td class="ms">60.7 ms</td><td class="ms">315.4 ms</td><td class="ms">31.0 ms</td><td class="ms">305.8 ms</td><td class="ms">1.8 s</td><td class="ms ms-best">125.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.4 s</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-drift">≈ +7e-16</td><td class="ms">2.60 ms</td><td class="ms">1.43 ms</td><td class="ms">13.4 ms</td><td class="ms ms-best">55.4 ms<span class="medal" title="fastest">🏆</span></td><td class="ms">1.26 ms</td><td class="ms">2.90 ms</td><td class="ms">6.90 ms</td><td class="ms">60.8 ms</td><td class="ms">309.1 ms</td><td class="ms">30.0 ms</td><td class="ms">309.2 ms</td><td class="ms">1.8 s</td><td class="ms">130.0 ms</td><td class="ms">1.4 s</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">2.62 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.30 ms</td><td class="ms">2.89 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">3.28 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.59 ms</td><td class="ms">6.17 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">12.7 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.34 ms</td><td class="ms">11.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">21.1 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">6.92 ms</td><td class="ms">11.8 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">23.9 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">6.26 ms</td><td class="ms">12.1 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">22.4 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.23 ms</td><td class="ms">14.2 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms">5.35 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.15 ms</td><td class="ms">4.02 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms">6.08 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.65 ms</td><td class="ms">5.19 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy 2.3.5 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">3.04 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.58 ms</td><td class="ms">3.47 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): multiblock 0.8.10 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_multiblock</code></td><td class="parity parity-ref-strict">✓ ref 2e-15</td><td class="ms">1.4 s</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.3 s</td><td class="ms">1.4 s</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.02 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.83 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.23 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.25 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">1.22 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.49 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.70 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.04 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">5.00 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">4.64 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.57 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.60 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy 2.3.5 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.60 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): multiblock 0.8.10 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_multiblock</code></td><td class="parity parity-ref-strict">✓ ref 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.3 s</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">100×50 (ms)</th><th class="size-col" scope="col">100×500 (ms)</th><th class="size-col" scope="col">100×2500 (ms)</th><th class="size-col" scope="col">200×30 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th><th class="size-col" scope="col">500×50 (ms)</th><th class="size-col" scope="col">500×500 (ms)</th><th class="size-col" scope="col">500×2500 (ms)</th><th class="size-col" scope="col">2500×50 (ms)</th><th class="size-col" scope="col">2500×500 (ms)</th><th class="size-col" scope="col">2500×2500 (ms)</th><th class="size-col" scope="col">10000×50 (ms)</th><th class="size-col" scope="col">10000×500 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-best">1.14 ms<span class="medal" title="fastest">🏆</span></td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.17 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.omp</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.19 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.ref</code></td><td class="parity parity-ref-strict">✓ ref 2e-16</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.16 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.81 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.44 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.33 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.84 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">3.96 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-matlab"><tr class="lang-band-row" data-lang="matlab"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>MATLAB · pls4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.90 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.matlab.classdef</code></td><td class="parity parity-divergent">✗ +9e+00</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">2.27 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): numpy 2.3.5 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.36 ms</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="16" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): multiblock 0.8.10 — strict (rmse_rel ≤ 1e-06)">📐</span><code>ref.r_multiblock</code></td><td class="parity parity-ref-strict">✓ ref 1e-15</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms">1.1 s</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td><td class="ms ms-empty">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)