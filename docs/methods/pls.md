# `pls` — PLS regression (SIMPLS)

_Group_: **Core PLS** · _Registry tolerance_: `1e-08`

## Description

SIMPLS PLS regression baseline

> **Registry note** — Baseline SIMPLS cell. sklearn uses NIPALS and ikpls uses improved-kernel PLS, so exact bit parity is not expected; the row exists to anchor timing comparisons.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `X` | `—` | `required` | current public binding signature |
| `y` | `—` | `required` | current public binding signature |
| `n_components` | `int` | `2` | current public binding signature |
| `pls_components` | `—` | `None` | current public binding signature |
| `cv` | `int` | `5` | current public binding signature |
| `fold_ids` | `—` | `None` | current public binding signature |
| `center_x` | `bool \| None` | `None` | current public binding signature |
| `scale_x` | `bool \| None` | `None` | current public binding signature |
| `center_y` | `bool \| None` | `None` | current public binding signature |
| `scale_y` | `bool \| None` | `None` | current public binding signature |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` | current public binding signature |
| `cuda_pls_min_device_features` | `int \| None` | `None` | current public binding signature |
| `cuda_pls_many_batched` | `bool \| None` | `None` | current public binding signature |

## Explanations

### Bibliographic source

de Jong, S. (1993). *SIMPLS: an alternative approach to partial least squares regression*. Chemometrics and Intelligent Laboratory Systems 18(3), 251--263. DOI [10.1016/0169-7439(93)85002-X](https://doi.org/10.1016/0169-7439(93)85002-X).

### Mathematical principle

Partial Least Squares regression seeks a set of latent directions in the predictor space that maximise the *covariance* with the response, in contrast to PCA which maximises only the variance of $\mathbf{X}$.

Given centred $\mathbf{X} \in \mathbb{R}^{n\times p}$ and $\mathbf{Y} \in \mathbb{R}^{n\times q}$, the first PLS component is the unit-norm direction $\mathbf{w}_1$ maximising $\operatorname{Cov}(\mathbf{X}\mathbf{w}_1, \mathbf{Y})$. Closed form: $\mathbf{w}_1 \propto \mathbf{X}^{\top}\mathbf{Y}$ (or its dominant left singular vector when $q>1$). Subsequent components are extracted from the deflated residual matrix so the resulting scores $\mathbf{T} = \mathbf{X}\mathbf{W}$ are orthogonal.

**SIMPLS** (de Jong 1993) is algebraically equivalent to NIPALS but computes the loading weights directly from the cross-product $\mathbf{S} = \mathbf{X}^{\top}\mathbf{Y}$ without re-deflating $\mathbf{X}$ at each step. This avoids accumulating floating-point error from iterative deflation and runs in roughly half the time of NIPALS for the same number of components. SIMPLS is the variant exposed by MATLAB's `plsregress`.

Once $k$ latent scores have been extracted the regression coefficients are reconstructed as $\mathbf{B} = \mathbf{W}(\mathbf{P}^{\top}\mathbf{W})^{-1}\mathbf{Q}^{\top}$, where $\mathbf{P}, \mathbf{Q}$ are the X- and Y-loadings. Predictions on new $\mathbf{X}^{\star}$ follow $\hat{\mathbf{Y}} = \mathbf{X}^{\star}\mathbf{B} + \bar{\mathbf{y}}$. The choice of $k$ trades bias and variance: use cross-validated PRESS or the one-SE rule of Hastie et al. (2009) to select it.

### Appropriate uses

Linear multivariate calibration when predictors are numerous, collinear, and response-linked latent directions are useful.

### Limits and validation

Predictions remain linear in X; component count and preprocessing must be validated without leaking validation samples.

### Implementation

Dispatched through `Algorithm.PLS_REGRESSION` + `Solver.SIMPLS` in libn4m (the `n4m_model_fit` C entry point). The same `Model.fit` / `Model.predict` surface is used by every binding. NIPALS, SVD, power-iteration, randomised-SVD, orthogonal-scores, kernel and wide-kernel solver variants are all available — see the `Solver` enum.

### Sources and provenance

Current implementation: [cpp/src/core/model.cpp](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/model.cpp).


### API and bindings

**C ABI (ABI 2):** [`n4m_estimators_pls_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimators/regression.h#L290). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.estimators.regression.latent import pls
result = pls(X, y)
```

Source signature: `pls(X, y, *, n_components: int = 2, pls_components = None, cv: int = 5, fold_ids = None, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)` ([`n4m/_impl/native.py`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8569)).

**R (source-verified):** [`pls(formula, data, ncomp = 2L, algo = "pls_nipals", na.action = stats::na.omit, ...)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/sklearn.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`pls_fit(X, Y, n_components)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/pls_fit.m).

The source signature has additional required inputs, so no example call is fabricated.

**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.python_ikpls`** (python · ikpls) — `ikpls` MISSING · strict (rmse_rel ≤ 1e-08) — ikpls.numpy_ikpls.PLS algorithm 1.

- 📐 **`ref.python_scikit_learn`** (python · python) — `scikit-learn` 1.7.2 · strict (rmse_rel ≤ 1e-08) — sklearn.cross_decomposition.PLSRegression(scale=False).

- 📐 **`ref.r_mixomics`** (R · mixOmics) — `mixOmics` 6.26.0 · strict (rmse_rel ≤ 1e-08) — Bioconductor mixOmics::pls(mode='regression', scale=FALSE).

- 📐 **`ref.r_pls`** (R · r) — `pls` 2.8.5 · strict (rmse_rel ≤ 1e-08) — R pls::plsr(method='simpls', scale=FALSE).

:::

### Benchmarks

**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** section above for current entry points.

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented by-design selector/RNG/model, noncanonical API/facade convention, or secondary oracle &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-08`).

Rows tagged with **📐** are the canonical parity references for this method (declared in [`parity_timing.registry`](../benchmarks/methodology.md)). C++ and external rows show reference parity; archived language-harness rows show binding parity against the C++ backend. Hover the icon for role and tolerance band.

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">1.70 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.69 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.97 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">4.79 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">5.26 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">5.92 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">9.99 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls MISSING — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-cross_check">⇄ +9e-03</td><td class="ms">1.92 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.7.2 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.16 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-cross_check">⇄ +6e-16</td><td class="ms">9.72 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_pls</code></td><td class="parity parity-cross_check">⇄ +1e-14</td><td class="ms">8.01 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 3 threads
:sync: threads-3

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">1.79 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.76 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.95 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">4.43 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">5.76 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">6.29 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">10.3 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls MISSING — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-cross_check">⇄ +9e-03</td><td class="ms">1.90 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.7.2 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.16 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-cross_check">⇄ +6e-16</td><td class="ms">9.82 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_pls</code></td><td class="parity parity-cross_check">⇄ +1e-14</td><td class="ms">7.50 ms</td></tr>
</tbody>
</table>
</div>

:::

:::{tab-item} 10 threads
:sync: threads-10

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">200×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref 6e-16</td><td class="ms">1.79 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.python</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">1.78 ms<span class="medal" title="fastest">🏆</span></td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">1.93 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · archived pls4all benchmark</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">4.69 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.formula</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">5.49 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.mdatools</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">5.59 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.R.pls</code></td><td class="parity parity-exact">✓ 7e-15</td><td class="ms">10.4 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (ikpls): ikpls MISSING — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_ikpls</code></td><td class="parity parity-cross_check">⇄ +9e-03</td><td class="ms">2.04 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (python): scikit-learn 1.7.2 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.python_scikit_learn</code></td><td class="parity parity-ref-source">source</td><td class="ms">2.17 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="3" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (mixOmics): mixOmics 6.26.0 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_mixomics</code></td><td class="parity parity-cross_check">⇄ +6e-16</td><td class="ms">9.91 ms</td></tr>
<tr class="bk-row truth-source truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (r): pls 2.8.5 — strict (rmse_rel ≤ 1e-08)">📐</span><code>ref.r_pls</code></td><td class="parity parity-cross_check">⇄ +1e-14</td><td class="ms">8.15 ms</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)