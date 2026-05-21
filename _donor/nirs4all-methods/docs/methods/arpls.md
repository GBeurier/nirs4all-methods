# `arpls` — ArPLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/arpls.md`](../algorithms/arpls.md)

## Description

Baek et al. 2015. Same skeleton as AsLS / AirPLS but with a logistic weight update centred on the negative-residual statistics.

For each row `y`:

1. Initialize `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `d = y - z`; `d_neg = d[d < 0]` (negative residuals).
   - `μ⁻ = mean(d_neg)`, `σ⁻ = std(d_neg, ddof=1)` (sample std; clamped to `DBL_MIN`).
   - `new_w[i] = 1 / (1 + exp(2 * (d[i] - (2σ⁻ - μ⁻)) / σ⁻))`  (numerically stable two-branch logistic).
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`: break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]`.

From the `n4m.ArPLS` Python wrapper docstring:

> Asymmetrically reweighted penalized least squares.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `100000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Baek, S.-J.; Park, A.; Ahn, Y.-J.; Choo, J. (2015). "Baseline Correction Using Asymmetrically Reweighted Penalized Least Squares Smoothing." *Analyst*, 140 (1), 250–257.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/arpls.py`.

### Mathematical principle

`arpls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal symmetric LDLT solver.
- Sample std uses `ddof=1` (matches `numpy.std(d_neg, ddof=1)`).
- Std clamped to `DBL_MIN` to avoid divide-by-zero on degenerate cases (all-zero residuals).
- Logistic stabilised via two-branch computation: for `arg >= 0` use `exp(-arg) / (1 + exp(-arg))`; for `arg < 0` use `1 / (1 + exp(arg))`. Avoids overflow for `|arg| >> 700`.
- Parity tolerance vs internal parity fixture: `1e-7 abs / 1e-8 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_arpls_create(n4m_pp_arpls_handle_t** out, double lam, int32_t max_iter, double tol);
void n4m_pp_arpls_destroy(n4m_pp_arpls_handle_t* handle);
n4m_status_t n4m_pp_arpls_transform(const n4m_pp_arpls_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_arpls` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.arpls` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.ArPLS` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `arpls(X, lam = 1e5, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.arpls` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_arpls_create(n4m_pp_arpls_handle_t** out, double lam, int32_t max_iter, double tol);
void n4m_pp_arpls_destroy(n4m_pp_arpls_handle_t* handle);
n4m_status_t n4m_pp_arpls_transform(const n4m_pp_arpls_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.arpls(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import ArPLS

op = ArPLS(lam=100000.0, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- arpls(X, lam = 1e5, max_iter = 20L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.arpls` · pybaselines 1.2.1
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `24×256` · seed `1234567895`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `pybaselines_arpls` — pybaselines.whittaker.arpls (pybaselines, 1.2.1)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pybaselines` | `pybaselines.arpls` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.6e-10</td><td class="ms ms-best">🏆 1.459 ms</td><td class="ms">8.459 ms</td><td class="ms">28.665 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.6e-10</td><td class="ms">1.514 ms</td><td class="ms ms-best">🏆 8.452 ms</td><td class="ms ms-best">🏆 28.509 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.6e-10</td><td class="ms">1.468 ms</td><td class="ms">8.860 ms</td><td class="ms">29.591 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.6e-10</td><td class="ms">1.500 ms</td><td class="ms">8.938 ms</td><td class="ms">29.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.arpls · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">46.777 ms</td><td class="ms">43.913 ms</td><td class="ms">74.207 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
