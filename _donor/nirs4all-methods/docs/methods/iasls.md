# `iasls` — IAsLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=5e-6` · _Source_: [`docs/algorithms/iasls.md`](../algorithms/iasls.md)

## Description

He et al. 2014 IAsLS baseline, aligned with
`pybaselines.whittaker.Baseline.iasls` for `diff_order = 2`. For each row `y`
of length `n`:

1. Fit polynomial of degree `polyorder` to `(positions, y)` -> initial baseline `z0`.
2. Initialise weights `w[i] = p` if `y[i] > z0[i]` else `1 - p`.
3. Build the fixed first-derivative residual term
   `d1_y = lam_1 * (D1^T D1) y`.
4. Repeat up to `max_iter`:
   - Solve `(diag(w^2) + lam D2^T D2 + lam_1 D1^T D1) z =
     w^2 * y + d1_y` (pentadiagonal LDLT).
   - `w_new[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `rel_l2_diff(w, w_new) < tol`, break.
   - `w = w_new`.
5. Output: `out = y - z`.

From the `n4m.IAsLS` Python wrapper docstring:

> Improved asymmetric least-squares baseline correction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `1000000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `p` | `float` | `0.01` | Asymmetry parameter in penalized least-squares baseline correction. |
| `lam_1` | `float` | `0.0001` |  |
| `polyorder` | `int` | `2` | Polynomial order. |
| `diff_order` | `int` | `2` |  |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- He, S., Zhang, X. et al. (2014). "Baseline correction for Raman spectra using an improved asymmetric least squares method." Analytical Methods 6, 4402-4407.
- External reference snapshot: `pybaselines.iasls` output stored under
  `benchmarks/reference_snapshots/cross_binding/iasls/`.

### Mathematical principle

`iasls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Shared 1st/2nd-difference penalties + in-place pentadiagonal LDLT from `core/common/banded_solver.h`. The fixed penalty is built once per matrix; the LDLT factor reuses row-scope `L` / `D` scratch buffers across iterations.
- Polynomial pre-fit uses the shared Householder QR (`core/common/linalg.h`), factored once per matrix and replayed per row.
- Convergence uses `n4m_relative_l2_diff` (shared with AsLS / ArPLS).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs pybaselines snapshot: `5e-6 abs / 1e-5 rel`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_pp_iasls_create(n4m_pp_iasls_handle_t** out, double lam, double p, int32_t polyorder, int32_t max_iter, double tol);
n4m_status_t n4m_pp_iasls_create_ex(n4m_pp_iasls_handle_t** out, double lam, double p, double lam_1, int32_t polyorder, int32_t diff_order, int32_t max_iter, double tol);
void n4m_pp_iasls_destroy(n4m_pp_iasls_handle_t* handle);
n4m_status_t n4m_pp_iasls_transform(const n4m_pp_iasls_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_pp_iasls` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.iasls` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.IAsLS` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `iasls(X, lam = 1e6, p = 0.01, lam_1 = 1e-4, polyorder = 2L, diff_order = 2L, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.iasls` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_pp_iasls_create(n4m_pp_iasls_handle_t** out, double lam, double p, int32_t polyorder, int32_t max_iter, double tol);
n4m_status_t n4m_pp_iasls_create_ex(n4m_pp_iasls_handle_t** out, double lam, double p, double lam_1, int32_t polyorder, int32_t diff_order, int32_t max_iter, double tol);
void n4m_pp_iasls_destroy(n4m_pp_iasls_handle_t* handle);
n4m_status_t n4m_pp_iasls_transform(const n4m_pp_iasls_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.iasls(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import IAsLS

op = IAsLS(lam=1000000.0, p=0.01, lam_1=0.0001, polyorder=2, diff_order=2, max_iter=50)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- iasls(X, lam = 1e5, p = 0.01, lam_1 = 1e-4, polyorder = 2L, diff_order = 2L, max_iter = 20L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.iasls` · pybaselines 1.2.1
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `iasls_allclose` · tolerance: `rtol=1e-05`, `atol=5e-06` · quality: **relaxed-atol**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Used by iasls only.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pybaselines` | `pybaselines.iasls` | Python / parity | `iasls_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-08</td><td class="ms">0.398 ms</td><td class="ms">5.515 ms</td><td class="ms">32.385 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-08</td><td class="ms ms-best">🏆 0.396 ms</td><td class="ms ms-best">🏆 5.454 ms</td><td class="ms ms-best">🏆 32.339 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-08</td><td class="ms">0.414 ms</td><td class="ms">5.591 ms</td><td class="ms">32.361 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.0e-08</td><td class="ms">0.438 ms</td><td class="ms">5.812 ms</td><td class="ms">32.750 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.iasls · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">13.911 ms</td><td class="ms">29.968 ms</td><td class="ms">102.126 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
