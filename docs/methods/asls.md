# `asls` — AsLS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/asls.md`](../algorithms/asls.md)

## Description

Eilers & Boelens 2005 asymmetric least squares baseline estimation. For each row `y` of length `n`:

1. Initialize weights `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - Compute new weights: `new_w[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`, break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]` (baseline-subtracted spectrum).

From the `chemometrics4all.AsLS` Python wrapper docstring:

> Asymmetric Least Squares (Eilers & Boelens 2005).

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam` | `float` | `1000000.0` | Smoothness penalty; larger values produce a smoother baseline. |
| `p` | `float` | `0.01` | Asymmetry parameter in penalized least-squares baseline correction. |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Eilers, P. H. C. & Boelens, H. F. M. (2005). "Baseline Correction with Asymmetric Least Squares Smoothing." Leiden University Medical Centre Report.
- Internal parity fixture: `parity/python_generator/src/c4a_parity_pybaselines_ref/asls.py` (historically validated against `pybaselines==1.1.4`); current upstream `pybaselines` drift is tracked by the reference parity gate.

### Mathematical principle

`asls` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pentadiagonal symmetric LDLT solver in `core/common/banded_solver.{c,h}` (O(n) per iteration, vs pls4all's dense O(n²)).
- Per-row scratch buffers reused across all iterations to minimize allocator churn.
- Convergence uses `relative_difference(w, new_w)` matching pybaselines's `_safe_relative_difference` (L2 norms with `max(||w||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate (matches pybaselines behaviour).
- Parity tolerance vs internal parity fixture: `1e-7 abs / 1e-8 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out, double lam, double p, int32_t max_iter, double tol);
void c4a_pp_asls_destroy(c4a_pp_asls_handle_t* handle);
c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_asls` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.asls` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.AsLS` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `asls(X, lam = 1e6, p = 0.01, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.asls` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out, double lam, double p, int32_t max_iter, double tol);
void c4a_pp_asls_destroy(c4a_pp_asls_handle_t* handle);
c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.asls(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import AsLS

op = AsLS(lam=1000000.0, p=0.01, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- asls(X, lam = 1e5, p = 0.01, max_iter = 20L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.asls` · pybaselines 1.2.1
:::

### Validation contract

- Operation: `fit_transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `24×256` · seed `1234567893`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `pybaselines_asls` — pybaselines.whittaker.asls (pybaselines, 1.2.1)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pybaselines` | `pybaselines.asls` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.3e-10</td><td class="ms">0.246 ms</td><td class="ms ms-best">🏆 3.025 ms</td><td class="ms ms-best">🏆 15.090 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.3e-10</td><td class="ms ms-best">🏆 0.246 ms</td><td class="ms">3.096 ms</td><td class="ms">15.094 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.3e-10</td><td class="ms">0.252 ms</td><td class="ms">3.105 ms</td><td class="ms">15.512 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">4.3e-10</td><td class="ms">0.268 ms</td><td class="ms">3.344 ms</td><td class="ms">16.625 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.asls · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">9.412 ms</td><td class="ms">22.555 ms</td><td class="ms">80.488 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
