# `derivate` — Derivate

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/derivate.md`](../algorithms/derivate.md)

## Description

From the `chemometrics4all.Derivate` Python wrapper docstring:

> Finite-difference derivative along the wavelength axis.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `order` | `int` | `1` |  |
| `delta` | `float` | `1.0` | Sample spacing along the wavelength axis. |

## Explanations

### Bibliographic source

Savitzky, A., Golay, M. J. E. (1964). "Smoothing and Differentiation of Data
by Simplified Least Squares Procedures." *Analytical Chemistry* 36 (8),
1627-1639 — for the broader family. The current implementation is the bare
finite-difference variant; Savitzky-Golay arrives in Phase 4.

### Mathematical principle

Finite-difference derivative of order $k$ along axis 1 (the wavelength
axis), normalised by the wavelength spacing $\delta$:

$$
X' = \frac{\mathrm{np.diff}(X,\; n=k,\; \mathrm{axis}=1)}{\delta^k}.
$$

`np.diff` applies first differences along the chosen axis; `n=k` repeats
the operation $k$ times, shortening the column count by one each pass.
Output shape:

$$
X' \in \mathbb{R}^{n \times (p - k)}.
$$

For $k = 1$, $X'_{i,j} = (X_{i, j+1} - X_{i, j}) / \delta$.
For $k = 2$, $X''_{i,j} = (X_{i, j+2} - 2 X_{i, j+1} + X_{i, j}) / \delta^2$.

Implementation: a single scratch buffer of length $p$ holds the row across
the $k$ passes. Each pass scans left-to-right and overwrites entries in
place — `scratch[j] := scratch[j+1] - scratch[j]`. After $k$ passes the
first $p - k$ slots hold the $k$-th differences, and we divide each by the
precomputed scalar $\delta^k$.

### Implementation

Mathematically stateless, but exposes the stateful contract
(`_create / _fit / _transform / _destroy`) to fit uniformly with MSC,
EMSC and Baseline. `_fit` records the input column count and marks the
state fitted; calling `_transform` before `_fit` returns
`C4A_ERR_NOT_FITTED`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_derivate_create(c4a_pp_derivate_handle_t** out, int32_t order, double delta);
void c4a_pp_derivate_destroy(c4a_pp_derivate_handle_t* handle);
c4a_status_t c4a_pp_derivate_fit(c4a_pp_derivate_handle_t* handle, c4a_matrix_view_t X);
int64_t c4a_pp_derivate_output_cols(int32_t order, int64_t input_cols);
c4a_status_t c4a_pp_derivate_transform( const c4a_pp_derivate_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_derivate` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.derivate` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.Derivate` | Python | scikit-learn-compatible estimator backed by ctypes. |
| ref.numpy | `numpy.diff(axis=1)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.Derivate(same-width)` | Python | context only; nirs4all pads/interpolates back to the input width; c4a exposes the strict finite-difference width p-1 and gates NumPy |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_derivate_create(c4a_pp_derivate_handle_t** out, int32_t order, double delta);
void c4a_pp_derivate_destroy(c4a_pp_derivate_handle_t* handle);
c4a_status_t c4a_pp_derivate_fit(c4a_pp_derivate_handle_t* handle, c4a_matrix_view_t X);
int64_t c4a_pp_derivate_output_cols(int32_t order, int64_t input_cols);
c4a_status_t c4a_pp_derivate_transform( const c4a_pp_derivate_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.derivate(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import Derivate

op = Derivate(order=1, delta=1.0)
Xt = op.fit_transform(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.numpy`** (Python · canonical) — `numpy.diff(axis=1)` · numpy 2.3.5
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.Derivate(same-width)` · nirs4all@cd731a23+dirty — nirs4all pads/interpolates back to the input width; c4a exposes the strict finite-difference width p-1 and gates NumPy
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.numpy` | `numpy.diff(axis=1)` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.Derivate(same-width)` | Python / context | `default_allclose` | nirs4all pads/interpolates back to the input width; c4a exposes the strict finite-difference width p-1 and gates NumPy |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.011 ms</td><td class="ms ms-best">🏆 0.038 ms</td><td class="ms ms-best">🏆 0.183 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.011 ms</td><td class="ms">0.040 ms</td><td class="ms">0.188 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.015 ms</td><td class="ms">0.040 ms</td><td class="ms">0.183 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Derivate(same-width) · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="no divergence recorded">—</td><td class="ms">0.015 ms</td><td class="ms">0.077 ms</td><td class="ms">—</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy.diff(axis=1) · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.008 ms</td><td class="ms">0.062 ms</td><td class="ms">0.282 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
