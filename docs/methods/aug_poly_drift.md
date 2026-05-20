# `aug_poly_drift` — Polynomial baseline drift

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_poly_drift.md`](../algorithms/aug_poly_drift.md)

## Description

Adds a per-sample polynomial baseline drift of configurable degree to spectra.

Algorithm (RNG draws are **order-major** to match nirs4all):

1. Build `lambdas = linspace(-1, 1, n_features)` (NumPy 1.26.4 conventions; for `n_features == 1` returns `[-1.0]`).
2. For each order `k` in `[0, degree]`, draw `n_samples` uniforms and form
   `coeffs[k, i] = coeff_min[k] + (coeff_max[k] - coeff_min[k]) * u_ik`.
3. `out[i, j] = X[i, j] + sum_k coeffs[k, i] * lambdas[j]^k`.

The polynomial accumulation uses the `lj_pow *= lj` recurrence rather than `np.polyval` so the multiplication order is bit-identical between the C reference and the Python ref.

From the `chemometrics4all.PolynomialBaselineDrift` Python wrapper docstring:

> Add random polynomial baseline drift.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `degree` | `int` | `2` |  |
| `coeff_min` | `object` | `None` |  |
| `coeff_max` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- Internal parity fixture: `parity/python_generator/src/c4a_parity_augmenters_ref/poly_drift.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_poly_drift` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_poly_drift_apply( const c4a_aug_poly_drift_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_poly_drift_create( c4a_aug_poly_drift_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t degree, const double* coeff_min, const double* coeff_max);
void c4a_aug_poly_drift_destroy(c4a_aug_poly_drift_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_poly_drift` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_poly_drift` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.PolynomialBaselineDrift` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_poly_drift(X, degree = 2L, coeff_min = -0.01, coeff_max = 0.01, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.PolynomialBaselineDrift` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_poly_drift_apply( const c4a_aug_poly_drift_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_poly_drift_create( c4a_aug_poly_drift_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t degree, const double* coeff_min, const double* coeff_max);
void c4a_aug_poly_drift_destroy(c4a_aug_poly_drift_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_poly_drift(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import PolynomialBaselineDrift

op = PolynomialBaselineDrift(degree=2, coeff_min=None, coeff_max=None, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_poly_drift(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.PolynomialBaselineDrift` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.PolynomialBaselineDrift` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.019 ms</td><td class="ms ms-best">🏆 0.081 ms</td><td class="ms">0.360 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.020 ms</td><td class="ms">0.084 ms</td><td class="ms ms-best">🏆 0.347 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.021 ms</td><td class="ms">0.082 ms</td><td class="ms">0.349 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.048 ms</td><td class="ms">0.201 ms</td><td class="ms">1.281 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.PolynomialBaselineDrift · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.059 ms</td><td class="ms">0.221 ms</td><td class="ms">1.216 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
