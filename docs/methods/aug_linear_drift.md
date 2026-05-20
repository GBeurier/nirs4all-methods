# `aug_linear_drift` — Linear baseline drift

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_linear_drift.md`](../algorithms/aug_linear_drift.md)

## Description

Adds a per-sample affine baseline drift (constant + linear-in-wavelength) to spectra.

Algorithm:

1. **Offsets** (`n_samples` uniforms in `[offset_min, offset_max)`):
   `offsets[i] = offset_min + (offset_max - offset_min) * u_off_i`.
2. **Slopes**  (`n_samples` uniforms in `[slope_min, slope_max)`):
   `slopes[i]  = slope_min  + (slope_max  - slope_min)  * u_slope_i`.
3. **Apply**: with `lambdas = arange(n_features)` centered at the mean,
   `out[i, j] = X[i, j] + offsets[i] + slopes[i] * (j - (n_features - 1)/2)`.

Mirrors `nirs4all.operators.augmentation.spectral.LinearBaselineDrift` with the implicit-wavelengths branch (no wavelength array supplied — the implicit branch uses `arange(n_features)`).

From the `n4m.LinearBaselineDrift` Python wrapper docstring:

> Add random offset and linear slope drift.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `offset_min` | `float` | `-0.05` |  |
| `offset_max` | `float` | `0.05` |  |
| `slope_min` | `float` | `-0.01` |  |
| `slope_max` | `float` | `0.01` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/linear_drift.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_linear_drift` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_linear_drift_apply( const n4m_aug_linear_drift_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_linear_drift_create( n4m_aug_linear_drift_handle_t** out, n4m_rng_pcg64_state_t* rng, double offset_min, double offset_max, double slope_min, double slope_max);
void n4m_aug_linear_drift_destroy(n4m_aug_linear_drift_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_linear_drift` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_linear_drift` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.LinearBaselineDrift` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_linear_drift(X, offset_min = -0.05, offset_max = 0.05, slope_min = -0.01, slope_max = 0.01, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.LinearBaselineDrift` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_linear_drift_apply( const n4m_aug_linear_drift_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_linear_drift_create( n4m_aug_linear_drift_handle_t** out, n4m_rng_pcg64_state_t* rng, double offset_min, double offset_max, double slope_min, double slope_max);
void n4m_aug_linear_drift_destroy(n4m_aug_linear_drift_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_linear_drift(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import LinearBaselineDrift

op = LinearBaselineDrift(offset_min=-0.05, offset_max=0.05, slope_min=-0.01, slope_max=0.01, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_linear_drift(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.LinearBaselineDrift` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.LinearBaselineDrift` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-15</td><td class="ms ms-best">🏆 0.010 ms</td><td class="ms">0.032 ms</td><td class="ms">0.129 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-15</td><td class="ms">0.011 ms</td><td class="ms">0.031 ms</td><td class="ms ms-best">🏆 0.127 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-15</td><td class="ms">0.011 ms</td><td class="ms ms-best">🏆 0.031 ms</td><td class="ms">0.140 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.2e-14</td><td class="ms">0.023 ms</td><td class="ms">0.209 ms</td><td class="ms">1.078 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.LinearBaselineDrift · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.038 ms</td><td class="ms">0.137 ms</td><td class="ms">0.532 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
