# `aug_hetero_noise` — Heteroscedastic noise

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_hetero_noise.md`](../algorithms/aug_hetero_noise.md)

## Description

Adds signal-dependent Gaussian noise to spectra.

For each input cell:

1. `sigma[i, j] = noise_base + noise_signal_dep * |X[i, j]|`
2. Draw `noise = standard_normal(rows * cols)` (one bulk call, row-major).
3. `out[i, j] = X[i, j] + noise[i*cols + j] * sigma[i, j]`.

Mirrors `nirs4all.operators.augmentation.synthesis.HeteroscedasticNoiseAugmenter` with `variation_scope="sample"` (default).

From the `n4m.HeteroscedasticNoiseAugmenter` Python wrapper docstring:

> Noise whose standard deviation depends on signal magnitude.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `noise_base` | `float` | `0.001` |  |
| `noise_signal_dep` | `float` | `0.01` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/hetero_noise.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_hetero_noise` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_hetero_noise_apply( const n4m_aug_hetero_noise_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_hetero_noise_create( n4m_aug_hetero_noise_handle_t** out, n4m_rng_pcg64_state_t* rng, double noise_base, double noise_signal_dep);
void n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_hetero_noise` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_hetero_noise` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.HeteroscedasticNoiseAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_hetero_noise(X, noise_base = 0.001, noise_signal_dep = 0.01, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.HeteroscedasticNoiseAugmenter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_hetero_noise_apply( const n4m_aug_hetero_noise_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_hetero_noise_create( n4m_aug_hetero_noise_handle_t** out, n4m_rng_pcg64_state_t* rng, double noise_base, double noise_signal_dep);
void n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_hetero_noise(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import HeteroscedasticNoiseAugmenter

op = HeteroscedasticNoiseAugmenter(noise_base=0.001, noise_signal_dep=0.01, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_hetero_noise(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.HeteroscedasticNoiseAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.HeteroscedasticNoiseAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.027 ms</td><td class="ms">0.321 ms</td><td class="ms">1.763 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.026 ms</td><td class="ms">0.351 ms</td><td class="ms ms-best">🏆 1.758 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.028 ms</td><td class="ms ms-best">🏆 0.303 ms</td><td class="ms">1.785 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.042 ms</td><td class="ms">0.414 ms</td><td class="ms">2.750 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.HeteroscedasticNoiseAugmenter · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.062 ms</td><td class="ms">0.563 ms</td><td class="ms">3.167 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
