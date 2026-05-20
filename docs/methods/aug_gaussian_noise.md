# `aug_gaussian_noise` — Gaussian additive noise

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_gaussian_noise.md`](../algorithms/aug_gaussian_noise.md)

## Description

Adds per-sample std-scaled Gaussian noise to spectra.

For each input row `X[i, :]`:

1. Compute the row's biased standard deviation `stds[i] = std(X[i, :])` (ddof=0).
2. Draw a single bulk `noise = standard_normal(rows * cols)` from the borrowed PCG64 handle.
3. `out[i, j] = X[i, j] + noise[i*cols + j] * stds[i] * sigma`.

The single-call bulk draw fixes the RNG stream order, so the C reference matches the internal parity fixture at 1e-15 abs.

From the `n4m.GaussianAdditiveNoise` Python wrapper docstring:

> Add IID Gaussian noise to each element of ``X``.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `sigma` | `float` | `0.01` | Gaussian standard deviation or kernel scale, depending on method. |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/gaussian_noise.py` (validated against `nirs4all.operators.augmentation.spectral.GaussianAdditiveNoise` with `variation_scope="sample"`).
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_gaussian_noise` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

The augmenter does not own the RNG. Re-seed the RNG with `n4m_rng_pcg64_set_seed(rng, seed)` before each `_apply` to lock the output. Tested in `cpp/tests/test_augmenters_noise_drift.cpp::aug_gaussian_noise_*`.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_gaussian_noise_apply( const n4m_aug_gaussian_noise_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_gaussian_noise_create( n4m_aug_gaussian_noise_handle_t** out, n4m_rng_pcg64_state_t* rng, double sigma);
void n4m_aug_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_gaussian_noise` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_gaussian_noise` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.GaussianAdditiveNoise` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_gaussian_noise(X, sigma = 0.01, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.GaussianAdditiveNoise` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_gaussian_noise_apply( const n4m_aug_gaussian_noise_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_gaussian_noise_create( n4m_aug_gaussian_noise_handle_t** out, n4m_rng_pcg64_state_t* rng, double sigma);
void n4m_aug_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_gaussian_noise(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import GaussianAdditiveNoise

op = GaussianAdditiveNoise(sigma=0.01, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_gaussian_noise(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.GaussianAdditiveNoise` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.GaussianAdditiveNoise` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.032 ms</td><td class="ms ms-best">🏆 0.350 ms</td><td class="ms">1.907 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.029 ms</td><td class="ms">0.351 ms</td><td class="ms">1.854 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.029 ms</td><td class="ms">0.351 ms</td><td class="ms ms-best">🏆 1.849 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.044 ms</td><td class="ms">0.465 ms</td><td class="ms">3.125 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.GaussianAdditiveNoise · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.082 ms</td><td class="ms">0.597 ms</td><td class="ms">3.078 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
