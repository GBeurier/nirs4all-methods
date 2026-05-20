# `aug_batch_effect` — Batch effect augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_batch_effect.md`](../algorithms/aug_batch_effect.md)

## Description

From the `n4m.BatchEffectAugmenter` Python wrapper docstring:

> Random offset, slope and gain batch effects.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `offset_std` | `float` | `0.0` |  |
| `slope_std` | `float` | `0.0` |  |
| `gain_std` | `float` | `0.0` |  |
| `variation_scope` | `int` | `0` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.BatchEffectAugmenter`.

### Mathematical principle

Simulates batch/session measurement drift via a wavelength-normalised
additive offset, a slope, and a multiplicative gain.

Let $\tilde\lambda_j$ be the centred and range-normalised wavelength axis:
$$\tilde\lambda_j = (\lambda_j - \bar\lambda) / (\lambda_{\max} - \lambda_{\min} + 10^{-10}).$$
If `wavelengths == NULL`, the integer index $0, 1, \dots, p-1$ is used in
its place.

Two modes:

**Per-sample** (`variation_scope = 0`):
$$o_i \sim \mathrm{Normal}(0, \sigma_o),\ m_i \sim \mathrm{Normal}(0, \sigma_m),\ g_i \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g_i + o_i + m_i \tilde\lambda_j$$

**Batch** (`variation_scope = 1`):
$$o \sim \mathrm{Normal}(0, \sigma_o),\ m \sim \mathrm{Normal}(0, \sigma_m),\ g \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g + o + m \tilde\lambda_j$$

In per-sample mode the reference draws the parameters as three separate
length-$n$ normal vectors (offsets, then slopes, then gains). The C engine
mirrors this draw order so the PCG64 state evolution matches NumPy.

### Implementation

When $\sigma_o = \sigma_m = \sigma_g = 0$ the augmenter is the identity
transform (gain ~ N(1, 0) = 1, offset/slope ~ N(0, 0) = 0). The parity
fixture exercises this case (`batch_effect_zero_batch`).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_batch_effect_apply( const n4m_aug_batch_effect_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_batch_effect_create( n4m_aug_batch_effect_handle_t** out, n4m_rng_pcg64_state_t* rng, double offset_std, double slope_std, double gain_std, int32_t variation_scope, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_batch_effect_destroy( n4m_aug_batch_effect_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_batch_effect` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_batch_effect` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.BatchEffectAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_batch_effect(X, wavelengths = NULL, offset_std = 0.02, slope_std = 0.01, gain_std = 0.03, variation_scope = 0L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.BatchEffectAugmenter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_batch_effect_apply( const n4m_aug_batch_effect_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_batch_effect_create( n4m_aug_batch_effect_handle_t** out, n4m_rng_pcg64_state_t* rng, double offset_std, double slope_std, double gain_std, int32_t variation_scope, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_batch_effect_destroy( n4m_aug_batch_effect_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_batch_effect(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import BatchEffectAugmenter

op = BatchEffectAugmenter(offset_std=0.0, slope_std=0.0, gain_std=0.0, variation_scope=0, wavelengths=None, rng=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_batch_effect(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.BatchEffectAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.BatchEffectAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.011 ms</td><td class="ms ms-best">🏆 0.022 ms</td><td class="ms ms-best">🏆 0.090 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.011 ms</td><td class="ms">0.024 ms</td><td class="ms">0.093 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.012 ms</td><td class="ms">0.025 ms</td><td class="ms">0.100 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.7e-16</td><td class="ms">0.050 ms</td><td class="ms">0.215 ms</td><td class="ms">1.141 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.BatchEffectAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.040 ms</td><td class="ms">0.143 ms</td><td class="ms">0.971 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
