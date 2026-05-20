# `aug_emsc_distort` — EMSC distortion augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_emsc_distort.md`](../algorithms/aug_emsc_distort.md)

## Description

From the `n4m.EMSCDistortionAugmenter` Python wrapper docstring:

> Random EMSC-like multiplicative, additive and polynomial distortion.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mult_low` | `float` | `0.9` |  |
| `mult_high` | `float` | `1.1` |  |
| `add_low` | `float` | `-0.05` |  |
| `add_high` | `float` | `0.05` |  |
| `polynomial_order` | `int` | `2` |  |
| `polynomial_strength` | `float` | `0.02` |  |
| `correlation` | `float` | `0.3` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.scattering.EMSCDistortionAugmenter`,
Martens et al. (2003).

### Mathematical principle

Simulates EMSC-style spectral distortion that EMSC is later designed to
correct:

$$X^{\mathrm{aug}}_{i,j} = a_i + b_i \cdot X_{i,j} + \sum_{k=1}^{d} c_{i,k} \cdot \tilde\lambda_j^k$$

where $\tilde\lambda_j = 2(\lambda_j - \lambda_{\min}) / (\lambda_{\max} - \lambda_{\min}) - 1$
is the normalised wavelength in $[-1, 1]$.

Per-sample parameter draws:

- $b_i \sim \mathrm{Normal}(\bar b, \sigma_b)$ where $\bar b = (b_{\mathrm{lo}} + b_{\mathrm{hi}})/2$ and $\sigma_b = (b_{\mathrm{hi}} - b_{\mathrm{lo}})/4$, then clipped to $[b_{\mathrm{lo}}, b_{\mathrm{hi}}]$.
- $\Delta_b = (b_i - 1) / \sigma_b$.
- $a_i \sim \mathrm{Normal}(\bar a - \rho \sigma_a \Delta_b, \sigma_a \sqrt{1 - \rho^2})$, then clipped to $[a_{\mathrm{lo}}, a_{\mathrm{hi}}]$.
- $c_{i,k} \sim \mathrm{Normal}(0, \mathrm{strength} / \sqrt{k})$ for $k = 1, \dots, d$.

The correlation parameter $\rho$ couples the additive and multiplicative
terms (positive $\rho$ → when $b$ is high, $a$ tends to be low).

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_emsc_distort_apply( const n4m_aug_emsc_distort_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_emsc_distort_create( n4m_aug_emsc_distort_handle_t** out, n4m_rng_pcg64_state_t* rng, double mult_low, double mult_high, double add_low, double add_high, int32_t polynomial_order, double polynomial_strength, double correlation, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_emsc_distort_destroy( n4m_aug_emsc_distort_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_emsc_distort` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_emsc_distort` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.EMSCDistortionAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_emsc_distort(X, wavelengths = NULL, mult_low = 0.9, mult_high = 1.1, add_low = -0.05, add_high = 0.05, polynomial_order = 2L, polynomial_strength = 0.02, correlation = 0.3, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.EMSCDistortionAugmenter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_emsc_distort_apply( const n4m_aug_emsc_distort_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_emsc_distort_create( n4m_aug_emsc_distort_handle_t** out, n4m_rng_pcg64_state_t* rng, double mult_low, double mult_high, double add_low, double add_high, int32_t polynomial_order, double polynomial_strength, double correlation, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_emsc_distort_destroy( n4m_aug_emsc_distort_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_emsc_distort(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import EMSCDistortionAugmenter

op = EMSCDistortionAugmenter(mult_low=0.9, mult_high=1.1, add_low=-0.05, add_high=0.05, polynomial_order=2, polynomial_strength=0.02)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_emsc_distort(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.EMSCDistortionAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.EMSCDistortionAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.017 ms</td><td class="ms ms-best">🏆 0.059 ms</td><td class="ms ms-best">🏆 0.239 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.018 ms</td><td class="ms">0.061 ms</td><td class="ms">0.242 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.019 ms</td><td class="ms">0.059 ms</td><td class="ms">0.252 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.044 ms</td><td class="ms">0.250 ms</td><td class="ms">1.289 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.EMSCDistortionAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.963 ms</td><td class="ms">1.031 ms</td><td class="ms">1.839 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
