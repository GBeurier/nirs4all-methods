# `aug_particle_size` — Particle size augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_particle_size.md`](../algorithms/aug_particle_size.md)

## Description

From the `n4m.ParticleSizeAugmenter` Python wrapper docstring:

> Particle-size and path-length scattering simulation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mean_size_um` | `float` | `50.0` |  |
| `size_variation_um` | `float` | `15.0` |  |
| `use_size_range` | `bool` | `False` |  |
| `size_range_low_um` | `float` | `5.0` |  |
| `size_range_high_um` | `float` | `500.0` |  |
| `reference_size_um` | `float` | `50.0` |  |
| `wavelength_exponent` | `float` | `1.5` |  |
| `size_effect_strength` | `float` | `0.1` |  |
| `include_path_length` | `bool` | `True` |  |
| `path_length_sensitivity` | `float` | `0.5` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.scattering.ParticleSizeAugmenter`.

### Mathematical principle

Wavelength-aware augmenter modelling particle-size induced scattering.
For each row:

1. Draw particle size $s_i$:
   - `use_size_range == 1`: $s_i \sim \mathrm{Uniform}(s_{\mathrm{lo}}, s_{\mathrm{hi}})$.
   - else: $s_i \sim \mathrm{Normal}(\mu_s, \sigma_s)$ clipped to $[5, 500]$ μm.
2. Compute size ratio $r_i = s_i / s_{\mathrm{ref}}$, clipped to $[0.1, 10]$.
3. Wavelength factor (per wavelength $\lambda_j$):
$$f_j = \mathrm{clip}(\lambda_j / 1500, 0.1, 10)^{-\text{exponent}}$$
4. Per-sample baseline:
$$b_{i,j} = \mathrm{strength} \cdot (r_i^{-1/2} - 1) \cdot f_j$$
and centre $b_{i,j} \leftarrow b_{i,j} - \mathrm{mean}_j b_{i,j}$.
5. $X^{\mathrm{aug}}_{i,j} = X_{i,j} + b_{i,j}$.
6. If `include_path_length`:
$$X^{\mathrm{aug}}_{i,:} \leftarrow X^{\mathrm{aug}}_{i,:} \cdot \mathrm{clip}(1 + \mathrm{sens} \cdot \log r_i, 0.7, 1.5)$$
7. Add Gaussian-smoothed noise:
$$\eta_i \sim \mathrm{Normal}(0, 0.005 \cdot \mathrm{strength})^p,\quad
X^{\mathrm{aug}}_{i,:} \leftarrow X^{\mathrm{aug}}_{i,:} + G_{\sigma=3}(\eta_i)$$
where $G_{\sigma=3}$ is `scipy.ndimage.gaussian_filter1d` with
default settings (mode `reflect`, `truncate=4.0`).

The noise convolution reuses the existing `n4m_pp_gaussian` engine to
preserve bit-exact kernel construction and convolution semantics.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_particle_size_apply( const n4m_aug_particle_size_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_particle_size_create( n4m_aug_particle_size_handle_t** out, n4m_rng_pcg64_state_t* rng, double mean_size_um, double size_variation_um, int use_size_range, double size_range_low_um, double size_range_high_um, double reference_size_um, double wavelength_exponent, double size_effect_strength, int include_path_length, double path_length_sensitivity, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_particle_size_destroy( n4m_aug_particle_size_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_particle_size` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_particle_size` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.ParticleSizeAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_particle_size(X, wavelengths = NULL, mean_size_um = 50.0, size_variation_um = 15.0, use_size_range = 0L, size_range_low_um = 5.0, size_range_high_um = 500.0, reference_size_um = 50.0, wavelength_exponent = 1.5, size_effect_strength = 0.1, include_path_length = 1L, path_length_sensitivity = 0.5, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.ParticleSizeAugmenter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_particle_size_apply( const n4m_aug_particle_size_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_particle_size_create( n4m_aug_particle_size_handle_t** out, n4m_rng_pcg64_state_t* rng, double mean_size_um, double size_variation_um, int use_size_range, double size_range_low_um, double size_range_high_um, double reference_size_um, double wavelength_exponent, double size_effect_strength, int include_path_length, double path_length_sensitivity, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_particle_size_destroy( n4m_aug_particle_size_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_particle_size(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import ParticleSizeAugmenter

op = ParticleSizeAugmenter(mean_size_um=50.0, size_variation_um=15.0, use_size_range=False, size_range_low_um=5.0, size_range_high_um=500.0, reference_size_um=50.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_particle_size(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.ParticleSizeAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.ParticleSizeAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.109 ms</td><td class="ms">0.709 ms</td><td class="ms">3.354 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms ms-best">🏆 0.107 ms</td><td class="ms ms-best">🏆 0.649 ms</td><td class="ms">3.507 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">2.2e-16</td><td class="ms">0.113 ms</td><td class="ms">0.660 ms</td><td class="ms ms-best">🏆 3.180 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.7e-16</td><td class="ms">0.146 ms</td><td class="ms">0.867 ms</td><td class="ms">4.250 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.ParticleSizeAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">2.261 ms</td><td class="ms">3.661 ms</td><td class="ms">9.002 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
