# `aug_temperature` — Temperature augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_temperature.md`](../algorithms/aug_temperature.md)

## Description

From the `n4m.TemperatureAugmenter` Python wrapper docstring:

> Temperature-induced shift, intensity and broadening perturbations.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `temperature_delta` | `float` | `0.0` |  |
| `use_temp_range` | `bool` | `False` |  |
| `temp_low` | `float` | `-5.0` |  |
| `temp_high` | `float` | `5.0` |  |
| `enable_shift` | `bool` | `True` |  |
| `enable_intensity` | `bool` | `True` |  |
| `enable_broadening` | `bool` | `True` |  |
| `region_specific` | `bool` | `True` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.environmental.TemperatureAugmenter`.

### Mathematical principle

Wavelength-aware NIR temperature effect simulator. Models three effects
across six O-H / C-H / N-H / water spectral regions:

1. **Peak position shift**: `wl_aug = wl + shift_per_°C * Δt`
2. **Intensity change**: `factor = 1 + intensity_per_°C * Δt`
3. **Band broadening**: `factor = 1 + broadening_per_°C * Δt`
   (applied via `gaussian_filter1d` with sigma = `(factor - 1) * 3`).

Each region has its own coefficients, baked into
`TEMPERATURE_REGION_PARAMS`:

| Region | range (nm) | shift/°C | intensity/°C | broadening/°C |
|--------|-----------|----------|--------------|---------------|
| OH 1st overtone | 1400–1520 | −0.30 | −0.002  | 0.001  |
| OH combination  | 1900–2000 | −0.40 | −0.003  | 0.0012 |
| CH 1st overtone | 1650–1780 | −0.05 | −0.0005 | 0.0008 |
| NH 1st overtone | 1490–1560 | −0.20 | −0.0015 | 0.001  |
| Water free      | 1380–1420 | −0.10 |  0.003  | 0.0008 |
| Water bound     | 1440–1500 | −0.35 | −0.004  | 0.0015 |

Effects are localised via a smooth sigmoid weight:

$$w_j = \frac{1}{1 + e^{-10(\lambda_j - \lambda_{\min})/20}} \cdot \frac{1}{1 + e^{10(\lambda_j - \lambda_{\max})/20}}$$

In **region-specific** mode (default), every region's effects are layered.
In **uniform mode** (`region_specific = 0`), the average coefficients are
applied across all wavelengths.

If `temperature_range` is provided, $\Delta t_i \sim \mathrm{Uniform}(\mathrm{low}, \mathrm{high})$
per row; otherwise the fixed `temperature_delta` is used for every row.

### Implementation

When `temperature_delta = 0` and no range is set, the augmenter is the
identity (early exit when `|Δt| < 0.01`). The parity fixture exercises
this case (`temperature_zero_delta`).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_temperature_apply( const n4m_aug_temperature_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_temperature_create( n4m_aug_temperature_handle_t** out, n4m_rng_pcg64_state_t* rng, double temperature_delta, int use_temp_range, double temp_low, double temp_high, int enable_shift, int enable_intensity, int enable_broadening, int region_specific, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_temperature_destroy( n4m_aug_temperature_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_temperature` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_temperature` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.TemperatureAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_temperature(X, wavelengths = NULL, temperature_delta = 5.0, use_temp_range = 0L, temp_low = -5.0, temp_high = 5.0, enable_shift = 1L, enable_intensity = 1L, enable_broadening = 1L, region_specific = 1L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.TemperatureAugmenter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_temperature_apply( const n4m_aug_temperature_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_temperature_create( n4m_aug_temperature_handle_t** out, n4m_rng_pcg64_state_t* rng, double temperature_delta, int use_temp_range, double temp_low, double temp_high, int enable_shift, int enable_intensity, int enable_broadening, int region_specific, const double* wavelengths, int64_t n_wavelengths);
void n4m_aug_temperature_destroy( n4m_aug_temperature_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_temperature(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import TemperatureAugmenter

op = TemperatureAugmenter(temperature_delta=0.0, use_temp_range=False, temp_low=-5.0, temp_high=5.0, enable_shift=True, enable_intensity=True)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_temperature(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.TemperatureAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.TemperatureAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.361 ms</td><td class="ms">4.373 ms</td><td class="ms">26.142 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.349 ms</td><td class="ms">4.336 ms</td><td class="ms">27.435 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.364 ms</td><td class="ms">4.446 ms</td><td class="ms">26.802 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms ms-best">🏆 0.332 ms</td><td class="ms ms-best">🏆 4.188 ms</td><td class="ms">48.000 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.TemperatureAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">5.206 ms</td><td class="ms">8.255 ms</td><td class="ms ms-best">🏆 20.802 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
