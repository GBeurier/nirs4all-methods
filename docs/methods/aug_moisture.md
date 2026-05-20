# `aug_moisture` — Moisture augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_moisture.md`](../algorithms/aug_moisture.md)

## Description

From the `chemometrics4all.MoistureAugmenter` Python wrapper docstring:

> Water activity and moisture-content spectral perturbation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `water_activity_delta` | `float` | `0.0` |  |
| `use_aw_range` | `bool` | `False` |  |
| `aw_low` | `float` | `0.0` |  |
| `aw_high` | `float` | `1.0` |  |
| `reference_water_activity` | `float` | `0.5` |  |
| `free_water_fraction` | `float` | `0.3` |  |
| `bound_water_shift` | `float` | `25.0` |  |
| `moisture_content` | `float` | `0.1` |  |
| `enable_shift` | `bool` | `True` |  |
| `enable_intensity` | `bool` | `True` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.environmental.MoistureAugmenter`,
Büning-Pfaue (2003), Luck (1998).

### Mathematical principle

Models water-band shifts due to changes in water activity ($a_w$) and
moisture content. Water exists in two states (free vs. hydrogen-bonded);
the relative fraction is governed by a sigmoid in $a_w$.

For each row:

1. $a_{w,i} = \mathrm{clip}(a_{w,\mathrm{ref}} + \Delta a_{w,i}, 0, 1)$
   where $\Delta a_{w,i}$ is uniform-sampled if a range is provided, else
   the fixed scalar.
2. $f_{\mathrm{free}} = \mathrm{free\_water\_fraction} \cdot \mathrm{sigmoid}(8(a_w - 0.5))$.
3. $f_{\mathrm{bound}} = 1 - f_{\mathrm{free}}$.
4. If `enable_shift`:
   - shift the water band centred at 1435 nm by $\mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   - shift the water band centred at 1930 nm by $0.8 \cdot \mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   Each shift is applied via `np.interp` with a Gaussian-weighted shift
   window centred at the band centre.
5. If `enable_intensity`:
   $$\mathrm{enhance} = \mathrm{moisture\_content} / 0.10 - 1$$
   $$X_{i,:} \mathrel{+}= \mathrm{enhance} \cdot 0.1 \cdot (g_{1435} + g_{1930}) \cdot |\overline{X_i}|$$
   where $g_c$ is a Gaussian region centred at wavelength $c$.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_moisture_apply( const c4a_aug_moisture_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_moisture_create( c4a_aug_moisture_handle_t** out, c4a_rng_pcg64_state_t* rng, double water_activity_delta, int use_aw_range, double aw_low, double aw_high, double reference_water_activity, double free_water_fraction, double bound_water_shift, double moisture_content, int enable_shift, int enable_intensity, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_moisture_destroy( c4a_aug_moisture_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_moisture` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_moisture` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.MoistureAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_moisture(X, wavelengths = NULL, water_activity_delta = 0.1, use_aw_range = 0L, aw_low = 0.0, aw_high = 1.0, reference_water_activity = 0.5, free_water_fraction = 0.3, bound_water_shift = 25.0, moisture_content = 0.1, enable_shift = 1L, enable_intensity = 1L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.MoistureAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_moisture_apply( const c4a_aug_moisture_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_moisture_create( c4a_aug_moisture_handle_t** out, c4a_rng_pcg64_state_t* rng, double water_activity_delta, int use_aw_range, double aw_low, double aw_high, double reference_water_activity, double free_water_fraction, double bound_water_shift, double moisture_content, int enable_shift, int enable_intensity, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_moisture_destroy( c4a_aug_moisture_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_moisture(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import MoistureAugmenter

op = MoistureAugmenter(water_activity_delta=0.0, use_aw_range=False, aw_low=0.0, aw_high=1.0, reference_water_activity=0.5, free_water_fraction=0.3)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_moisture(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.MoistureAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.MoistureAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.117 ms</td><td class="ms">1.491 ms</td><td class="ms">9.277 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.119 ms</td><td class="ms ms-best">🏆 1.477 ms</td><td class="ms">9.272 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.121 ms</td><td class="ms">1.549 ms</td><td class="ms">9.455 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.145 ms</td><td class="ms">1.922 ms</td><td class="ms">19.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.MoistureAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">1.772 ms</td><td class="ms">2.824 ms</td><td class="ms ms-best">🏆 8.036 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
