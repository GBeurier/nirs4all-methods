# `aug_instrument_broaden` — Instrumental broadening augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_instrument_broaden.md`](../algorithms/aug_instrument_broaden.md)

## Description

From the `chemometrics4all.InstrumentalBroadeningAugmenter` Python wrapper docstring:

> Instrumental spectral broadening via Gaussian convolution.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `fwhm` | `float` | `5.0` |  |
| `use_fwhm_range` | `bool` | `False` |  |
| `fwhm_low` | `float` | `3.0` |  |
| `fwhm_high` | `float` | `8.0` |  |
| `variation_scope` | `int` | `0` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.InstrumentalBroadeningAugmenter`.

### Mathematical principle

Models finite-resolution instrumental broadening via Gaussian convolution
with FWHM-derived sigma. The relationship between FWHM (in wavelength
units, typically nm) and Gaussian sigma in points is:

$$\sigma_{\mathrm{pts}} = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2} \cdot \mathrm{wl\_step}}$$

where `wl_step = median(diff(wavelengths))` when wavelengths is supplied,
else `1.0`.

Three behaviours:

- **Fixed FWHM** (`use_fwhm_range = 0`): one $\sigma_{\mathrm{pts}}$ for all rows; no RNG draws.
- **Per-sample range** (`use_fwhm_range = 1`, `variation_scope = 0`): one
  `FWHM ~ U(low, high)` per row.
- **Batch range** (`use_fwhm_range = 1`, `variation_scope = 1`): one
  `FWHM ~ U(low, high)` shared across the batch.

Each row is then row-wise convolved with a Gaussian of the corresponding
$\sigma_{\mathrm{pts}}$. The convolution is delegated to the existing
`c4a_pp_gaussian` engine (mode `reflect`, `truncate=4.0`), which has
$10^{-9}$ parity with `scipy.ndimage.gaussian_filter1d`.

### Implementation

`use_fwhm_range = 0` makes the augmenter fully deterministic — no RNG draws,
output is uniquely determined by `fwhm`, `wavelengths`, and `X`. The parity
fixture exercises this case (`instrument_broaden_fixed_fwhm5`).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_instrument_broaden_apply( const c4a_aug_instrument_broaden_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_instrument_broaden_create( c4a_aug_instrument_broaden_handle_t** out, c4a_rng_pcg64_state_t* rng, double fwhm, int use_fwhm_range, double fwhm_low, double fwhm_high, int32_t variation_scope, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_instrument_broaden_destroy( c4a_aug_instrument_broaden_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_instrument_broaden` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_instrument_broaden` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.InstrumentalBroadeningAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_instrument_broaden(X, wavelengths = NULL, fwhm = 3.0, use_fwhm_range = 0L, fwhm_low = 3.0, fwhm_high = 8.0, variation_scope = 0L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.InstrumentalBroadeningAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_instrument_broaden_apply( const c4a_aug_instrument_broaden_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_instrument_broaden_create( c4a_aug_instrument_broaden_handle_t** out, c4a_rng_pcg64_state_t* rng, double fwhm, int use_fwhm_range, double fwhm_low, double fwhm_high, int32_t variation_scope, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_instrument_broaden_destroy( c4a_aug_instrument_broaden_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_instrument_broaden(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import InstrumentalBroadeningAugmenter

op = InstrumentalBroadeningAugmenter(fwhm=5.0, use_fwhm_range=False, fwhm_low=3.0, fwhm_high=8.0, variation_scope=0, wavelengths=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_instrument_broaden(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.InstrumentalBroadeningAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.InstrumentalBroadeningAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms ms-best">🏆 0.014 ms</td><td class="ms">0.124 ms</td><td class="ms">2.124 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.016 ms</td><td class="ms ms-best">🏆 0.107 ms</td><td class="ms ms-best">🏆 2.018 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.016 ms</td><td class="ms">0.114 ms</td><td class="ms">2.048 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-15</td><td class="ms">0.043 ms</td><td class="ms">0.299 ms</td><td class="ms">3.078 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.InstrumentalBroadeningAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.684 ms</td><td class="ms">0.927 ms</td><td class="ms">2.172 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
