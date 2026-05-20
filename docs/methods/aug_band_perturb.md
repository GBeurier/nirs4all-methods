# `aug_band_perturb` — Band perturbation augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_wavelength_spectral.md`](../algorithms/aug_wavelength_spectral.md)

## Description

Ten stochastic augmenters in the new `n4m_aug_*` ABI category. All ten
implement the locked 3-symbol ABI from
`roadmap/phase-15-18-augmenters-abi-contract.md` (`_create`, `_apply`,
`_destroy`); the RNG handle is the first constructor parameter and is
stored by reference (the augmenter does NOT own it).

| Operator                  | Symbol prefix                  | Algorithm |
| ------------------------- | ------------------------------ | --------- |
| WavelengthShift           | `n4m_aug_wavelength_shift_*`   | `out[i] = np.interp(lambdas - shift_i, lambdas, X[i])`, `shift_i ~ U(lo, hi)` |
| WavelengthStretch         | `n4m_aug_wavelength_stretch_*` | `query = center + (lambdas - center) / f_i`, `out[i] = np.interp(query, lambdas, X[i])`, `f_i ~ U(lo, hi)` |
| LocalWavelengthWarp       | `n4m_aug_local_warp_*`         | linearly-interp `n_control_points` random shifts `~ U(-mx, mx)` to wavelengths, then resample |
| BandPerturbation          | `n4m_aug_band_perturb_*`       | for each (sample, band): multiply by random gain, add random offset, within a random band |
| BandMasking               | `n4m_aug_band_mask_*`          | zero (or ramp-interpolate) `n_per_sample ~ U[n_lo, n_hi]` random bands per sample |
| ChannelDropout            | `n4m_aug_channel_dropout_*`    | mask `rng.random() < p` channels; zero or interpolate from kept neighbours |
| GaussianSmoothingJitter   | `n4m_aug_gauss_jitter_*`       | per-row reflect-padded Gaussian convolution with `sigma_i ~ U(sigma_lo, sigma_hi)` |
| UnsharpSpectralMask       | `n4m_aug_unsharp_mask_*`       | `out = X + amount_i * (X - convolve(X, gauss))`, `amount_i ~ U(amt_lo, amt_hi)` |
| SmoothMagnitudeWarp       | `n4m_aug_magnitude_warp_*`     | linearly-interp `n_control_points` random gains `~ U(g_lo, g_hi)`, multiply spectrum elementwise |
| LocalClipping             | `n4m_aug_local_clip_*`         | clip each of `n_regions` random bands to the 90th-percentile of the band |

From the `n4m.BandPerturbationAugmenter` Python wrapper docstring:

> Random band-local gain and offset perturbations.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_bands` | `int` | `3` |  |
| `bw_lo` | `int` | `5` |  |
| `bw_hi` | `int` | `15` |  |
| `gain_lo` | `float` | `0.9` |  |
| `gain_hi` | `float` | `1.1` |  |
| `offset_lo` | `float` | `-0.01` |  |
| `offset_hi` | `float` | `0.01` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.BandPerturbation (Python).

### Mathematical principle

`aug_band_perturb` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

NumPy 1.26.4 dispatches `rng.uniform` to PCG64's `next_double` callback
(`(uint64 >> 11) * 2^-53`) and `rng.integers(low, high)` to the buffered
32-bit Lemire path when `high - low <= 2^32`. The C engine replicates
both exactly:

- `n4m_aug_uniform`: `lo + (hi - lo) * next_double` — bit-equivalent to
  `rng.uniform(lo, hi)`.
- `n4m_aug_randint`: buffered 32-bit Lemire (or unbuffered 64-bit Lemire
  for wider ranges) — bit-equivalent to `rng.integers(lo, hi)`.

Random-stream order in each operator matches the Python reference's
allocation pattern (e.g. BandPerturbation draws centers, widths, gains,
offsets as four successive batches in row-major order).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_band_perturb_apply( const n4m_aug_band_perturb_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_band_perturb_create( n4m_aug_band_perturb_handle_t** out, n4m_rng_pcg64_state_t* rng, int32_t n_bands, int32_t bw_lo, int32_t bw_hi, double gain_lo, double gain_hi, double offset_lo, double offset_hi);
void n4m_aug_band_perturb_destroy(n4m_aug_band_perturb_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_band_perturb` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_band_perturb` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.BandPerturbationAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_band_perturb(X, n_bands = 3L, bw_lo = 5L, bw_hi = 15L, gain_lo = 0.9, gain_hi = 1.1, offset_lo = -0.01, offset_hi = 0.01, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.BandPerturbation` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_band_perturb_apply( const n4m_aug_band_perturb_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_band_perturb_create( n4m_aug_band_perturb_handle_t** out, n4m_rng_pcg64_state_t* rng, int32_t n_bands, int32_t bw_lo, int32_t bw_hi, double gain_lo, double gain_hi, double offset_lo, double offset_hi);
void n4m_aug_band_perturb_destroy(n4m_aug_band_perturb_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_band_perturb(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import BandPerturbationAugmenter

op = BandPerturbationAugmenter(n_bands=3, bw_lo=5, bw_hi=15, gain_lo=0.9, gain_hi=1.1, offset_lo=-0.01)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_band_perturb(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.BandPerturbation` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.BandPerturbation` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.011 ms</td><td class="ms">0.020 ms</td><td class="ms">0.070 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.012 ms</td><td class="ms ms-best">🏆 0.020 ms</td><td class="ms">0.068 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.012 ms</td><td class="ms">0.023 ms</td><td class="ms ms-best">🏆 0.068 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.048 ms</td><td class="ms">0.204 ms</td><td class="ms">1.203 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.BandPerturbation · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.477 ms</td><td class="ms">0.512 ms</td><td class="ms">0.706 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
