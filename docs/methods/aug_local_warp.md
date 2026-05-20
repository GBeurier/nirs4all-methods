# `aug_local_warp` — Local wavelength warp augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_wavelength_spectral.md`](../algorithms/aug_wavelength_spectral.md)

## Description

Ten stochastic augmenters in the new `c4a_aug_*` ABI category. All ten
implement the locked 3-symbol ABI from
`roadmap/phase-15-18-augmenters-abi-contract.md` (`_create`, `_apply`,
`_destroy`); the RNG handle is the first constructor parameter and is
stored by reference (the augmenter does NOT own it).

| Operator                  | Symbol prefix                  | Algorithm |
| ------------------------- | ------------------------------ | --------- |
| WavelengthShift           | `c4a_aug_wavelength_shift_*`   | `out[i] = np.interp(lambdas - shift_i, lambdas, X[i])`, `shift_i ~ U(lo, hi)` |
| WavelengthStretch         | `c4a_aug_wavelength_stretch_*` | `query = center + (lambdas - center) / f_i`, `out[i] = np.interp(query, lambdas, X[i])`, `f_i ~ U(lo, hi)` |
| LocalWavelengthWarp       | `c4a_aug_local_warp_*`         | linearly-interp `n_control_points` random shifts `~ U(-mx, mx)` to wavelengths, then resample |
| BandPerturbation          | `c4a_aug_band_perturb_*`       | for each (sample, band): multiply by random gain, add random offset, within a random band |
| BandMasking               | `c4a_aug_band_mask_*`          | zero (or ramp-interpolate) `n_per_sample ~ U[n_lo, n_hi]` random bands per sample |
| ChannelDropout            | `c4a_aug_channel_dropout_*`    | mask `rng.random() < p` channels; zero or interpolate from kept neighbours |
| GaussianSmoothingJitter   | `c4a_aug_gauss_jitter_*`       | per-row reflect-padded Gaussian convolution with `sigma_i ~ U(sigma_lo, sigma_hi)` |
| UnsharpSpectralMask       | `c4a_aug_unsharp_mask_*`       | `out = X + amount_i * (X - convolve(X, gauss))`, `amount_i ~ U(amt_lo, amt_hi)` |
| SmoothMagnitudeWarp       | `c4a_aug_magnitude_warp_*`     | linearly-interp `n_control_points` random gains `~ U(g_lo, g_hi)`, multiply spectrum elementwise |
| LocalClipping             | `c4a_aug_local_clip_*`         | clip each of `n_regions` random bands to the 90th-percentile of the band |

From the `chemometrics4all.LocalWarpAugmenter` Python wrapper docstring:

> Random local wavelength warping.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_control_points` | `int` | `5` |  |
| `max_shift` | `float` | `1.0` |  |
| `wavelengths` | `object` | `None` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.LocalWavelengthWarp (Python).

### Mathematical principle

`aug_local_warp` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

NumPy 1.26.4 dispatches `rng.uniform` to PCG64's `next_double` callback
(`(uint64 >> 11) * 2^-53`) and `rng.integers(low, high)` to the buffered
32-bit Lemire path when `high - low <= 2^32`. The C engine replicates
both exactly:

- `c4a_aug_uniform`: `lo + (hi - lo) * next_double` — bit-equivalent to
  `rng.uniform(lo, hi)`.
- `c4a_aug_randint`: buffered 32-bit Lemire (or unbuffered 64-bit Lemire
  for wider ranges) — bit-equivalent to `rng.integers(lo, hi)`.

Random-stream order in each operator matches the Python reference's
allocation pattern (e.g. BandPerturbation draws centers, widths, gains,
offsets as four successive batches in row-major order).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_local_warp_apply( const c4a_aug_local_warp_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_local_warp_create( c4a_aug_local_warp_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t n_control_points, double max_shift, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_local_warp_destroy(c4a_aug_local_warp_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_local_warp` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_local_warp` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.LocalWarpAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_local_warp(X, wavelengths = NULL, n_control_points = 3L, max_shift = 1.0, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.LocalWavelengthWarp` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_local_warp_apply( const c4a_aug_local_warp_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_local_warp_create( c4a_aug_local_warp_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t n_control_points, double max_shift, const double* wavelengths, int64_t n_wavelengths);
void c4a_aug_local_warp_destroy(c4a_aug_local_warp_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_local_warp(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import LocalWarpAugmenter

op = LocalWarpAugmenter(n_control_points=5, max_shift=1.0, wavelengths=None, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_local_warp(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.LocalWavelengthWarp` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.LocalWavelengthWarp` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms">0.052 ms</td><td class="ms">0.790 ms</td><td class="ms">8.557 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms ms-best">🏆 0.052 ms</td><td class="ms">0.811 ms</td><td class="ms">8.341 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms">0.056 ms</td><td class="ms ms-best">🏆 0.781 ms</td><td class="ms">8.554 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-14</td><td class="ms">0.083 ms</td><td class="ms">1.008 ms</td><td class="ms">9.812 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.LocalWavelengthWarp · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.726 ms</td><td class="ms">2.061 ms</td><td class="ms ms-best">🏆 7.346 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
