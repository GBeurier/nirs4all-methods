# `aug_wavelength_spectral` — Wavelength/spectral augmenters

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

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all wavelength/spectral augmenter bundle (Python).

### Mathematical principle

`aug_wavelength_spectral` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

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

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_wavelength_spectral` | Python | ABI-close function backed by ctypes. |
| R | `aug_wavelength_spectral(X, wavelengths = NULL, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all wavelength/spectral augmenter bundle` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
n4m_*
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_wavelength_spectral(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_wavelength_spectral(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all wavelength/spectral augmenter bundle` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all wavelength/spectral augmenter bundle` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms ms-best">🏆 0.293 ms</td><td class="ms">3.703 ms</td><td class="ms">30.147 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms">0.304 ms</td><td class="ms">3.823 ms</td><td class="ms">30.535 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.0e-14</td><td class="ms">0.326 ms</td><td class="ms ms-best">🏆 3.454 ms</td><td class="ms ms-best">🏆 29.786 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-14</td><td class="ms">0.438 ms</td><td class="ms">5.625 ms</td><td class="ms">43.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all wavelength/spectral augmenter bundle · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">8.021 ms</td><td class="ms">12.803 ms</td><td class="ms">31.262 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
