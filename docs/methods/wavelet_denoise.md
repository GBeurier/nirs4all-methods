# `wavelet_denoise` — Wavelet denoise

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/wavelet_denoise.md`](../algorithms/wavelet_denoise.md)

## Description

From the `chemometrics4all.WaveletDenoise` Python wrapper docstring:

> Multi-level DWT VisuShrink denoising.

<details><summary>Full wrapper docstring</summary>

```text
Multi-level DWT VisuShrink denoising.

Stateless: matches PyWavelets' ``waverec(threshold(coeffs))`` pipeline.
```

</details>

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `family` | `str` | `'db4'` | Wavelet family. |
| `mode` | `str` | `'periodization'` | Boundary handling mode. |
| `level` | `int` | `3` | Wavelet decomposition level. |
| `threshold_mode` | `str` | `'soft'` | Wavelet thresholding rule. |
| `noise_estimator` | `str` | `'median'` | Noise estimator used for denoising threshold. |

## Explanations

### Bibliographic source

`nirs4all.operators.transforms.wavelet_denoise.WaveletDenoise` and the
Donoho & Johnstone (1994) ideal-spatial-adaptation paper.

### Mathematical principle

Multi-level wavelet denoising with the Donoho universal threshold and
soft / hard thresholding.  Implements the recipe from
`nirs4all.operators.transforms.wavelet_denoise.WaveletDenoise`.

For each row $x \in \mathbb{R}^n$:

1. Multi-level DWT decomposition produces
   $\{c_A^{(K)}, c_D^{(K)}, c_D^{(K-1)}, \ldots, c_D^{(1)}\}$
   where $K$ is the effective level (clamped to
   `pywt.dwt_max_level(n, family)`).
2. Per-row noise estimate from the finest detail $c_D^{(1)}$:

   $$
   \sigma_{\text{MAD}} = \frac{\mathrm{median}\bigl(|c_D^{(1)}|\bigr)}{0.6745},
   \qquad
   \sigma_{\text{std}} = \mathrm{std}\bigl(c_D^{(1)}\bigr).
   $$
3. Universal threshold:
   $\theta = \sigma \cdot \sqrt{2 \log n}$.
4. Soft / hard thresholding applied to every detail band:

   $$
   T_{\text{soft}}(y; \theta) = \mathrm{sign}(y) \cdot \max(|y| - \theta, 0),
   \qquad
   T_{\text{hard}}(y; \theta) = \begin{cases} y & |y| > \theta \\ 0 & \text{otherwise} \end{cases}.
   $$
   The approximation band $c_A^{(K)}$ is kept untouched.
5. Multi-level reconstruction yields the denoised row, truncated to the
   original length $n$.

If $K$ clamps to 0 (filter longer than input) the operator passes the
input through unchanged.

### Implementation

Stateless: `_create / _transform / _destroy`.  Output shape matches the
input shape.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_wavelet_denoise_create( c4a_pp_wavelet_denoise_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t level, c4a_pp_wavelet_threshold_t threshold_mode, c4a_pp_wavelet_noise_t noise_estimator);
void c4a_pp_wavelet_denoise_destroy( c4a_pp_wavelet_denoise_handle_t* handle);
c4a_status_t c4a_pp_wavelet_denoise_transform( const c4a_pp_wavelet_denoise_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_wavelet_denoise` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.WaveletDenoise` | Python | sklearn-style wrapper backed by ctypes. |
| R | `wavelet_denoise(X, family = "db4", boundary = "periodization", level = 3L, threshold_mode = "soft", noise_estimator = "median")` | R | Public package wrapper around the C ABI. |
| ref.pywavelets | `PyWavelets.wavedec/waverec(db4, periodization)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.WaveletDenoise` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_wavelet_denoise_create( c4a_pp_wavelet_denoise_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t level, c4a_pp_wavelet_threshold_t threshold_mode, c4a_pp_wavelet_noise_t noise_estimator);
void c4a_pp_wavelet_denoise_destroy( c4a_pp_wavelet_denoise_handle_t* handle);
c4a_status_t c4a_pp_wavelet_denoise_transform( const c4a_pp_wavelet_denoise_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import WaveletDenoise

op = WaveletDenoise(family='db4', mode='periodization', level=3, threshold_mode='soft', noise_estimator='median')
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- wavelet_denoise(X, level = 2L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pywavelets`** (Python · canonical) — `PyWavelets.wavedec/waverec(db4, periodization)` · pywt 1.8.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.WaveletDenoise` · nirs4all@cd731a23+dirty
:::

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ context/drift · ✗ divergent · ⊘ not available · — not run · ⚠ error.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Parity</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.054 ms</td><td class="ms ms-best">🏆 0.546 ms</td><td class="ms ms-best">🏆 2.807 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.059 ms</td><td class="ms">0.571 ms</td><td class="ms">2.890 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.077 ms</td><td class="ms">0.805 ms</td><td class="ms">4.750 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.WaveletDenoise · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.143 ms</td><td class="ms">0.724 ms</td><td class="ms">3.812 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): PyWavelets.wavedec/waverec(db4, periodization) · pywt 1.8.0 — canonical">◆</span><code>ref.pywavelets</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">3.546 ms</td><td class="ms">3.897 ms</td><td class="ms">7.090 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
