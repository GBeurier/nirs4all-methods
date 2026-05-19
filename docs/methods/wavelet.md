# `wavelet` — Wavelet

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/wavelet.md`](../algorithms/wavelet.md)

## Description

From the `chemometrics4all.Wavelet` Python wrapper docstring:

> Single-level DWT coefficient transform.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `family` | `str` | `'haar'` | Wavelet family. |
| `mode` | `str` | `'periodization'` | Boundary handling mode. |

## Explanations

### Bibliographic source

`nirs4all.operators.transforms.nirs.Wavelet` (single-level DWT;
chemometrics4all returns both `cA` and `cD` instead of resampling the
detail band). Filter banks: frozen PyWavelets 1.6.0 oracle, with current
upstream comparison handled by reference parity.

### Mathematical principle

Single-level discrete wavelet transform (DWT) applied row-by-row.  For
each row $x \in \mathbb{R}^n$ the operator computes:

$$
c_A[k] = \sum_{i=0}^{L-1} h_{\text{lo}}[L - 1 - i] \cdot \tilde{x}[2k + i - (L/2 - 1)],
\qquad
c_D[k] = \sum_{i=0}^{L-1} h_{\text{hi}}[L - 1 - i] \cdot \tilde{x}[2k + i - (L/2 - 1)],
$$

where $h_{\text{lo}}, h_{\text{hi}}$ are the analysis low-pass / high-pass
filters of length $L$ for the chosen wavelet family and $\tilde{x}$ is
the input signal under the chosen boundary extension.

The output is the row-major concatenation $[c_A \,\Vert\, c_D]$ of
length $2m$ where $m$ is the per-mode output length:

- **periodization**: $m = \lceil n / 2 \rceil$ (odd $n$ is internally
  padded with one copy of the last sample).
- **symmetric** / **zero**: $m = \lfloor (n + L - 1) / 2 \rfloor$.

Supported wavelet families: **haar** ($L=2$), **db4** ($L=8$), **sym4**
($L=8$), **coif1** ($L=6$). Filter banks are vendored from the frozen
PyWavelets 1.6.0 oracle; current PyWavelets package drift is checked
separately by the reference parity gate.

### Implementation

Stateless: `_create / _transform / _destroy` plus the
`_output_cols` helper.  No `_fit` step.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_wavelet_create(c4a_pp_wavelet_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode);
c4a_status_t c4a_pp_wavelet_denoise_create( c4a_pp_wavelet_denoise_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t level, c4a_pp_wavelet_threshold_t threshold_mode, c4a_pp_wavelet_noise_t noise_estimator);
void c4a_pp_wavelet_denoise_destroy( c4a_pp_wavelet_denoise_handle_t* handle);
c4a_status_t c4a_pp_wavelet_denoise_transform( const c4a_pp_wavelet_denoise_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
void c4a_pp_wavelet_destroy(c4a_pp_wavelet_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_create( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level);
c4a_status_t c4a_pp_wavelet_features_create_ex( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level, c4a_pp_wavelet_features_entropy_t entropy_mode);
void c4a_pp_wavelet_features_destroy( c4a_pp_wavelet_features_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_output_cols( const c4a_pp_wavelet_features_handle_t* handle, int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_wavelet_features_transform( const c4a_pp_wavelet_features_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_pp_wavelet_output_cols( const c4a_pp_wavelet_handle_t* handle, int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_wavelet_pca_create( c4a_pp_wavelet_pca_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level, double n_components);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_wavelet` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.Wavelet` | Python | sklearn-style wrapper backed by ctypes. |
| R | `wavelet(X, family = "haar", boundary = "periodization")` | R | Public package wrapper around the C ABI. |
| ref.pywavelets | `PyWavelets.dwt(haar, periodization)` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.Wavelet(detail-resampled)` | Python | context only; nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_wavelet_create(c4a_pp_wavelet_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode);
c4a_status_t c4a_pp_wavelet_denoise_create( c4a_pp_wavelet_denoise_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t level, c4a_pp_wavelet_threshold_t threshold_mode, c4a_pp_wavelet_noise_t noise_estimator);
void c4a_pp_wavelet_denoise_destroy( c4a_pp_wavelet_denoise_handle_t* handle);
c4a_status_t c4a_pp_wavelet_denoise_transform( const c4a_pp_wavelet_denoise_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
void c4a_pp_wavelet_destroy(c4a_pp_wavelet_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_create( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level);
c4a_status_t c4a_pp_wavelet_features_create_ex( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level, c4a_pp_wavelet_features_entropy_t entropy_mode);
void c4a_pp_wavelet_features_destroy( c4a_pp_wavelet_features_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_output_cols( const c4a_pp_wavelet_features_handle_t* handle, int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_wavelet_features_transform( const c4a_pp_wavelet_features_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import Wavelet

op = Wavelet(family='haar', mode='periodization')
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- wavelet(X, family = 'haar', boundary = 'periodization')
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.pywavelets`** (Python · canonical) — `PyWavelets.dwt(haar, periodization)` · pywt 1.8.0
- ℹ **`ref.nirs4all`** (Python · context) — `nirs4all.Wavelet(detail-resampled)` · nirs4all@cd731a23+dirty — nirs4all returns resampled detail coefficients only; c4a/PyWavelets gate the concatenated approximation+detail coefficients
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.002 ms</td><td class="ms ms-best">🏆 0.015 ms</td><td class="ms ms-best">🏆 0.081 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.008 ms</td><td class="ms">0.023 ms</td><td class="ms">0.095 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.024 ms</td><td class="ms">0.225 ms</td><td class="ms">1.422 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Wavelet(detail-resampled) · nirs4all@cd731a23+dirty — context">📐</span><code>ref.nirs4all</code></td><td class="parity parity-context">✓ ref</td><td class="ms">0.061 ms</td><td class="ms">0.304 ms</td><td class="ms">1.802 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): PyWavelets.dwt(haar, periodization) · pywt 1.8.0 — canonical">📐</span><code>ref.pywavelets</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.013 ms</td><td class="ms">0.066 ms</td><td class="ms">0.333 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
