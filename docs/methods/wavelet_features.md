# `wavelet_features` — Wavelet features

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/wavelet_features.md`](../algorithms/wavelet_features.md)

## Description

From the `chemometrics4all.WaveletFeatures` Python wrapper docstring:

> Multi-level DWT summary features.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `family` | `str` | `'haar'` | Wavelet family. |
| `mode` | `str` | `'periodization'` | Boundary handling mode. |
| `max_level` | `int` | `3` |  |
| `entropy` | `str` | `'energy'` |  |

## Explanations

### Bibliographic source

`PyWavelets.wavedec(...)+band stats` is the canonical external reference used
for exact parity snapshots. The dashboard benchmark currently uses
`haar + symmetric + histogram` so the c4a, PyWavelets, and nirs4all rows compare
the same feature contract.

`nirs4all.operators.transforms.nirs.WaveletFeatures` is also timed as an
external comparator. Its default surface adds top-K retained coefficients per
band; the benchmark calls it with `n_coeffs_per_level=0` so its output has the
same feature count as chemometrics4all. The benchmark selects the same
histogram entropy and symmetric boundary contract, so this row carries an
external comparator parity gate.

chemometrics4all reduces the nirs4all surface to the four canonical statistics
in v1; top-K coefficient retention is omitted to keep the ABI and parity
surface compact.

### Mathematical principle

Multi-level wavelet decomposition followed by per-band statistical
summarisation.  For each input row a multi-level DWT produces
$K + 1$ coefficient bands $\{c_A^{(K)}, c_D^{(K)}, c_D^{(K-1)}, \ldots, c_D^{(1)}\}$
(where $K$ is the effective level, clamped to
`pywt.dwt_max_level(n, family)`), and each band is summarised by four
statistics:

$$
\text{mean}(c) = \frac{1}{|c|} \sum_i c_i, \qquad
\text{std}(c) = \sqrt{\tfrac{1}{|c|} \sum_i (c_i - \text{mean}(c))^2}
$$
$$
\text{energy}(c) = \sum_i c_i^2
$$

(`std` uses population variance `ddof=0`.)

The entropy statistic is selectable:

- `entropy="energy"`: Shannon entropy of the squared-coefficient
  distribution, `p_i = c_i^2 / sum(c^2)`. Zero-energy bands return 0.
- `entropy="histogram"`: ten equal-width histogram bins, normalized as a
  probability distribution before `-sum(p log p)`. This matches
  `nirs4all.WaveletFeatures` when the boundary mode is also aligned to
  PyWavelets' default `symmetric`.

The output row has length $4 (K + 1)$ laid out as
$[\text{approx stats}, \text{detail}_K \text{ stats}, \ldots,
\text{detail}_1 \text{ stats}]$.  `_output_cols(input_cols)` returns
this value.

### Implementation

Stateless: `_create / _transform / _destroy` plus `_output_cols`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_wavelet_features_create( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level);
c4a_status_t c4a_pp_wavelet_features_create_ex( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level, c4a_pp_wavelet_features_entropy_t entropy_mode);
void c4a_pp_wavelet_features_destroy( c4a_pp_wavelet_features_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_output_cols( const c4a_pp_wavelet_features_handle_t* handle, int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_wavelet_features_transform( const c4a_pp_wavelet_features_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_wavelet_features` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.wavelet_features` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.WaveletFeatures` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `wavelet_features(X, family = "haar", boundary = "periodization", max_level = 2L, entropy = "energy")` | R | Public package wrapper around the C ABI. |
| ref.pywavelets | `PyWavelets.wavedec(haar, symmetric)+histogram band stats` | Python | canonical/comparator |
| ref.nirs4all | `nirs4all.WaveletFeatures(n_coeffs_per_level=0)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_wavelet_features_create( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level);
c4a_status_t c4a_pp_wavelet_features_create_ex( c4a_pp_wavelet_features_handle_t** out, c4a_pp_wavelet_family_t family, c4a_pp_wavelet_boundary_t mode, int32_t max_level, c4a_pp_wavelet_features_entropy_t entropy_mode);
void c4a_pp_wavelet_features_destroy( c4a_pp_wavelet_features_handle_t* handle);
c4a_status_t c4a_pp_wavelet_features_output_cols( const c4a_pp_wavelet_features_handle_t* handle, int64_t input_cols, int64_t* out_cols);
c4a_status_t c4a_pp_wavelet_features_transform( const c4a_pp_wavelet_features_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.wavelet_features(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import WaveletFeatures

op = WaveletFeatures(family='haar', mode='periodization', max_level=3, entropy='energy')
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- wavelet_features(X, family = 'haar', boundary = 'symmetric', max_level = 2L, entropy = 'histogram')
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pywavelets`** (Python · canonical) — `PyWavelets.wavedec(haar, symmetric)+histogram band stats` · PyWavelets 1.9.0
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.WaveletFeatures(n_coeffs_per_level=0)` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.pywavelets` | `PyWavelets.wavedec(haar, symmetric)+histogram band stats` | Python / parity | `default_allclose` |  |
| `ref.nirs4all` | `nirs4all.WaveletFeatures(n_coeffs_per_level=0)` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.4e-12</td><td class="ms ms-best">🏆 0.037 ms</td><td class="ms">0.219 ms</td><td class="ms">2.015 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.4e-12</td><td class="ms">0.040 ms</td><td class="ms ms-best">🏆 0.218 ms</td><td class="ms">2.799 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.4e-12</td><td class="ms">0.044 ms</td><td class="ms">0.235 ms</td><td class="ms">3.274 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.8e-12</td><td class="ms">0.061 ms</td><td class="ms">0.357 ms</td><td class="ms ms-best">🏆 1.875 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.WaveletFeatures(n_coeffs_per_level=0) · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">8.9e-16</td><td class="ms">61.667 ms</td><td class="ms">64.588 ms</td><td class="ms">74.595 ms</td></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): PyWavelets.wavedec(haar, symmetric)+histogram band stats · PyWavelets 1.9.0 — canonical">◆</span><code>ref.pywavelets</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">16.052 ms</td><td class="ms">17.344 ms</td><td class="ms">36.178 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
