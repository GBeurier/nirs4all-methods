# `lsnv` — LSNV

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/lsnv.md`](../algorithms/lsnv.md)

## Description

From the `chemometrics4all.LSNV` Python wrapper docstring:

> Sliding-window (local) SNV.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `window` | `int` | `11` |  |
| `pad_mode` | `str` | `'reflect'` |  |
| `constant_value` | `float` | `0.0` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.scalers.LocalStandardNormalVariate`. Sliding-window mean/variance via cumulative sums is the conventional $O(m)$ formulation used in spectral preprocessing toolkits.

### Mathematical principle

A sliding-window variant of SNV that normalises each sample with statistics computed in a local neighbourhood along the wavelength axis. With odd window size $w = 2h + 1$, for each spectrum $x_i$ and each feature index $j$:

$$
\mu_{i,j} = \frac{1}{w} \sum_{k = j - h}^{j + h} \tilde{x}_{i,k}, \qquad
\sigma_{i,j} = \sqrt{\max\!\left(\frac{1}{w} \sum_{k = j - h}^{j + h} \tilde{x}_{i,k}^2 - \mu_{i,j}^2, \; 0\right)}
$$

$$
x'_{i,j} = \frac{x_{i,j} - \mu_{i,j}}{\sigma_{i,j}}
$$

where $\tilde{x}_i$ is the row padded by $h$ samples on each side. Boundary padding follows scipy's convention: **reflect** (mirror without repeating the edge, iterating when the pad exceeds $m - 1$), **edge** (repeat the boundary sample), or **constant** (a user-supplied value).

Variance is clamped at zero to absorb the cancellation that arises when $\mu^2 \approx \overline{x^2}$ on smooth spectra. If $\sigma_{i,j} = 0$ exactly, the divisor is replaced by $1.0$.

### Implementation

* Bit-exact against `np.cumsum(Xp, axis=1) / w` for both first and second moment moving averages.
* The reference reflect padding matches `np.pad(..., mode='reflect')` including the iterative reflection used when pad width exceeds $m - 1$.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_lsnv_create(c4a_pp_lsnv_handle_t** out, int32_t window, int32_t pad_mode, double constant_value);
void c4a_pp_lsnv_destroy(c4a_pp_lsnv_handle_t* handle);
c4a_status_t c4a_pp_lsnv_transform(const c4a_pp_lsnv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_lsnv` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.LSNV` | Python | sklearn-style wrapper backed by ctypes. |
| R | `lsnv(X, window = 11L, pad_mode = "reflect", constant_value = 0.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.LocalStandardNormalVariate` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_lsnv_create(c4a_pp_lsnv_handle_t** out, int32_t window, int32_t pad_mode, double constant_value);
void c4a_pp_lsnv_destroy(c4a_pp_lsnv_handle_t* handle);
c4a_status_t c4a_pp_lsnv_transform(const c4a_pp_lsnv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import LSNV

op = LSNV(window=11, pad_mode='reflect', constant_value=0.0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- lsnv(X, window = 11L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.LocalStandardNormalVariate` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.049 ms</td><td class="ms ms-best">🏆 0.489 ms</td><td class="ms ms-best">🏆 2.521 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.054 ms</td><td class="ms">0.628 ms</td><td class="ms">2.609 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.071 ms</td><td class="ms">0.711 ms</td><td class="ms">4.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.LocalStandardNormalVariate · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.178 ms</td><td class="ms">0.714 ms</td><td class="ms">4.133 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
