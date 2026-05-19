# `snv` — SNV

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/snv.md`](../algorithms/snv.md)

## Description

From the `chemometrics4all.SNV` Python wrapper docstring:

> Standard Normal Variate normalisation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `with_mean` | `bool` | `True` |  |
| `with_std` | `bool` | `True` |  |
| `ddof` | `int` | `0` |  |

## Explanations

### Bibliographic source

Barnes, R. J., Dhanoa, M. S., Lister, S. J. (1989). "Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra." *Applied Spectroscopy* 43 (5), 772-777. Implementation matches `nirs4all.operators.transforms.scalers.StandardNormalVariate`.

### Mathematical principle

Per-sample (row-wise) centering and scaling. For each spectrum $x_i \in \mathbb{R}^m$:

$$
\bar{x}_i = \frac{1}{m} \sum_{j=1}^{m} x_{i,j}, \qquad
s_i = \sqrt{\frac{1}{m - \text{ddof}} \sum_{j=1}^{m} (x_{i,j} - \bar{x}_i)^2}
$$

$$
x'_{i,j} = \frac{x_{i,j} - \bar{x}_i}{s_i}
$$

When the row standard deviation falls below $10^{-15}$ (a flat or quasi-flat spectrum) the divisor is replaced by $1.0$ to avoid amplifying numerical noise — this mirrors `np.std`-then-`std[std == 0] = 1.0` in nirs4all.

### Implementation

* Bit-exact against `numpy.mean(X, axis=1, keepdims=True)` followed by `numpy.std(X, axis=1, ddof=ddof, keepdims=True)`.
* Both `mean = sum / N` and `(X - mean) / std` are evaluated as true element-wise divisions (not by reciprocal-multiplication) to match NumPy's rounding sequence.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_snv_create(c4a_pp_snv_handle_t** out, int with_mean, int with_std, int ddof);
void c4a_pp_snv_destroy(c4a_pp_snv_handle_t* handle);
c4a_status_t c4a_pp_snv_transform(const c4a_pp_snv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_snv` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.SNV` | Python | sklearn-style wrapper backed by ctypes. |
| R | `snv(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)` | R | Public package wrapper around the C ABI. |
| ref.numpy | `numpy` | Python | canonical/comparator |
| ref.r.base | `R base` | R | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_snv_create(c4a_pp_snv_handle_t** out, int with_mean, int with_std, int ddof);
void c4a_pp_snv_destroy(c4a_pp_snv_handle_t* handle);
c4a_status_t c4a_pp_snv_transform(const c4a_pp_snv_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import SNV

op = SNV(with_mean=True, with_std=True, ddof=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- snv(X)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.numpy`** (Python · canonical) — `numpy` · numpy 2.3.5
- 📐 **`ref.r.base`** (R · comparator) — `R base`
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms ms-best">🏆 0.063 ms</td><td class="ms ms-best">🏆 0.319 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.011 ms</td><td class="ms">0.068 ms</td><td class="ms">0.328 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.027 ms</td><td class="ms">0.287 ms</td><td class="ms">1.438 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy · numpy 2.3.5 — canonical">📐</span><code>ref.numpy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.024 ms</td><td class="ms">0.133 ms</td><td class="ms">0.702 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R base · R base — comparator">📐</span><code>ref.r.base</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.080 ms</td><td class="ms">0.590 ms</td><td class="ms">3.469 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
