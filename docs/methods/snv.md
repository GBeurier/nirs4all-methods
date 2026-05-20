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

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_snv` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.snv` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.SNV` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `snv(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)` | R | Public package wrapper around the C ABI. |
| ref.r.prospectr | `prospectr.standardNormalVariate` | R | canonical/comparator; c4a benchmark uses ddof=1 to match prospectr's sample-standard-deviation convention |
| ref.nirs4all | `nirs4all.StandardNormalVariate` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

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

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.snv(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import SNV

op = SNV(with_mean=True, with_std=True, ddof=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- snv(X, ddof = 1L)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.r.prospectr`** (R · canonical) — `prospectr.standardNormalVariate` · prospectr 0.2.8 — c4a benchmark uses ddof=1 to match prospectr's sample-standard-deviation convention
- ◆ **`ref.nirs4all`** (Python · comparator) — `nirs4all.StandardNormalVariate` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `transform` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `64×128` · seed `1234567890`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `finite_output`, `max_abs_diff`, `shape_equal`
- Truth sources:
  - `nirs4all_snv` — nirs4all.operators.preprocessing.snv (nirs4all, git-pinned by benchmark environment)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.r.prospectr` | `prospectr.standardNormalVariate` | R / parity | `default_allclose` | c4a benchmark uses ddof=1 to match prospectr's sample-standard-deviation convention |
| `ref.nirs4all` | `nirs4all.StandardNormalVariate` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-14</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms ms-best">🏆 0.062 ms</td><td class="ms ms-best">🏆 0.319 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-14</td><td class="ms">0.011 ms</td><td class="ms">0.072 ms</td><td class="ms">0.333 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-14</td><td class="ms">0.014 ms</td><td class="ms">0.073 ms</td><td class="ms">0.348 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">3.0e-14</td><td class="ms">0.030 ms</td><td class="ms">0.303 ms</td><td class="ms">1.984 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.StandardNormalVariate · nirs4all@cd731a23+dirty — comparator">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">7.3e-15</td><td class="ms">0.075 ms</td><td class="ms">0.241 ms</td><td class="ms">1.251 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): prospectr.standardNormalVariate · prospectr 0.2.8 — canonical">◆</span><code>ref.r.prospectr</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.641 ms</td><td class="ms">1.719 ms</td><td class="ms">7.250 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
