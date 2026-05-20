# `from_absorbance` — From absorbance

_Group_: **Preprocessing** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/from_absorbance.md`](../algorithms/from_absorbance.md)

## Description

From the `chemometrics4all.FromAbsorbance` Python wrapper docstring:

> R = 10**(-A), optionally returned as percent.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `is_percent` | `bool` | `False` |  |

## Explanations

### Bibliographic source

Implementation matches `nirs4all.operators.transforms.signal_conversion.FromAbsorbance` (0.8.11).

### Mathematical principle

Element-wise inverse log-10: convert absorbance back to reflectance or transmittance.

$$
y_{i,j} = 10^{-x_{i,j}}, \qquad
\text{out}_{i,j} = \begin{cases} 100 \cdot y_{i,j} & \text{if `is\_percent`} \\ y_{i,j} & \text{otherwise} \end{cases}
$$

### Implementation

* The power is computed as `pow(10.0, -x)` — element-wise `np.power(10.0, -X)` semantics. The alternative `exp(-x · ln10)` introduces one extra multiply rounding step that would diverge in the last bit.
* The percent multiply uses `* 100.0` as a separate inline operation (not folded into the exponent), mirroring `result = result * 100.0` in nirs4all.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_from_absorbance_create( c4a_pp_from_absorbance_handle_t** out, int is_percent);
void c4a_pp_from_absorbance_destroy( c4a_pp_from_absorbance_handle_t* handle);
c4a_status_t c4a_pp_from_absorbance_transform( const c4a_pp_from_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_from_absorbance` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.from_absorbance` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.FromAbsorbance` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `from_absorbance(X, is_percent = FALSE)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.FromAbsorbance` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_from_absorbance_create( c4a_pp_from_absorbance_handle_t** out, int is_percent);
void c4a_pp_from_absorbance_destroy( c4a_pp_from_absorbance_handle_t* handle);
c4a_status_t c4a_pp_from_absorbance_transform( const c4a_pp_from_absorbance_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.from_absorbance(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import FromAbsorbance

op = FromAbsorbance(is_percent=False)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- from_absorbance(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.FromAbsorbance` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.FromAbsorbance` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.017 ms</td><td class="ms ms-best">🏆 0.164 ms</td><td class="ms">0.828 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.022 ms</td><td class="ms">0.173 ms</td><td class="ms">0.821 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.025 ms</td><td class="ms">0.173 ms</td><td class="ms ms-best">🏆 0.806 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">6.1e-16</td><td class="ms">0.039 ms</td><td class="ms">0.473 ms</td><td class="ms">2.453 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.FromAbsorbance · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.048 ms</td><td class="ms">0.440 ms</td><td class="ms">2.401 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
