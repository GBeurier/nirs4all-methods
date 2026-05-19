# `kbins_discretizer` — K-bins discretizer

_Group_: **Resampling** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `chemometrics4all.IntegerKBinsDiscretizer` Python wrapper docstring:

> Per-column integer binning using uniform or quantile edges.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_bins` | `int` | `5` | Number of bins. |
| `strategy` | `str \| int` | `0` | Binning strategy. |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.IntegerKBinsDiscretizer (Python).
- `ref.sklearn` — sklearn.preprocessing.KBinsDiscretizer (Python).

### Mathematical principle

`kbins_discretizer` follows the standard resampling operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_kbins_disc_create(c4a_pp_kbins_disc_handle_t** out, int32_t n_bins, int32_t strategy);
void c4a_pp_kbins_disc_destroy(c4a_pp_kbins_disc_handle_t* h);
c4a_status_t c4a_pp_kbins_disc_fit(c4a_pp_kbins_disc_handle_t* h, c4a_matrix_view_t X);
c4a_status_t c4a_pp_kbins_disc_is_fitted( const c4a_pp_kbins_disc_handle_t* h, int* out_fitted);
c4a_status_t c4a_pp_kbins_disc_transform( const c4a_pp_kbins_disc_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_kbins_disc` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.IntegerKBinsDiscretizer` | Python | sklearn-style wrapper backed by ctypes. |
| R | `kbins_discretizer(X, n_bins = 5L, strategy = "uniform")` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.IntegerKBinsDiscretizer` | Python | canonical/comparator |
| ref.sklearn | `sklearn.preprocessing.KBinsDiscretizer` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_kbins_disc_create(c4a_pp_kbins_disc_handle_t** out, int32_t n_bins, int32_t strategy);
void c4a_pp_kbins_disc_destroy(c4a_pp_kbins_disc_handle_t* h);
c4a_status_t c4a_pp_kbins_disc_fit(c4a_pp_kbins_disc_handle_t* h, c4a_matrix_view_t X);
c4a_status_t c4a_pp_kbins_disc_is_fitted( const c4a_pp_kbins_disc_handle_t* h, int* out_fitted);
c4a_status_t c4a_pp_kbins_disc_transform( const c4a_pp_kbins_disc_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import IntegerKBinsDiscretizer

op = IntegerKBinsDiscretizer(n_bins=5, strategy=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- kbins_discretizer(X, n_bins = 5L, strategy = 'uniform')
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.IntegerKBinsDiscretizer` · nirs4all@cd731a23+dirty
- 📐 **`ref.sklearn`** (Python · comparator) — `sklearn.preprocessing.KBinsDiscretizer` · sklearn 1.8.0
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.027 ms</td><td class="ms ms-best">🏆 0.305 ms</td><td class="ms ms-best">🏆 1.467 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.029 ms</td><td class="ms">0.306 ms</td><td class="ms">1.549 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.069 ms</td><td class="ms">0.555 ms</td><td class="ms">3.188 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.IntegerKBinsDiscretizer · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.499 ms</td><td class="ms">3.312 ms</td><td class="ms">16.108 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): sklearn.preprocessing.KBinsDiscretizer · sklearn 1.8.0 — comparator">📐</span><code>ref.sklearn</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.436 ms</td><td class="ms">3.266 ms</td><td class="ms">16.391 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
