# `detrend` — Detrend

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/detrend.md`](../algorithms/detrend.md)

## Description

Per-row polynomial detrending. For each row of `X` of length `p`:

1. Build positions `t = [0, 1, …, p-1]`.
2. Fit a polynomial of degree `polyorder` to `(t, X[i, :])` via least-squares (Householder QR).
3. Evaluate the fitted polynomial at the same positions.
4. Subtract: `out[i, j] = X[i, j] - polyval(coefs, t)[j]`.

From the `chemometrics4all.Detrend` Python wrapper docstring:

> Polynomial baseline subtraction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `polyorder` | `int` | `1` | Polynomial order. |

## Explanations

### Bibliographic source

- Standard polynomial detrending; see `numpy.polyfit` / `numpy.polyval` documentation.

### Mathematical principle

`detrend` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Closed-form least-squares via the shared Householder QR helper in `core/common/linalg.{c,h}`.
- True per-element division and subtraction; no reciprocal-multiplication shortcut.
- Parity tolerance vs frozen NumPy reference: `1e-11 abs / 1e-12 rel`.
- Reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/detrend.py` (uses `np.polyfit`/`np.polyval`).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out, int32_t polyorder);
void c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* handle);
c4a_status_t c4a_pp_detrend_transform( const c4a_pp_detrend_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_detrend` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.Detrend` | Python | sklearn-style wrapper backed by ctypes. |
| R | `detrend(X, polyorder = 1L)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.Detrend` | Python | canonical/comparator |
| ref.scipy | `scipy.signal.detrend` | Python | canonical/comparator |
| ref.r.stats | `R stats` | R | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out, int32_t polyorder);
void c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* handle);
c4a_status_t c4a_pp_detrend_transform( const c4a_pp_detrend_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import Detrend

op = Detrend(polyorder=1)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- detrend(X, polyorder = 1L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.Detrend` · nirs4all@cd731a23+dirty
- ◆ **`ref.scipy`** (Python · comparator) — `scipy.signal.detrend` · scipy 1.17.1
- ◆ **`ref.r.stats`** (R · comparator) — `R stats`
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.012 ms</td><td class="ms ms-best">🏆 0.112 ms</td><td class="ms ms-best">🏆 0.604 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.017 ms</td><td class="ms">0.117 ms</td><td class="ms">0.645 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.032 ms</td><td class="ms">0.326 ms</td><td class="ms">1.938 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Detrend · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.083 ms</td><td class="ms">0.331 ms</td><td class="ms">3.438 ms</td></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): scipy.signal.detrend · scipy 1.17.1 — comparator">◆</span><code>ref.scipy</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.071 ms</td><td class="ms">0.278 ms</td><td class="ms">3.032 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (R): R stats · R stats — comparator">◆</span><code>ref.r.stats</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">1.547 ms</td><td class="ms">3.219 ms</td><td class="ms">13.250 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
