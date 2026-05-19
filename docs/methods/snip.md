# `snip` — SNIP

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/snip.md`](../algorithms/snip.md)

## Description

Ryan 1988 / Morháč 1997 baseline correction, implemented on the same public
contract as `pybaselines.smooth.snip` with `filter_order=2`.

For each row `y` of length `n`:

1. Clamp `max_half_window` to the largest valid half-window for the row.
2. Linearly extrapolate `max_half_window` samples at both edges.
3. For each `w = 1, 2, ..., max_half_window`, compute
   `filters[i] = (baseline[i - w] + baseline[i + w]) / 2` for the whole valid
   padded slice, then update `baseline[i] = min(baseline[i], filters[i])`.
4. Output `out = y - baseline[unpadded]`.

From the `chemometrics4all.SNIP` Python wrapper docstring:

> Statistics-sensitive nonlinear iterative peak-clipping baseline.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `max_half_window` | `int` | `20` |  |

## Explanations

### Bibliographic source

- Ryan, C. et al. (1988). "SNIP: A statistics-sensitive background treatment
  for the quantitative analysis of PIXE spectra in geoscience applications."
  Nuclear Instruments and Methods B34, 396-402.
- Morháč, M. et al. (1997). "Background elimination methods for
  multidimensional coincidence γ-ray spectra." Nuclear Instruments and Methods
  A 401(1), 113-132.
- Pybaselines SNIP reference: `pybaselines.smooth.snip`.

### Mathematical principle

`snip` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Pure arithmetic, row-wise.
- Edge handling and per-window update order match `pybaselines.smooth.snip`.
- External benchmark gate: `pybaselines.Baseline().snip`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_snip_create(c4a_pp_snip_handle_t** out, int32_t max_half_window);
void c4a_pp_snip_destroy(c4a_pp_snip_handle_t* handle);
c4a_status_t c4a_pp_snip_transform(const c4a_pp_snip_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_snip` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.SNIP` | Python | sklearn-style wrapper backed by ctypes. |
| R | `snip(X, max_half_window = 20L)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.snip(raw)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_snip_create(c4a_pp_snip_handle_t** out, int32_t max_half_window);
void c4a_pp_snip_destroy(c4a_pp_snip_handle_t* handle);
c4a_status_t c4a_pp_snip_transform(const c4a_pp_snip_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import SNIP

op = SNIP(max_half_window=20)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- snip(X, max_half_window = 10L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.snip(raw)` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">0.123 ms</td><td class="ms ms-best">🏆 1.458 ms</td><td class="ms ms-best">🏆 7.110 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.122 ms</td><td class="ms">1.581 ms</td><td class="ms">7.258 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 0.119 ms</td><td class="ms">1.562 ms</td><td class="ms">8.938 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.snip(raw) · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">19.872 ms</td><td class="ms">19.469 ms</td><td class="ms">25.489 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
