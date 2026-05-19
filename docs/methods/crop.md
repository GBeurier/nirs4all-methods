# `crop` — Crop

_Group_: **Resampling** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8`

## Description

From the `chemometrics4all.CropTransformer` Python wrapper docstring:

> Slice wavelength columns in the half-open interval ``[start, end)``.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `start` | `int` | `required` |  |
| `end` | `int` | `required` |  |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.CropTransformer (Python).

### Mathematical principle

`crop` follows the standard resampling operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_crop_create(c4a_pp_crop_handle_t** out, int64_t start, int64_t end);
void c4a_pp_crop_destroy(c4a_pp_crop_handle_t* h);
int64_t c4a_pp_crop_output_cols(const c4a_pp_crop_handle_t* h, int64_t input_cols);
c4a_status_t c4a_pp_crop_transform(const c4a_pp_crop_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_crop` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.CropTransformer` | Python | sklearn-style wrapper backed by ctypes. |
| R | `{start <- ncol(X) %/% 8L; end <- ncol(X) - ncol(X) %/% 8L; crop_transform(X, start = start, end = end)}` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.CropTransformer` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_crop_create(c4a_pp_crop_handle_t** out, int64_t start, int64_t end);
void c4a_pp_crop_destroy(c4a_pp_crop_handle_t* h);
int64_t c4a_pp_crop_output_cols(const c4a_pp_crop_handle_t* h, int64_t input_cols);
c4a_status_t c4a_pp_crop_transform(const c4a_pp_crop_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import CropTransformer

op = CropTransformer()
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- {start <- ncol(X) %/% 8L; end <- ncol(X) - ncol(X) %/% 8L; crop_transform(X, start = start, end = end)}
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.CropTransformer` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.001 ms</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms ms-best">🏆 0.056 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.012 ms</td><td class="ms">0.013 ms</td><td class="ms">0.064 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.023 ms</td><td class="ms">0.213 ms</td><td class="ms">1.422 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.CropTransformer · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.003 ms</td><td class="ms">0.011 ms</td><td class="ms">0.064 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
