# `y_outlier_iqr` — Y outlier IQR

_Group_: **Filter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/y_outlier_filter.md`](../algorithms/y_outlier_filter.md)

## Description

First member of the `c4a_filter_*` ABI category (Phase 12). Identifies samples
whose 1-D target (`y`) values are statistical outliers using one of four
detection methods, and returns a per-sample keep/exclude mask together with a
stats record.

Mirrors `nirs4all.operators.filters.YOutlierFilter` at
`/home/delete/nirs4all/nirs4all/nirs4all/operators/filters/y_outlier.py`.

From the `chemometrics4all.YOutlierFilter` Python wrapper docstring:

> Univariate outlier filter on the target vector ``y``.

<details><summary>Full wrapper docstring</summary>

```text
Univariate outlier filter on the target vector ``y``.

``method`` is one of ``"iqr"``, ``"zscore"``, ``"percentile"``, ``"mad"``.
Threshold semantics follow nirs4all's :class:`YOutlierFilter`:

* For ``"iqr"`` / ``"zscore"`` / ``"mad"``: ``threshold`` is the
  multiplier on the IQR / σ / MAD bounds.
* For ``"percentile"``: ``lower_percentile`` and ``upper_percentile``
  define the keep band.
```

</details>

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `method` | `str` | `'iqr'` | Algorithm variant selected by the wrapper. |
| `threshold` | `float` | `1.5` | Outlier or quality threshold. |
| `lower_percentile` | `float` | `1.0` |  |
| `upper_percentile` | `float` | `99.0` |  |

## Explanations

### Bibliographic source

- `nirs4all.operators.filters.YOutlierFilter`
- See `roadmap/phase-12-y-filter.md` for the brief and acceptance criteria.

### Mathematical principle

`y_outlier_iqr` follows the standard filter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Bounds use NumPy linear-interpolation percentiles (closed-form).
- Z-score uses the biased (`ddof=0`) standard deviation.
- MAD uses the unbiased median of `|y - median(y)|`, scaled by `1.4826`.
- Parity tolerance vs the frozen NumPy reference: **exact** equality for the
  mask bytes and integer counts (`n_samples`, `n_kept`, `n_excluded`),
  **1e-12 abs** for the `exclusion_rate` (single division by `n`).
- Reference: `parity/python_generator/src/c4a_parity_filters_ref/y_outlier.py`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_filter_y_outlier_apply( const c4a_filter_y_outlier_handle_t* handle, const double* y, int64_t n, uint8_t* mask_out, c4a_filter_stats_t* stats_out);
c4a_status_t c4a_filter_y_outlier_create( c4a_filter_y_outlier_handle_t** out, c4a_y_outlier_method_t method, double threshold, double lower_pct, double upper_pct);
void c4a_filter_y_outlier_destroy( c4a_filter_y_outlier_handle_t* handle);
c4a_status_t c4a_filter_y_outlier_fit( c4a_filter_y_outlier_handle_t* handle, const double* y, int64_t n);
c4a_status_t c4a_filter_y_outlier_is_fitted( const c4a_filter_y_outlier_handle_t* handle, int* out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_filter_y_outlier` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.YOutlierFilter` | Python | sklearn-style wrapper backed by ctypes. |
| R | `y_outlier_filter(y, method = "iqr", threshold = NULL, lower_pct = 1.0, upper_pct = 99.0)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.YOutlierFilter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_filter_y_outlier_apply( const c4a_filter_y_outlier_handle_t* handle, const double* y, int64_t n, uint8_t* mask_out, c4a_filter_stats_t* stats_out);
c4a_status_t c4a_filter_y_outlier_create( c4a_filter_y_outlier_handle_t** out, c4a_y_outlier_method_t method, double threshold, double lower_pct, double upper_pct);
void c4a_filter_y_outlier_destroy( c4a_filter_y_outlier_handle_t* handle);
c4a_status_t c4a_filter_y_outlier_fit( c4a_filter_y_outlier_handle_t* handle, const double* y, int64_t n);
c4a_status_t c4a_filter_y_outlier_is_fitted( const c4a_filter_y_outlier_handle_t* handle, int* out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import YOutlierFilter

flt = YOutlierFilter(method='iqr', threshold=1.5, lower_percentile=1.0, upper_percentile=99.0)
mask, stats = flt.fit_apply(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- y_outlier_filter(y, method = 'iqr')$mask
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.nirs4all`** (Python · canonical) — `nirs4all.YOutlierFilter` · nirs4all@cd731a23+dirty
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">0.009 ms</td><td class="ms">0.009 ms</td><td class="ms">0.009 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.009 ms</td><td class="ms">0.010 ms</td><td class="ms">0.010 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 0.004 ms</td><td class="ms ms-best">🏆 0.004 ms</td><td class="ms ms-best">🏆 0.004 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.YOutlierFilter · nirs4all@cd731a23+dirty — canonical">📐</span><code>ref.nirs4all</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">0.073 ms</td><td class="ms">0.085 ms</td><td class="ms">0.146 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
