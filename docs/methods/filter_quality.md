# `filter_quality` — Spectral Quality Filter

_Group_: **Sample / feature filters** · _Binding_: `n4m.sklearn.SpectralQualityFilter` · _C ABI_: `n4m_filter_quality_*`

## Description

Stateless row-level spectrum quality filter.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `max_nan_ratio` | `float` | `0.1` |
| `max_zero_ratio` | `float` | `0.5` |
| `min_variance` | `float` | `1e-08` |
| `max_value` | `float | None` | `None` |
| `min_value` | `float | None` | `None` |
| `check_inf` | `bool` | `True` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Stateless row-level spectrum quality filter.

### Implementation

C ABI `n4m_filter_quality_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.SpectralQualityFilter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import SpectralQualityFilter
op = SpectralQualityFilter()
X_transformed = op.fit_transform(X)
```

### Benchmarks

Adaptive wall-clock per cell measured against [`full_matrix.csv`](../benchmarks/overview.md). Only backends that implement this method are listed; libraries without the method are omitted.

**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a reference-gate pass at strict / relaxed / qualitative tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees with the C++ baseline &nbsp;·&nbsp; ✗ divergent &nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The fastest backend per column is marked 🏆.

**Reference gate**: strict — numeric equivalence (`rmse_rel_tol ≤ 1e-12`).

::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th scope="col">Backend</th><th scope="col">Parity</th><th class="size-col" scope="col">50×250 (ms)</th><th class="size-col" scope="col">250×50 (ms)</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>pls4all.cpp.blas+omp</code></td><td class="parity parity-ref-strict">✓ ref</td><td class="ms">—</td><td class="ms">—</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="4" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row"><td class="bk-name"><code>nirs4all</code></td><td class="parity parity-ref-source">source</td><td class="ms">—</td><td class="ms">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)