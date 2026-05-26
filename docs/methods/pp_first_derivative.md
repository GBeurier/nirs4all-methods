# `pp_first_derivative` — First derivative

_Group_: **Preprocessing** · _Binding_: `n4m.sklearn.FirstDerivative` · _C ABI_: `n4m_pp_first_derivative_*`

## Description

``np.gradient(X, delta, axis=1, edge_order=...)`` (shape-preserving).

### Parameters

| Name | Type | Default |
|------|------|---------|
| `delta` | `float` | `1.0` |
| `edge_order` | `int` | `2` |

## Explanations

### Bibliographic source

Standard finite-difference / gap derivative; see Savitzky & Golay (1964) and Norris & Williams (1984).

### Mathematical principle

Approximates $\mathrm{d}\mathbf{x}/\mathrm{d}\lambda$ by finite differences. The first derivative removes constant baseline offsets (additive scatter) and accentuates inflection points of overlapping bands.

### Implementation

C ABI `n4m_pp_first_derivative_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.FirstDerivative`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import FirstDerivative
op = FirstDerivative()
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
<tr class="bk-row"><td class="bk-name"><code>ref.python_numpy</code></td><td class="parity parity-ref-source">source</td><td class="ms">—</td><td class="ms">—</td></tr>
</tbody>
</table>
</div>

:::

::::


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)