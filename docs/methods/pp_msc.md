# `pp_msc` — Multiplicative Scatter Correction (MSC)

_Group_: **Preprocessing** · _Binding_: `n4m.sklearn.MSC` · _C ABI_: `n4m_pp_msc_*`

## Description

Multiplicative Scatter Correction.

<details>
<summary>Full binding docstring</summary>

```text
Multiplicative Scatter Correction.

Fit learns the mean reference spectrum from the training matrix. Transform
regresses each row against that reference and applies the conventional
row-wise MSC correction used by prospectr and pls.
```
</details>

### Parameters

_No constructor parameters._

## Explanations

### Bibliographic source

Geladi, P., MacDougall, D. & Martens, H. (1985). *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat*. Applied Spectroscopy 39(3), 491–500.

### Mathematical principle

Each spectrum is regressed on a reference spectrum (typically the training-set mean $\bar{\mathbf{x}}$): $\mathbf{x}_i = a_i + b_i\,\bar{\mathbf{x}} + \mathbf{e}_i$. The corrected spectrum $(\mathbf{x}_i - a_i)/b_i$ removes the additive ($a_i$) and multiplicative ($b_i$) scatter estimated by ordinary least squares. Unlike SNV, MSC needs a reference and so is stateful (the reference is fit on training data).

### Implementation

C ABI `n4m_pp_msc_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.MSC`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import MSC
op = MSC()
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