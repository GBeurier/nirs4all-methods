# `aug_local_mixup` — Local mixup augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_local_mixup.md`](../algorithms/aug_local_mixup.md)

## Description

From the `chemometrics4all.LocalMixupAugmenter` Python wrapper docstring:

> Neighbor-constrained mixup augmentation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `alpha` | `float` | `0.2` |  |
| `k_neighbors` | `int` | `5` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.LocalMixupAugmenter`. The reference
implementation uses `sklearn.neighbors.NearestNeighbors` with default
parameters (brute force, Euclidean distance), then
`rng.choice(indices[i, 1:])` and `rng.beta(alpha, alpha)`.

### Mathematical principle

`c4a_aug_local_mixup` is a neighborhood-aware variant of mixup. For each
sample, it draws a random neighbor among the $k$ nearest non-self rows and
mixes the two with a $\mathrm{Beta}(\alpha, \alpha)$ weight.

For each $i \in \{0, \dots, n-1\}$:

1. Compute the squared-Euclidean distance from $X_{i,:}$ to every row of
   $X$.
2. Partial-sort to obtain the $k+1$ smallest indices (the first one is
   $i$ itself).
3. Draw `pick ∈ [0, k)` uniformly via PCG64 integers and select
   `neighbor = indices[1 + pick]`.
4. Draw $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
5. Write $X^{\mathrm{aug}}_{i,:} = \lambda \cdot X_{i,:} + (1-\lambda) \cdot X_{\mathrm{neighbor},:}$.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_local_mixup_apply(const c4a_aug_local_mixup_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_local_mixup_create(c4a_aug_local_mixup_handle_t** out, c4a_rng_pcg64_state_t* rng, double alpha, int32_t k_neighbors);
void c4a_aug_local_mixup_destroy(c4a_aug_local_mixup_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_local_mixup` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_local_mixup` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.LocalMixupAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_local_mixup(X, alpha = 1.0, k_neighbors = 5L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.LocalMixupAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_local_mixup_apply(const c4a_aug_local_mixup_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_local_mixup_create(c4a_aug_local_mixup_handle_t** out, c4a_rng_pcg64_state_t* rng, double alpha, int32_t k_neighbors);
void c4a_aug_local_mixup_destroy(c4a_aug_local_mixup_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_local_mixup(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import LocalMixupAugmenter

op = LocalMixupAugmenter(alpha=0.2, k_neighbors=5, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_local_mixup(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.LocalMixupAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.LocalMixupAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.173 ms</td><td class="ms">2.011 ms</td><td class="ms">10.602 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.166 ms</td><td class="ms">1.928 ms</td><td class="ms">10.048 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.171 ms</td><td class="ms">1.906 ms</td><td class="ms">10.725 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.273 ms</td><td class="ms">2.219 ms</td><td class="ms">12.000 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.LocalMixupAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.821 ms</td><td class="ms ms-best">🏆 1.092 ms</td><td class="ms ms-best">🏆 2.309 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
