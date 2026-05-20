# `spxy` — SPXY

_Group_: **Splitter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/splitters.md`](../algorithms/splitters.md)

## Description

Phase 11 introduces the `n4m_split_*` ABI category — nine sample-partitioning
operators that produce `(train_idx[], test_idx[])` integer arrays.

| Operator                       | Public symbol prefix                            | Public type                                       |
|--------------------------------|-------------------------------------------------|---------------------------------------------------|
| KennardStone                   | `n4m_split_kennard_stone_*`                     | `n4m_split_kennard_stone_handle_t`                |
| SPXY                           | `n4m_split_spxy_*`                              | `n4m_split_spxy_handle_t`                         |
| SPXYFold                       | `n4m_split_spxy_fold_*`                         | `n4m_split_spxy_fold_handle_t`                    |
| SPXYGFold                      | `n4m_split_spxy_g_fold_*`                       | `n4m_split_spxy_g_fold_handle_t`                  |
| KMeans                         | `n4m_split_kmeans_*`                            | `n4m_split_kmeans_handle_t`                       |
| KBinsStratified                | `n4m_split_kbins_stratified_*`                  | `n4m_split_kbins_stratified_handle_t`             |
| BinnedStratifiedGroupKFold     | `n4m_split_binned_strat_group_kfold_*`          | `n4m_split_binned_strat_group_kfold_handle_t`     |
| SystematicCircular             | `n4m_split_systematic_circular_*`               | `n4m_split_systematic_circular_handle_t`          |
| SPlit (data twinning)          | `n4m_split_split_splitter_*`                    | `n4m_split_split_splitter_handle_t`               |

All operators share the result type:

```c
typedef struct n4m_split_result_t {
    int64_t* train_idx;    /* length n_train */
    int64_t  n_train;
    int64_t* test_idx;     /* length n_test  */
    int64_t  n_test;
    void*    _owner;       /* opaque allocation handle */
} n4m_split_result_t;

N4M_API void n4m_split_result_destroy(n4m_split_result_t* r);
```

Buffers are owned by the splitter side; the caller must invoke
`n4m_split_result_destroy` exactly once per filled result. Calling
`_destroy` on an empty result is safe.

From the `n4m.SPXYSplitter` Python wrapper docstring:

> SPXY (Sample set Partitioning based on X and Y) train/test split.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `test_size` | `float` | `0.25` | Fraction of samples assigned to the test set. |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.SPXYSplitter (Python).

### Mathematical principle

`spxy` follows the standard splitter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_split_spxy_create(n4m_split_spxy_handle_t** out, double test_size);
void n4m_split_spxy_destroy(n4m_split_spxy_handle_t* h);
n4m_status_t n4m_split_spxy_fold_create(n4m_split_spxy_fold_handle_t** out, int32_t n_splits, int32_t y_metric);
void n4m_split_spxy_fold_destroy(n4m_split_spxy_fold_handle_t* h);
n4m_status_t n4m_split_spxy_fold_n_splits( const n4m_split_spxy_fold_handle_t* h, int32_t* out);
n4m_status_t n4m_split_spxy_fold_split_fold( const n4m_split_spxy_fold_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t Y, int32_t fold_idx, n4m_split_result_t* out);
n4m_status_t n4m_split_spxy_g_fold_create(n4m_split_spxy_g_fold_handle_t** out, int32_t n_splits, int32_t y_metric, int32_t aggregation);
void n4m_split_spxy_g_fold_destroy(n4m_split_spxy_g_fold_handle_t* h);
n4m_status_t n4m_split_spxy_g_fold_n_splits( const n4m_split_spxy_g_fold_handle_t* h, int32_t* out);
n4m_status_t n4m_split_spxy_g_fold_split_fold( const n4m_split_spxy_g_fold_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t Y, const int64_t* groups, int64_t groups_len, int32_t fold_idx, n4m_split_result_t* out);
n4m_status_t n4m_split_spxy_split(const n4m_split_spxy_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t Y, n4m_split_result_t* out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_split_spxy` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.spxy` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.SPXYSplitter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `spxy(X, Y, test_size = 0.25)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.SPXYSplitter` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_split_spxy_create(n4m_split_spxy_handle_t** out, double test_size);
void n4m_split_spxy_destroy(n4m_split_spxy_handle_t* h);
n4m_status_t n4m_split_spxy_fold_create(n4m_split_spxy_fold_handle_t** out, int32_t n_splits, int32_t y_metric);
void n4m_split_spxy_fold_destroy(n4m_split_spxy_fold_handle_t* h);
n4m_status_t n4m_split_spxy_fold_n_splits( const n4m_split_spxy_fold_handle_t* h, int32_t* out);
n4m_status_t n4m_split_spxy_fold_split_fold( const n4m_split_spxy_fold_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t Y, int32_t fold_idx, n4m_split_result_t* out);
n4m_status_t n4m_split_spxy_g_fold_create(n4m_split_spxy_g_fold_handle_t** out, int32_t n_splits, int32_t y_metric, int32_t aggregation);
void n4m_split_spxy_g_fold_destroy(n4m_split_spxy_g_fold_handle_t* h);
n4m_status_t n4m_split_spxy_g_fold_n_splits( const n4m_split_spxy_g_fold_handle_t* h, int32_t* out);
n4m_status_t n4m_split_spxy_g_fold_split_fold( const n4m_split_spxy_g_fold_handle_t* h, n4m_matrix_view_t X, n4m_matrix_view_t Y, const int64_t* groups, int64_t groups_len, int32_t fold_idx, n4m_split_result_t* out);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

train_idx, test_idx = n4m.spxy(X, y)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import SPXYSplitter

splitter = SPXYSplitter(test_size=0.25)
train_idx, test_idx = splitter.split(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- spxy(X, matrix(y, ncol = 1))
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.SPXYSplitter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.SPXYSplitter` | Python / parity | `default_allclose` |  |

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Divergence is the worst finite value over the visible sizes for each backend, preferring reference max-abs difference and falling back to binding max-abs difference when no reference comparison is recorded. Rows without a recorded comparison show `—`; the fastest backend per column is marked 🏆.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Divergence</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libn4m</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.127 ms</td><td class="ms">1.043 ms</td><td class="ms">5.500 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.121 ms</td><td class="ms">1.089 ms</td><td class="ms">5.504 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.125 ms</td><td class="ms ms-best">🏆 1.043 ms</td><td class="ms ms-best">🏆 5.379 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.154 ms</td><td class="ms">1.375 ms</td><td class="ms">6.000 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.SPXYSplitter · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.710 ms</td><td class="ms">1.794 ms</td><td class="ms">6.387 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
