# `kennard_stone` — Kennard-Stone

_Group_: **Splitter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/splitters.md`](../algorithms/splitters.md)

## Description

Phase 11 introduces the `c4a_split_*` ABI category — nine sample-partitioning
operators that produce `(train_idx[], test_idx[])` integer arrays.

| Operator                       | Public symbol prefix                            | Public type                                       |
|--------------------------------|-------------------------------------------------|---------------------------------------------------|
| KennardStone                   | `c4a_split_kennard_stone_*`                     | `c4a_split_kennard_stone_handle_t`                |
| SPXY                           | `c4a_split_spxy_*`                              | `c4a_split_spxy_handle_t`                         |
| SPXYFold                       | `c4a_split_spxy_fold_*`                         | `c4a_split_spxy_fold_handle_t`                    |
| SPXYGFold                      | `c4a_split_spxy_g_fold_*`                       | `c4a_split_spxy_g_fold_handle_t`                  |
| KMeans                         | `c4a_split_kmeans_*`                            | `c4a_split_kmeans_handle_t`                       |
| KBinsStratified                | `c4a_split_kbins_stratified_*`                  | `c4a_split_kbins_stratified_handle_t`             |
| BinnedStratifiedGroupKFold     | `c4a_split_binned_strat_group_kfold_*`          | `c4a_split_binned_strat_group_kfold_handle_t`     |
| SystematicCircular             | `c4a_split_systematic_circular_*`               | `c4a_split_systematic_circular_handle_t`          |
| SPlit (data twinning)          | `c4a_split_split_splitter_*`                    | `c4a_split_split_splitter_handle_t`               |

All operators share the result type:

```c
typedef struct c4a_split_result_t {
    int64_t* train_idx;    /* length n_train */
    int64_t  n_train;
    int64_t* test_idx;     /* length n_test  */
    int64_t  n_test;
    void*    _owner;       /* opaque allocation handle */
} c4a_split_result_t;

C4A_API void c4a_split_result_destroy(c4a_split_result_t* r);
```

Buffers are owned by the splitter side; the caller must invoke
`c4a_split_result_destroy` exactly once per filled result. Calling
`_destroy` on an empty result is safe.

From the `chemometrics4all.KennardStoneSplitter` Python wrapper docstring:

> Kennard-Stone train/test split.

<details><summary>Full wrapper docstring</summary>

```text
Kennard-Stone train/test split.

Picks the most diverse samples for the training set, in descending order
of pairwise Euclidean distance.
```

</details>

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `test_size` | `float` | `0.25` | Fraction of samples assigned to the test set. |

## Explanations

### Bibliographic source

- `ref.nirs4all` — nirs4all.KennardStoneSplitter (Python).

### Mathematical principle

`kennard_stone` follows the standard splitter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_split_kennard_stone_create( c4a_split_kennard_stone_handle_t** out, double test_size);
void c4a_split_kennard_stone_destroy(c4a_split_kennard_stone_handle_t* h);
c4a_status_t c4a_split_kennard_stone_split( const c4a_split_kennard_stone_handle_t* h, c4a_matrix_view_t X, c4a_split_result_t* out);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_split_kennard_stone` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.kennard_stone` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.KennardStoneSplitter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `kennard_stone(X, test_size = 0.25)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.KennardStoneSplitter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_split_kennard_stone_create( c4a_split_kennard_stone_handle_t** out, double test_size);
void c4a_split_kennard_stone_destroy(c4a_split_kennard_stone_handle_t* h);
c4a_status_t c4a_split_kennard_stone_split( const c4a_split_kennard_stone_handle_t* h, c4a_matrix_view_t X, c4a_split_result_t* out);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

train_idx, test_idx = c4a.kennard_stone(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import KennardStoneSplitter

splitter = KennardStoneSplitter(test_size=0.25)
train_idx, test_idx = splitter.split(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- kennard_stone(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.KennardStoneSplitter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `split` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `80×32` · seed `1234567896`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `coverage_equal`, `deterministic`, `indices_equal`
- Truth sources:
  - `nirs4all_kennard_stone` — nirs4all.operators.split.kennard_stone (nirs4all, git-pinned by benchmark environment)

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.KennardStoneSplitter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.089 ms</td><td class="ms ms-best">🏆 0.975 ms</td><td class="ms ms-best">🏆 5.275 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.097 ms</td><td class="ms">1.044 ms</td><td class="ms">5.311 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.097 ms</td><td class="ms">1.031 ms</td><td class="ms">5.335 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.113 ms</td><td class="ms">1.227 ms</td><td class="ms">6.125 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.KennardStoneSplitter · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.660 ms</td><td class="ms">1.750 ms</td><td class="ms">6.125 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
