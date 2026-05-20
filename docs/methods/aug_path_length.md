# `aug_path_length` — Path length augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_path_length.md`](../algorithms/aug_path_length.md)

## Description

Applies a per-sample multiplicative path-length factor (with a lower clip to prevent sign inversion).

Algorithm:

1. Draw `n_samples` standard normals in one bulk call.
2. `L[i] = 1.0 + path_length_std * normal[i]`.
3. `L[i] = max(L[i], min_path_length)`.
4. `out[i, j] = L[i] * X[i, j]`.

Mirrors `nirs4all.operators.augmentation.synthesis.PathLengthAugmenter` with `variation_scope="sample"` (default).

From the `chemometrics4all.PathLengthAugmenter` Python wrapper docstring:

> Simulate multiplicative path-length variation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `path_length_std` | `float` | `0.05` |  |
| `min_path_length` | `float` | `0.1` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- Internal parity fixture: `parity/python_generator/src/c4a_parity_augmenters_ref/path_length.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_path_length` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_path_length_apply( const c4a_aug_path_length_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_path_length_create( c4a_aug_path_length_handle_t** out, c4a_rng_pcg64_state_t* rng, double path_length_std, double min_path_length);
void c4a_aug_path_length_destroy(c4a_aug_path_length_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_path_length` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_path_length` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.PathLengthAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_path_length(X, path_length_std = 0.05, min_path_length = 0.1, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.PathLengthAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_path_length_apply( const c4a_aug_path_length_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_path_length_create( c4a_aug_path_length_handle_t** out, c4a_rng_pcg64_state_t* rng, double path_length_std, double min_path_length);
void c4a_aug_path_length_destroy(c4a_aug_path_length_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_path_length(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import PathLengthAugmenter

op = PathLengthAugmenter(path_length_std=0.05, min_path_length=0.1, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_path_length(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.PathLengthAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.PathLengthAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.007 ms</td><td class="ms">0.018 ms</td><td class="ms">0.090 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.007 ms</td><td class="ms ms-best">🏆 0.016 ms</td><td class="ms">0.074 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.008 ms</td><td class="ms">0.018 ms</td><td class="ms ms-best">🏆 0.073 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.022 ms</td><td class="ms">0.180 ms</td><td class="ms">1.414 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.PathLengthAugmenter · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.022 ms</td><td class="ms">0.057 ms</td><td class="ms">0.293 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
