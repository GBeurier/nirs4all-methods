# `aug_rotate_translate` — Rotate/translate augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_edge_spline_random.md`](../algorithms/aug_edge_spline_random.md)

## Description

Twelve data-augmentation operators in the unified `n4m_aug_*` ABI category.

All twelve operators share the universal ABI shape:

```c
typedef struct n4m_aug_<NAME>_handle_t n4m_aug_<NAME>_handle_t;

n4m_status_t n4m_aug_<NAME>_create(
    n4m_aug_<NAME>_handle_t** out,
    n4m_rng_pcg64_state_t* rng,   /* MUST be non-NULL */
    /* operator-specific parameters */);

n4m_status_t n4m_aug_<NAME>_apply(
    const n4m_aug_<NAME>_handle_t* handle,
    n4m_matrix_view_t X,
    [n4m_matrix_view_t wavelengths,]   /* only for wavelength-aware ops */
    n4m_matrix_view_t out);

void n4m_aug_<NAME>_destroy(n4m_aug_<NAME>_handle_t* handle);
```

The RNG handle is stored by reference; the augmenter does NOT own it. The
caller MUST keep the RNG alive for the augmenter's lifetime. With a fixed
RNG state at `_apply` time the output is bit-exact reproducible.

From the `n4m.RotateTranslateAugmenter` Python wrapper docstring:

> Random rotate/translate spectral augmentation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `p_range` | `float` | `2.0` |  |
| `y_factor` | `float` | `3.0` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

- nirs4all 0.8.x —
  `nirs4all.operators.augmentation.edge_artifacts`,
  `nirs4all.operators.augmentation.splines`,
  `nirs4all.operators.augmentation.random`.
- `roadmap/phase-15-18-augmenters-abi-contract.md`
- `DEFERRALS.md`

### Mathematical principle

`aug_rotate_translate` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- All 10 fully-implemented operators are deterministic: re-seeding the
  RNG handle before two consecutive `_apply` calls yields bit-equal
  outputs. The smoke fixtures under `parity/fixtures/aug_*_v1.json`
  exercise the shape + finite-value path; bit-exact NumPy parity of
  `rng.uniform`, `rng.choice` is gated behind v2 of the augmenter ABI.
- The 2 v2-deferred operators (`Spline_X_Simplification`,
  `Spline_Curve_Simplification`) return `N4M_ERR_NOT_IMPLEMENTED` from
  `_apply` until v2; their `_create` / `_destroy` symbols are stable.

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_aug_rotate_translate_apply( const n4m_aug_rotate_translate_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_rotate_translate_create( n4m_aug_rotate_translate_handle_t** out, n4m_rng_pcg64_state_t* rng, double p_range, double y_factor);
void n4m_aug_rotate_translate_destroy( n4m_aug_rotate_translate_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_aug_rotate_translate` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.aug_rotate_translate` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `n4m.sklearn.RotateTranslateAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_rotate_translate(X, p_range = 2.0, y_factor = 3.0, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.Rotate_Translate` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_aug_rotate_translate_apply( const n4m_aug_rotate_translate_handle_t* handle, n4m_matrix_view_t X, n4m_matrix_view_t out);
n4m_status_t n4m_aug_rotate_translate_create( n4m_aug_rotate_translate_handle_t** out, n4m_rng_pcg64_state_t* rng, double p_range, double y_factor);
void n4m_aug_rotate_translate_destroy( n4m_aug_rotate_translate_handle_t* handle);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.aug_rotate_translate(X)
```

:::

:::{tab-item} Python sklearn · n4m.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from n4m.sklearn import RotateTranslateAugmenter

op = RotateTranslateAugmenter(p_range=2.0, y_factor=3.0, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · nirs4all-methods
:sync: r
:class-label: lang-r

```r
library(n4m)
res <- aug_rotate_translate(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.Rotate_Translate` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.Rotate_Translate` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.015 ms</td><td class="ms">0.093 ms</td><td class="ms ms-best">🏆 0.426 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.014 ms</td><td class="ms ms-best">🏆 0.092 ms</td><td class="ms">0.449 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.016 ms</td><td class="ms">0.095 ms</td><td class="ms">0.435 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.030 ms</td><td class="ms">0.221 ms</td><td class="ms">1.734 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.Rotate_Translate · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.105 ms</td><td class="ms">0.397 ms</td><td class="ms">1.996 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
