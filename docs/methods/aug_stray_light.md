# `aug_stray_light` — Stray light augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_edge_spline_random.md`](../algorithms/aug_edge_spline_random.md)

## Description

Twelve data-augmentation operators in the unified `c4a_aug_*` ABI category.

All twelve operators share the universal ABI shape:

```c
typedef struct c4a_aug_<NAME>_handle_t c4a_aug_<NAME>_handle_t;

c4a_status_t c4a_aug_<NAME>_create(
    c4a_aug_<NAME>_handle_t** out,
    c4a_rng_pcg64_state_t* rng,   /* MUST be non-NULL */
    /* operator-specific parameters */);

c4a_status_t c4a_aug_<NAME>_apply(
    const c4a_aug_<NAME>_handle_t* handle,
    c4a_matrix_view_t X,
    [c4a_matrix_view_t wavelengths,]   /* only for wavelength-aware ops */
    c4a_matrix_view_t out);

void c4a_aug_<NAME>_destroy(c4a_aug_<NAME>_handle_t* handle);
```

The RNG handle is stored by reference; the augmenter does NOT own it. The
caller MUST keep the RNG alive for the augmenter's lifetime. With a fixed
RNG state at `_apply` time the output is bit-exact reproducible.

From the `chemometrics4all.StrayLightAugmenter` Python wrapper docstring:

> Stray-light edge artifact.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `stray_light_fraction` | `float` | `0.001` |  |
| `edge_enhancement` | `float` | `2.0` |  |
| `edge_width` | `float` | `0.1` |  |
| `include_peak_truncation` | `bool` | `True` |  |
| `wavelengths` | `object` | `None` |  |
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

`aug_stray_light` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- All 10 fully-implemented operators are deterministic: re-seeding the
  RNG handle before two consecutive `_apply` calls yields bit-equal
  outputs. The smoke fixtures under `parity/fixtures/aug_*_v1.json`
  exercise the shape + finite-value path; bit-exact NumPy parity of
  `rng.uniform`, `rng.choice` is gated behind v2 of the augmenter ABI.
- The 2 v2-deferred operators (`Spline_X_Simplification`,
  `Spline_Curve_Simplification`) return `C4A_ERR_NOT_IMPLEMENTED` from
  `_apply` until v2; their `_create` / `_destroy` symbols are stable.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_stray_light_apply( const c4a_aug_stray_light_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t wavelengths, c4a_matrix_view_t out);
c4a_status_t c4a_aug_stray_light_create( c4a_aug_stray_light_handle_t** out, c4a_rng_pcg64_state_t* rng, double stray_light_fraction, double edge_enhancement, double edge_width, int32_t include_peak_truncation);
void c4a_aug_stray_light_destroy( c4a_aug_stray_light_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_stray_light` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_stray_light` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.StrayLightAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_stray_light(X, wavelengths = NULL, stray_light_fraction = 0.001, edge_enhancement = 2.0, edge_width = 0.1, include_peak_truncation = 1L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.StrayLightAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_stray_light_apply( const c4a_aug_stray_light_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t wavelengths, c4a_matrix_view_t out);
c4a_status_t c4a_aug_stray_light_create( c4a_aug_stray_light_handle_t** out, c4a_rng_pcg64_state_t* rng, double stray_light_fraction, double edge_enhancement, double edge_width, int32_t include_peak_truncation);
void c4a_aug_stray_light_destroy( c4a_aug_stray_light_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_stray_light(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import StrayLightAugmenter

op = StrayLightAugmenter(stray_light_fraction=0.001, edge_enhancement=2.0, edge_width=0.1, include_peak_truncation=True, wavelengths=None, rng=None)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_stray_light(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.StrayLightAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.StrayLightAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.134 ms</td><td class="ms ms-best">🏆 1.248 ms</td><td class="ms">6.558 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms">0.134 ms</td><td class="ms">1.298 ms</td><td class="ms ms-best">🏆 6.100 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">1.1e-16</td><td class="ms ms-best">🏆 0.130 ms</td><td class="ms">1.267 ms</td><td class="ms">6.357 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.156 ms</td><td class="ms">1.703 ms</td><td class="ms">7.938 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.StrayLightAugmenter · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">1.044 ms</td><td class="ms">2.075 ms</td><td class="ms">7.165 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
