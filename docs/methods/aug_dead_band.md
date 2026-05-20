# `aug_dead_band` — Dead band augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_dead_band.md`](../algorithms/aug_dead_band.md)

## Description

From the `chemometrics4all.DeadBandAugmenter` Python wrapper docstring:

> Simulate dead spectral detector bands.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `n_bands` | `int` | `1` |  |
| `width_low` | `int` | `5` |  |
| `width_high` | `int` | `10` |  |
| `noise_std` | `float` | `0.05` |  |
| `probability` | `float` | `0.0` |  |
| `variation_scope` | `int` | `0` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.DeadBandAugmenter`.

### Mathematical principle

Simulates detector dead bands by replacing random contiguous spectral
regions with noise.

Per-sample mode (`variation_scope = 0`):

```
for each row i in 0..n-1:
    u ~ Uniform(0, 1)
    if u < probability:
        for b in 0..n_bands:
            width ~ UniformInt[width_low, width_high]   # inclusive
            start ~ UniformInt[0, max(1, cols - width))
            end   = min(start + width, cols)
            row[start:end] ~ Normal(0, noise_std)
```

Batch mode (`variation_scope = 1`):

```
u ~ Uniform(0, 1)
if u < probability:
    for b in 0..n_bands:
        width, start, end = ... (as above)
        # all rows get the same band positions, but
        # the noise samples are still drawn independently
        # with size = (rows, end - start)
        X[:, start:end] ~ Normal(0, noise_std)
```

### Implementation

When `probability = 0` the augmenter is the identity. The parity fixture
exercises this case (`dead_band_zero_probability`).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_dead_band_apply( const c4a_aug_dead_band_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_dead_band_create( c4a_aug_dead_band_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t n_bands, int32_t width_low, int32_t width_high, double noise_std, double probability, int32_t variation_scope);
void c4a_aug_dead_band_destroy( c4a_aug_dead_band_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_dead_band` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_dead_band` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.DeadBandAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_dead_band(X, n_bands = 1L, width_low = 5L, width_high = 10L, noise_std = 0.05, probability = 1.0, variation_scope = 0L, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.DeadBandAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_dead_band_apply( const c4a_aug_dead_band_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_dead_band_create( c4a_aug_dead_band_handle_t** out, c4a_rng_pcg64_state_t* rng, int32_t n_bands, int32_t width_low, int32_t width_high, double noise_std, double probability, int32_t variation_scope);
void c4a_aug_dead_band_destroy( c4a_aug_dead_band_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_dead_band(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import DeadBandAugmenter

op = DeadBandAugmenter(n_bands=1, width_low=5, width_high=10, noise_std=0.05, probability=0.0, variation_scope=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_dead_band(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.DeadBandAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.DeadBandAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.011 ms</td><td class="ms">0.020 ms</td><td class="ms">0.064 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.011 ms</td><td class="ms ms-best">🏆 0.019 ms</td><td class="ms ms-best">🏆 0.063 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.012 ms</td><td class="ms">0.021 ms</td><td class="ms">0.065 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.044 ms</td><td class="ms">0.220 ms</td><td class="ms">1.180 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.DeadBandAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.301 ms</td><td class="ms">0.311 ms</td><td class="ms">0.506 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
