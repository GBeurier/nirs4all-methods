# `aug_scatter_sim` — Scatter simulation MSC

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_scatter_sim.md`](../algorithms/aug_scatter_sim.md)

## Description

From the `chemometrics4all.ScatterSimulationMSC` Python wrapper docstring:

> MSC-style multiplicative/additive scatter simulation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `a_low` | `float` | `-0.05` |  |
| `a_high` | `float` | `0.05` |  |
| `b_low` | `float` | `0.9` |  |
| `b_high` | `float` | `1.1` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.ScatterSimulationMSC`.

### Mathematical principle

Simulates Multiplicative Scatter Correction (MSC) reversibility by drawing
a random additive offset and multiplicative gain per sample, then
applying

$$X^{\mathrm{aug}}_{i,:} = a_i + b_i \cdot X_{i,:}$$

with $a_i \sim \mathrm{Uniform}(a_{\mathrm{low}}, a_{\mathrm{high}})$ and
$b_i \sim \mathrm{Uniform}(b_{\mathrm{low}}, b_{\mathrm{high}})$.

The reference draws $n$ uniform `a` values followed by $n$ uniform `b`
values — the C engine mirrors this exact order so the PCG64 stream
consumption matches NumPy's vectorised `rng.uniform(low, high, size=n)`
calls.

### Implementation

When `a_low == a_high` and `b_low == b_high`, the augmenter is a
deterministic affine transform regardless of the RNG state. The parity
fixture exercises this case (`scatter_sim_constant` in
`aug_phase17_v1.json`).

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_scatter_sim_apply(const c4a_aug_scatter_sim_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_scatter_sim_create(c4a_aug_scatter_sim_handle_t** out, c4a_rng_pcg64_state_t* rng, double a_low, double a_high, double b_low, double b_high);
void c4a_aug_scatter_sim_destroy(c4a_aug_scatter_sim_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_scatter_sim` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_scatter_sim` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.ScatterSimulationMSC` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_scatter_sim(X, a_low = -0.05, a_high = 0.05, b_low = 0.9, b_high = 1.1, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.ScatterSimulationMSC` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_scatter_sim_apply(const c4a_aug_scatter_sim_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_scatter_sim_create(c4a_aug_scatter_sim_handle_t** out, c4a_rng_pcg64_state_t* rng, double a_low, double a_high, double b_low, double b_high);
void c4a_aug_scatter_sim_destroy(c4a_aug_scatter_sim_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_scatter_sim(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import ScatterSimulationMSC

op = ScatterSimulationMSC(a_low=-0.05, a_high=0.05, b_low=0.9, b_high=1.1, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_scatter_sim(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.ScatterSimulationMSC` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.ScatterSimulationMSC` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.008 ms</td><td class="ms">0.019 ms</td><td class="ms ms-best">🏆 0.080 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.008 ms</td><td class="ms">0.020 ms</td><td class="ms">0.083 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.009 ms</td><td class="ms ms-best">🏆 0.018 ms</td><td class="ms">0.080 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.022 ms</td><td class="ms">0.172 ms</td><td class="ms">1.367 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.ScatterSimulationMSC · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.025 ms</td><td class="ms">0.090 ms</td><td class="ms">0.431 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
