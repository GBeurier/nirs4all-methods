# `rng_pcg64` — PCG64 RNG

_Group_: **Utility** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/rng_pcg64.md`](../algorithms/rng_pcg64.md)

## Description

`rng_pcg64` is a chemometrics4all utility method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

1. Melissa O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014). [pcg-random.org](https://www.pcg-random.org/paper.html)
2. Marsaglia, G. & Tsang, W. W., "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8) (2000).
3. F. R. Brown, "Random Number Generation with Arbitrary Strides", *Transactions of the American Nuclear Society* 71 (1994). (Jump-ahead algorithm.)
4. NumPy random module source: [github.com/numpy/numpy/tree/v1.26.4/numpy/random](https://github.com/numpy/numpy/tree/v1.26.4/numpy/random)

### Mathematical principle

`rng_pcg64` follows the standard utility operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng, uint64_t delta);
c4a_status_t c4a_rng_pcg64_create(uint64_t seed, c4a_rng_pcg64_state_t** out);
void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);
c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng, uint64_t seed);
c4a_status_t c4a_rng_pcg64_standard_normal_fill( c4a_rng_pcg64_state_t* rng, double* out, size_t n);
c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng, uint64_t* out);
c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng, uint64_t* out, size_t n);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_rng_pcg64` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.rng_pcg64` | Python | ABI-close function backed by ctypes. |
| ref.numpy | `numpy.random.default_rng(PCG64).standard_normal` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng, uint64_t delta);
c4a_status_t c4a_rng_pcg64_create(uint64_t seed, c4a_rng_pcg64_state_t** out);
void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);
c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng, uint64_t seed);
c4a_status_t c4a_rng_pcg64_standard_normal_fill( c4a_rng_pcg64_state_t* rng, double* out, size_t n);
c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng, uint64_t* out);
c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng, uint64_t* out, size_t n);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.rng_pcg64(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.numpy`** (Python · canonical) — `numpy.random.default_rng(PCG64).standard_normal` · numpy 2.3.5
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.numpy` | `numpy.random.default_rng(PCG64).standard_normal` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms">0.007 ms</td><td class="ms">0.007 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.006 ms</td><td class="ms">0.007 ms</td><td class="ms">0.007 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.006 ms</td><td class="ms ms-best">🏆 0.006 ms</td><td class="ms ms-best">🏆 0.006 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): numpy.random.default_rng(PCG64).standard_normal · numpy 2.3.5 — canonical">◆</span><code>ref.numpy</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.010 ms</td><td class="ms">0.010 ms</td><td class="ms">0.009 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
