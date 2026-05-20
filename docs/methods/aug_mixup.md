# `aug_mixup` — Mixup augmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_mixup.md`](../algorithms/aug_mixup.md)

## Description

From the `chemometrics4all.MixupAugmenter` Python wrapper docstring:

> Batch-wise mixup augmentation.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `alpha` | `float` | `0.2` |  |
| `rng` | `Optional[PCG64]` | `None` |  |
| `seed` | `int` | `0` | Random seed for deterministic splitting or filtering. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.MixupAugmenter`. The reference
implementation uses `numpy.random.Generator.permutation` followed by
`Generator.beta(alpha, alpha, size=(n,1))`.

For a recent overview of mixup, see Zhang et al. 2018, "mixup: Beyond
Empirical Risk Minimization".

### Mathematical principle

`c4a_aug_mixup` augments a batch of spectra by replacing every row with a
convex combination of two original rows drawn from the same batch.

Given an input matrix $X \in \mathbb{R}^{n \times p}$, the algorithm:

1. Draws a Fisher-Yates permutation $\pi \in S_n$ of the sample indices.
2. Draws $\lambda_i \sim \mathrm{Beta}(\alpha, \alpha)$ for $i = 0, \dots, n-1$.
3. Writes
$$X^{\mathrm{aug}}_{i,:} = \lambda_i \cdot X_{i,:} + (1-\lambda_i) \cdot X_{\pi(i),:}$$

The output keeps the same shape $(n, p)$ as the input. **Each output row is
a mix of two rows from the SAME batch** — this is the canonical "within-batch
mixup" used in vision and audio augmentation pipelines, applied here to NIR
spectra.

### Implementation

A fixed PCG64 state at call time yields a bit-exact output. Seeding the
RNG with `c4a_rng_pcg64_set_seed(rng, S)` before each `_apply` produces a
reproducible stream regardless of prior consumption.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_aug_mixup_apply(const c4a_aug_mixup_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_mixup_create(c4a_aug_mixup_handle_t** out, c4a_rng_pcg64_state_t* rng, double alpha);
void c4a_aug_mixup_destroy(c4a_aug_mixup_handle_t* handle);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_aug_mixup` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.python.aug_mixup` | Python | ABI-close function backed by ctypes. |
| Python sklearn | `chemometrics4all.sklearn.MixupAugmenter` | Python | scikit-learn-compatible estimator backed by ctypes. |
| R | `aug_mixup(X, alpha = 1.0, seed = 17)` | R | Public package wrapper around the C ABI. |
| ref.nirs4all | `nirs4all.MixupAugmenter` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_aug_mixup_apply(const c4a_aug_mixup_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
c4a_status_t c4a_aug_mixup_create(c4a_aug_mixup_handle_t** out, c4a_rng_pcg64_state_t* rng, double alpha);
void c4a_aug_mixup_destroy(c4a_aug_mixup_handle_t* handle);
```

:::

:::{tab-item} Python ABI · chemometrics4all.python
:sync: python-abi
:class-label: lang-python

```python
from chemometrics4all import python as c4a

Xt = c4a.aug_mixup(X)
```

:::

:::{tab-item} Python sklearn · chemometrics4all.sklearn
:sync: python-sklearn
:class-label: lang-python

```python
from chemometrics4all.sklearn import MixupAugmenter

op = MixupAugmenter(alpha=0.2, rng=None, seed=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- aug_mixup(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · context) — `nirs4all.MixupAugmenter` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.MixupAugmenter` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.010 ms</td><td class="ms ms-best">🏆 0.023 ms</td><td class="ms ms-best">🏆 0.096 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.python</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.010 ms</td><td class="ms">0.023 ms</td><td class="ms">0.109 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst binding max abs diff over visible sizes">0</td><td class="ms">0.012 ms</td><td class="ms">0.023 ms</td><td class="ms">0.105 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">5.6e-16</td><td class="ms">0.029 ms</td><td class="ms">0.170 ms</td><td class="ms">1.234 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-relaxed"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.MixupAugmenter · nirs4all@cd731a23+dirty — context">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-context" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.032 ms</td><td class="ms">0.135 ms</td><td class="ms">0.650 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
