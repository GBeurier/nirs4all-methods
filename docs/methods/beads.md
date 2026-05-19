# `beads` — BEADS

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/beads.md`](../algorithms/beads.md)

## Description

Ning & Selesnick 2014 BEADS, implemented against the
`pybaselines.Baseline().beads` full banded contract for the public c4a
parameter surface.

The exposed options are:

- `freq_cutoff = 0.005`
- `asymmetry = 6`
- `filter_type = 1`
- `cost_function = 2`
- `eps_0 = eps_1 = 1e-6`
- `fit_parabola = true`
- no derivative smoothing

For each row `y` of length `n`:

1. Subtract pybaselines' endpoint parabola.
2. Build the second-order high-pass filter matrices `A` and `B`.
3. Iterate the 9-diagonal BEADS system with L1-v2 derivative weights,
   asymmetric signal weights, and `lam_0 / lam_1 / lam_2`.
4. Reconstruct the low-pass baseline and output `out = y - baseline`.

From the `chemometrics4all.BEADS` Python wrapper docstring:

> Baseline estimation and denoising with sparsity.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `lam_0` | `float` | `100.0` |  |
| `lam_1` | `float` | `0.5` |  |
| `lam_2` | `float` | `0.5` |  |
| `max_iter` | `int` | `50` | Maximum number of solver or reweighting iterations. |
| `tol` | `float` | `0.001` | Convergence tolerance. |

## Explanations

### Bibliographic source

- Ning, X., Selesnick, I. W. & Duval, L. (2014). "Chromatogram baseline
  estimation and denoising using sparsity (BEADS)." Chemometrics and
  Intelligent Laboratory Systems 139, 156-167.
- Pybaselines BEADS reference: `pybaselines.misc.beads`.

### Mathematical principle

`beads` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Full pybaselines-compatible banded formulation, specialized to
  `filter_type=1`.
- The iterative linear system has four lower and four upper diagonals.
- On `max_iter` reached without convergence, the last iterate is returned.
- External benchmark gate: `pybaselines.Baseline().beads` with the parameters
  above and `tol=1e-3`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out, double lam_0, double lam_1, double lam_2, int32_t max_iter, double tol);
void c4a_pp_beads_destroy(c4a_pp_beads_handle_t* handle);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_beads` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.BEADS` | Python | sklearn-style wrapper backed by ctypes. |
| R | `beads(X, lam_0 = 100.0, lam_1 = 0.5, lam_2 = 0.5, max_iter = 50L, tol = 1e-3)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.beads(full)` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out, double lam_0, double lam_1, double lam_2, int32_t max_iter, double tol);
void c4a_pp_beads_destroy(c4a_pp_beads_handle_t* handle);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import BEADS

op = BEADS(lam_0=100.0, lam_1=0.5, lam_2=0.5, max_iter=50, tol=0.001)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- beads(X, max_iter = 20L)
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.pybaselines`** (Python · canonical) — `pybaselines.beads(full)` · pybaselines 1.2.1
:::

### Benchmarks
Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). Verdict legend: ✓ exact · ≈ context/drift · ✗ divergent · ⊘ not available · — not run · ⚠ error.
::::{tab-set}
:class: parity-tabs

:::{tab-item} 1 thread
:sync: threads-1

<div class="parity-table-wrap">
<table class="docutils parity-grouped">
<thead><tr><th>Backend</th><th>Parity</th><th>100×50</th><th>100×500</th><th>100×2500</th></tr></thead>
<tbody class="lang-band lang-cpp"><tr class="lang-band-row" data-lang="cpp"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>C++ native · libc4a</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.cpp</code></td><td class="parity parity-exact">✓ exact</td><td class="ms">8.178 ms</td><td class="ms ms-best">🏆 84.687 ms</td><td class="ms ms-best">🏆 428.499 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.sklearn</code></td><td class="parity parity-exact">✓ bind</td><td class="ms ms-best">🏆 7.916 ms</td><td class="ms">85.049 ms</td><td class="ms">431.208 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>C4A.R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">9.188 ms</td><td class="ms">96.500 ms</td><td class="ms">493.000 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.beads(full) · pybaselines 1.2.1 — canonical">◆</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">214.991 ms</td><td class="ms">364.137 ms</td><td class="ms">1037.992 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
