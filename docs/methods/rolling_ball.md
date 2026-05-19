# `rolling_ball` — Rolling ball

_Group_: **Baseline** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/rolling_ball.md`](../algorithms/rolling_ball.md)

## Description

Kneen & Annegarn 1996 rolling-ball baseline. For each row `y` of length `n`:

1. **Min filter** of half-window `W`: `e[i] = min over [max(0, i - W), min(n - 1, i + W)] of y`.
2. **Max filter** of half-window `W` on the result: `z[i] = max over [max(0, i - W), min(n - 1, i + W)] of e`.
3. **Optional moving-average smoothing** of `z` over a `(2 * S + 1)` clipped centred window (skipped when `S = 0`).
4. Output: `out = y - z`.

Edges use centred windows clipped to `[0, n - 1]` (no padding / reflection).

From the `chemometrics4all.RollingBall` Python wrapper docstring:

> Rolling-ball morphological baseline correction.

### Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `half_window` | `int` | `20` |  |
| `smooth_half_window` | `int` | `0` |  |

## Explanations

### Bibliographic source

- Kneen, M. A. & Annegarn, H. J. (1996). "Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds." Nuclear Instruments and Methods B109/110, 209-213.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/rolling_ball.py`.

### Mathematical principle

`rolling_ball` follows the standard baseline operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Two scratch buffers per row (length `n` doubles). No linear algebra.
- Parity tolerance vs frozen NumPy reference: `1e-12 abs / 1e-13 rel`.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_pp_rolling_ball_create( c4a_pp_rolling_ball_handle_t** out, int32_t half_window, int32_t smooth_half_window);
void c4a_pp_rolling_ball_destroy( c4a_pp_rolling_ball_handle_t* handle);
c4a_status_t c4a_pp_rolling_ball_transform( const c4a_pp_rolling_ball_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_pp_rolling_ball` | C/C++ | Stable libc4a entry point family. |
| Python | `chemometrics4all.RollingBall` | Python | sklearn-style wrapper backed by ctypes. |
| R | `rolling_ball(X, half_window = 20L, smooth_half_window = 0L)` | R | Public package wrapper around the C ABI. |
| ref.pybaselines | `pybaselines.rolling_ball` | Python | canonical/comparator |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_pp_rolling_ball_create( c4a_pp_rolling_ball_handle_t** out, int32_t half_window, int32_t smooth_half_window);
void c4a_pp_rolling_ball_destroy( c4a_pp_rolling_ball_handle_t* handle);
c4a_status_t c4a_pp_rolling_ball_transform( const c4a_pp_rolling_ball_handle_t* handle, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

:::

:::{tab-item} Python · chemometrics4all
:sync: python
:class-label: lang-python

```python
from chemometrics4all import RollingBall

op = RollingBall(half_window=20, smooth_half_window=0)
Xt = op.fit_transform(X)
```

:::

:::{tab-item} R · chemometrics4all
:sync: r
:class-label: lang-r

```r
library(chemometrics4all)
res <- rolling_ball(X, half_window = 10L, smooth_half_window = 0L)
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- 📐 **`ref.pybaselines`** (Python · canonical) — `pybaselines.rolling_ball` · pybaselines 1.2.1
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
<tr class="bk-row"><td class="bk-name"><code>C++</code></td><td class="parity parity-exact">✓ exact</td><td class="ms ms-best">🏆 0.054 ms</td><td class="ms ms-best">🏆 0.631 ms</td><td class="ms">3.419 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>Py</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.071 ms</td><td class="ms">0.677 ms</td><td class="ms ms-best">🏆 3.261 ms</td></tr>
</tbody>
<tbody class="lang-band lang-r"><tr class="lang-band-row" data-lang="r"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>R · chemometrics4all</th></tr>
<tr class="bk-row"><td class="bk-name"><code>R</code></td><td class="parity parity-exact">✓ bind</td><td class="ms">0.069 ms</td><td class="ms">0.688 ms</td><td class="ms">4.875 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): pybaselines.rolling_ball · pybaselines 1.2.1 — canonical">📐</span><code>ref.pybaselines</code></td><td class="parity parity-exact">✓ ref</td><td class="ms">2.980 ms</td><td class="ms">3.635 ms</td><td class="ms">6.386 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
