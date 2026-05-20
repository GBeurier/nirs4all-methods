# `signal_type_detector` — Signal type detector

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/signal_type_detector.md`](../algorithms/signal_type_detector.md)

## Description

Heuristic detector for the measurement class of a spectral matrix. Mirrors
`nirs4all.data.signal_type.SignalTypeDetector.detect`.

Given a `rows × cols` row-major F64 matrix `X` (and an optional
`wavelengths` axis in nanometres), the detector returns one of:

| Enum value | Meaning |
|------------|---------|
| `N4M_SIGNAL_UNKNOWN`               | Could not determine (low confidence or preprocessed). |
| `N4M_SIGNAL_ABSORBANCE`            | Typical absorbance, values in ~[0, 3]. |
| `N4M_SIGNAL_REFLECTANCE`           | Fractional reflectance in [0, 1]. |
| `N4M_SIGNAL_REFLECTANCE_PERCENT`   | Percent reflectance in [0, 100]. |
| `N4M_SIGNAL_TRANSMITTANCE`         | Fractional transmittance in [0, 1]. |
| `N4M_SIGNAL_TRANSMITTANCE_PERCENT` | Percent transmittance in [0, 100]. |
| `N4M_SIGNAL_KUBELKA_MUNK`          | Reserved; the heuristic never returns this on raw data. |
| `N4M_SIGNAL_LOG_1_R`               | Reserved; never returned by the value-range heuristic. |
| `N4M_SIGNAL_LOG_1_T`               | Reserved; never returned by the value-range heuristic. |

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- `nirs4all.data.signal_type.SignalTypeDetector.detect`.
- Fixture: `parity/fixtures/signal_type_detect_v1.json`.

### Mathematical principle

1. Compute NaN-skipping `min`, `max`, `mean`, `std` of `X` (ddof = 0).
2. **Preprocessed shortcut**: if any of the following holds, return
   `(UNKNOWN, 0.9, "Data appears preprocessed (centered/normalized)")`:
   - `|mean| < 0.01` and `std > 0.1` (mean-centred);
   - `|std - 1| < 0.1` and `|mean| < 0.1` (SNV / standardised);
   - `min < -0.5` and `max < 0.5` and `|mean| < 0.01` (derivative-shaped).
3. Score each of the five raw signal classes from value-range / mean
   heuristics (identical to the Python reference, see source for exact
   coefficients).
4. **Optional water-band cue**: with a wavelength axis, examine
   1450 / 1940 / 2500 nm. Peaks at those bands nudge the absorbance score
   up; dips nudge reflectance / transmittance up.
5. Pick the argmax. Confidence = `best_score / sum_scores`.
6. If `confidence < confidence_threshold`, downgrade to `UNKNOWN`.
7. Return `(enum, confidence, reason_string)`.

### Implementation

- Deterministic heuristic; no random sampling, no iteration.
- Parity tolerance against the Python reference:
  - Enum value: exact match.
  - Confidence: 1e-12 absolute (bit-for-bit hex in fixtures).
  - Reason string: byte-for-byte identical (matches the Python `.3f` / `.1%`
    format).

C ABI entry points used by the language bindings:

```c
n4m_status_t n4m_signal_detect(n4m_matrix_view_t X, const double* wavelengths_optional, int64_t wl_length, double confidence_threshold, n4m_signal_type_t* type_out, double* confidence_out, char reason_buf[256]);
```

Benchmark comparator backends are registered in the matrix and stored as reproducible snapshots when they define the canonical contract.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `n4m_signal_detect` | C/C++ | Stable libn4m entry point family. |
| Python | `n4m.python.signal_type_detector` | Python | ABI-close function backed by ctypes. |
| ref.nirs4all | `nirs4all.detect_signal_type` | Python | canonical/comparator |

### Usage

Every nirs4all-methods binding dispatches into the same C kernel. Registered comparator/source rows are listed in the benchmark card below.

::::{tab-set}
:class: nirs4all-methods-bindings


:::{tab-item} C ABI · libn4m
:sync: c
:class-label: lang-c

```c
n4m_status_t n4m_signal_detect(n4m_matrix_view_t X, const double* wavelengths_optional, int64_t wl_length, double confidence_threshold, n4m_signal_type_t* type_out, double* confidence_out, char reason_buf[256]);
```

:::

:::{tab-item} Python ABI · n4m.python
:sync: python-abi
:class-label: lang-python

```python
from n4m import python as n4m

Xt = n4m.signal_type_detector(X)
```

:::

::::


**Benchmark Comparators And Sources** ◆

:::{card}
:class-card: external-refs

- ◆ **`ref.nirs4all`** (Python · canonical) — `nirs4all.detect_signal_type` · nirs4all@cd731a23+dirty
:::

### Validation contract

- Operation: `cross_binding_callable` · comparator: `default_allclose` · tolerance: `rtol=1e-05`, `atol=1e-08` · quality: **strict**
- Default validation dataset: `100×50` · seed `20260556`
- Suites: smoke `3` cells; benchmark `11` cells · Default C/Python/reference parity comparator.
- Metrics: `max_abs_diff`, `rel_l2_diff`, `rms_diff`, `shape_equal`
- Truth sources: cross-binding references declared directly in `benchmarks/cross_binding/orchestrator.py`.

| Backend | Library | Gate | Comparator | Note |
|---------|---------|------|------------|------|
| `ref.nirs4all` | `nirs4all.detect_signal_type` | Python / parity | `default_allclose` |  |

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
<tr class="bk-row"><td class="bk-name"><code>N4M.cpp</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.015 ms</td><td class="ms">0.079 ms</td><td class="ms">0.364 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · nirs4all-methods</th></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.python</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms ms-best">🏆 0.014 ms</td><td class="ms ms-best">🏆 0.075 ms</td><td class="ms">0.383 ms</td></tr>
<tr class="bk-row"><td class="bk-name"><code>N4M.sklearn</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.015 ms</td><td class="ms">0.076 ms</td><td class="ms ms-best">🏆 0.362 ms</td></tr>
</tbody>
<tbody class="lang-band lang-python"><tr class="lang-band-row" data-lang="python"><th colspan="5" scope="rowgroup"><span class="lang-band-dot"></span>Python · external</th></tr>
<tr class="bk-row truth-source-strict"><td class="bk-name"><span class="truth-mark" title="Registry parity reference (Python): nirs4all.detect_signal_type · nirs4all@cd731a23+dirty — canonical">◆</span><code>ref.nirs4all</code></td><td class="parity parity-divergence parity-exact" title="worst reference max abs diff over visible sizes">0</td><td class="ms">0.085 ms</td><td class="ms">0.323 ms</td><td class="ms">1.655 ms</td></tr>
</tbody>
</table>
</div>

:::
::::

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
