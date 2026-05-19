# `signal_type_detector` — SignalTypeDetector

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/signal_type_detector.md`](../algorithms/signal_type_detector.md)

## Description

Heuristic detector for the measurement class of a spectral matrix. Mirrors
`nirs4all.data.signal_type.SignalTypeDetector.detect`.

Given a `rows × cols` row-major F64 matrix `X` (and an optional
`wavelengths` axis in nanometres), the detector returns one of:

| Enum value | Meaning |
|------------|---------|
| `C4A_SIGNAL_UNKNOWN`               | Could not determine (low confidence or preprocessed). |
| `C4A_SIGNAL_ABSORBANCE`            | Typical absorbance, values in ~[0, 3]. |
| `C4A_SIGNAL_REFLECTANCE`           | Fractional reflectance in [0, 1]. |
| `C4A_SIGNAL_REFLECTANCE_PERCENT`   | Percent reflectance in [0, 100]. |
| `C4A_SIGNAL_TRANSMITTANCE`         | Fractional transmittance in [0, 1]. |
| `C4A_SIGNAL_TRANSMITTANCE_PERCENT` | Percent transmittance in [0, 100]. |
| `C4A_SIGNAL_KUBELKA_MUNK`          | Reserved; the heuristic never returns this on raw data. |
| `C4A_SIGNAL_LOG_1_R`               | Reserved; never returned by the value-range heuristic. |
| `C4A_SIGNAL_LOG_1_T`               | Reserved; never returned by the value-range heuristic. |

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
c4a_status_t c4a_signal_detect(c4a_matrix_view_t X, const double* wavelengths_optional, int64_t wl_length, double confidence_threshold, c4a_signal_type_t* type_out, double* confidence_out, char reason_buf[256]);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_signal_detect` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_signal_detect(c4a_matrix_view_t X, const double* wavelengths_optional, int64_t wl_length, double confidence_threshold, c4a_signal_type_t* type_out, double* confidence_out, char reason_buf[256]);
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
