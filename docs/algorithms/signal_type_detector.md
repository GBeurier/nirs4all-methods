# SignalTypeDetector

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

## Algorithm

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

## ABI

```c
typedef enum n4m_signal_type_t {
    N4M_SIGNAL_UNKNOWN               = 0,
    N4M_SIGNAL_ABSORBANCE            = 1,
    N4M_SIGNAL_REFLECTANCE           = 2,
    N4M_SIGNAL_REFLECTANCE_PERCENT   = 3,
    N4M_SIGNAL_TRANSMITTANCE         = 4,
    N4M_SIGNAL_TRANSMITTANCE_PERCENT = 5,
    N4M_SIGNAL_KUBELKA_MUNK          = 6,
    N4M_SIGNAL_LOG_1_R               = 7,
    N4M_SIGNAL_LOG_1_T               = 8
} n4m_signal_type_t;

n4m_status_t n4m_signal_detect(n4m_matrix_view_t X,
                                const double* wavelengths_optional /* may be NULL */,
                                int64_t wl_length,
                                double confidence_threshold,
                                n4m_signal_type_t* type_out,
                                double* confidence_out,
                                char reason_buf[256]);
```

`reason_buf` is a 256-byte caller-owned buffer; the routine writes a
NUL-terminated UTF-8 summary in the form
`"Range: [<min>, <max>] | Mean: <mean> | Detected: <label> | Confidence: <pct>%"`.
The caller may pass `NULL` to skip the reason write.

## Numerical contract

- Deterministic heuristic; no random sampling, no iteration.
- Parity tolerance against the Python reference:
  - Enum value: exact match.
  - Confidence: 1e-12 absolute (bit-for-bit hex in fixtures).
  - Reason string: byte-for-byte identical (matches the Python `.3f` / `.1%`
    format).

## Reference

- `nirs4all.data.signal_type.SignalTypeDetector.detect`.
- Fixture: `parity/fixtures/signal_type_detect_v1.json`.
