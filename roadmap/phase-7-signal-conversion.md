# Phase 7 — Signal type conversion preprocessings

## Goal

Land the five stateless signal-conversion operators that bridge the
reflectance / transmittance / absorbance domains in the chemometrics4all C
ABI. All five are pure closed-form arithmetic — no learned state, no
iterative loops, no boundary modes — and reach bit-for-bit parity with
`nirs4all.operators.transforms.signal_conversion` at the 1e-12 abs / 1e-13
rel tolerance.

## Operators (5)

| Operator | Algorithm | Parameters | Iterative? |
|----------|-----------|------------|:---:|
| **ToAbsorbance**      | `A = -log10(max(X / [100 if percent else 1], eps))`           | `is_percent: bool`, `epsilon: double`, `clip_negative: bool` | no |
| **FromAbsorbance**    | `R = 10^(-A); R *= 100 if percent`                            | `is_percent: bool`                                          | no |
| **PercentToFraction** | `X / 100.0`                                                   | none                                                        | no |
| **FractionToPercent** | `X * 100.0`                                                   | none                                                        | no |
| **KubelkaMunk**       | `F(R) = (1 - R)^2 / (2 R)` after symmetric clamp `[eps, 1-eps]` | `is_percent: bool`, `epsilon: double`                       | no |

All five are stateless w.r.t. fit (lifecycle: `_create / _transform /
_destroy`).

## ABI surface added

5 ops × 3 symbols (`_create / _destroy / _transform`) = **15 new symbols**.

Total: 126 → **141 symbols**. ABI bump 1.6.0 → 1.7.0.

```c
/* §14 — Phase 7 Signal type conversion preprocessings */
typedef struct c4a_pp_to_absorbance_handle_t   c4a_pp_to_absorbance_handle_t;
typedef struct c4a_pp_from_absorbance_handle_t c4a_pp_from_absorbance_handle_t;
typedef struct c4a_pp_pct_to_frac_handle_t     c4a_pp_pct_to_frac_handle_t;
typedef struct c4a_pp_frac_to_pct_handle_t     c4a_pp_frac_to_pct_handle_t;
typedef struct c4a_pp_kubelka_munk_handle_t    c4a_pp_kubelka_munk_handle_t;

C4A_API c4a_status_t c4a_pp_to_absorbance_create(c4a_pp_to_absorbance_handle_t** out,
                                                  int is_percent, double epsilon,
                                                  int clip_negative);
C4A_API void         c4a_pp_to_absorbance_destroy(c4a_pp_to_absorbance_handle_t* h);
C4A_API c4a_status_t c4a_pp_to_absorbance_transform(
    const c4a_pp_to_absorbance_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);

C4A_API c4a_status_t c4a_pp_from_absorbance_create(
    c4a_pp_from_absorbance_handle_t** out, int is_percent);
C4A_API void         c4a_pp_from_absorbance_destroy(c4a_pp_from_absorbance_handle_t* h);
C4A_API c4a_status_t c4a_pp_from_absorbance_transform(
    const c4a_pp_from_absorbance_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);

C4A_API c4a_status_t c4a_pp_pct_to_frac_create(c4a_pp_pct_to_frac_handle_t** out);
C4A_API void         c4a_pp_pct_to_frac_destroy(c4a_pp_pct_to_frac_handle_t* h);
C4A_API c4a_status_t c4a_pp_pct_to_frac_transform(
    const c4a_pp_pct_to_frac_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);

C4A_API c4a_status_t c4a_pp_frac_to_pct_create(c4a_pp_frac_to_pct_handle_t** out);
C4A_API void         c4a_pp_frac_to_pct_destroy(c4a_pp_frac_to_pct_handle_t* h);
C4A_API c4a_status_t c4a_pp_frac_to_pct_transform(
    const c4a_pp_frac_to_pct_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);

C4A_API c4a_status_t c4a_pp_kubelka_munk_create(c4a_pp_kubelka_munk_handle_t** out,
                                                 int is_percent, double epsilon);
C4A_API void         c4a_pp_kubelka_munk_destroy(c4a_pp_kubelka_munk_handle_t* h);
C4A_API c4a_status_t c4a_pp_kubelka_munk_transform(
    const c4a_pp_kubelka_munk_handle_t* h, c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol |
|----------|-----------|---------|---------|
| ToAbsorbance      | nirs4all 0.8.11 | 1e-12 | 1e-13 |
| FromAbsorbance    | nirs4all 0.8.11 | 1e-12 | 1e-13 |
| PercentToFraction | nirs4all 0.8.11 | 1e-12 | 1e-13 |
| FractionToPercent | nirs4all 0.8.11 | 1e-12 | 1e-13 |
| KubelkaMunk       | nirs4all 0.8.11 | 1e-12 | 1e-13 |

All five operators are pure closed-form arithmetic — no iteration, no linear
algebra — and reach the tight 1e-12 / 1e-13 floor in every fixture case.
The fixture loop verified all 14 case combinations bit-for-bit (or within
the tolerance window) against the captured Python reference.

## Files created

### Operators (5 × {.c, .h})
- `cpp/src/core/preprocessing/signal_conversion/to_absorbance.{c,h}`
- `cpp/src/core/preprocessing/signal_conversion/from_absorbance.{c,h}`
- `cpp/src/core/preprocessing/signal_conversion/percent_to_fraction.{c,h}`
- `cpp/src/core/preprocessing/signal_conversion/fraction_to_percent.{c,h}`
- `cpp/src/core/preprocessing/signal_conversion/kubelka_munk.{c,h}`

### C ABI wrappers
- `cpp/src/c_api/c_api_signal_conversion.cpp` — 15 extern-C wrappers
  (5 ops × `create / destroy / transform`).

### Tests
- `cpp/tests/test_preprocessing_signal_conversion.cpp` — 10 tests
  (5 smoke + 5 parity).

### Fixtures
- `parity/fixtures/to_absorbance_v1.json`   (1.2 MB, 5 cases)
- `parity/fixtures/from_absorbance_v1.json` (1.0 MB, 4 cases)
- `parity/fixtures/pct_to_frac_v1.json`     (400 kB, 1 case)
- `parity/fixtures/frac_to_pct_v1.json`     (400 kB, 1 case)
- `parity/fixtures/kubelka_munk_v1.json`    (800 kB, 3 cases)

### Generator + docs
- `parity/python_generator/scripts/generate_phase7_fixtures.py`
- `docs/algorithms/{to_absorbance, from_absorbance, percent_to_fraction,
  fraction_to_percent, kubelka_munk}.md`
- `roadmap/phase-7-signal-conversion.md` (this file)

## Files to modify (central integration)

- `cpp/include/chemometrics4all/c4a.h` — append §14 (15 new symbols).
- `cpp/include/chemometrics4all/c4a_version.h` — bump MINOR 6 → 7.
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` — add 15 new lines.
- `cpp/src/CMakeLists.txt` — add 5 new C engine sources + 1 wrapper source.
- `cpp/tests/CMakeLists.txt` — add `test_preprocessing_signal_conversion.cpp`.
- `cpp/tests/main.cpp` — register
  `register_preprocessing_signal_conversion_tests`.
- `CHANGELOG.md` — Phase 7 entry under `[0.1.0+abi.1.7.0]`.

## Acceptance criteria

- ✅ Build clean.
- ✅ Tests: 82 → 82 + 10 = **92/92**.
- ✅ ABI: 126 → **141 symbols**. ABI bump 1.6.0 → 1.7.0.
- ✅ All 14 case combinations across the 5 operators pass the tight 1e-12 /
  1e-13 tolerance window against the captured nirs4all 0.8.11 reference.

## Verification

```bash
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests       # 92/92 pass
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.7.0 \
  | awk '$2=="T" {print $3}' | sort -u | wc -l           # 141
```

## Next phase

Phase 8 will tackle the remaining preprocessing operators that round out the
nirs4all coverage matrix — likely the wavelet family
(`Haar`, `WaveletDenoise`, the explicit `Wavelet` transformer) which
share the boundary-mode and decomposition-level machinery.
