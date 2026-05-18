# Phase 2 — Stateless scatter & scaling preprocessings

## Goal

Implement 7 stateless preprocessing operators that don't need a `fit` step (parameters captured at `create` time, then `transform` is a pure function of `X`).

## Operators

| Operator | Algorithm | Parameters | Inverse? |
|----------|-----------|------------|----------|
| **SNV** (Standard Normal Variate) | row-wise centering + scaling: `(X - mean(X, axis=1)) / std(X, axis=1)` | `with_mean: bool`, `with_std: bool`, `ddof: int` | no |
| **LocalSNV** | sliding-window SNV along wavelengths | `window: odd int`, `pad_mode: enum {reflect, edge, constant}`, `constant_value: double` | no |
| **RobustSNV** (RNV) | row-wise median + IQR: `(X - median) / IQR_scaled` (IQR_scaled = IQR / 1.349) | `with_center: bool`, `with_scale: bool` | no |
| **AreaNormalization** | `X / sum(\|X\|, axis=1)` (or `X / sum(X, axis=1)` for non-negative) | `method: enum {sum, abs_sum, max, l1, l2}` | no |
| **Normalize** | `X / norm(X, axis=axis)` | `norm: enum {l1, l2, max}`, `axis: 0|1` | no |
| **SimpleScale** | min-max scale: `(X - min) / (max - min) * (high - low) + low` (row-wise) | `low: double`, `high: double` | no |
| **LogTransform** | `log(X + offset)` with optional base | `offset: double`, `base: double` (0 → natural log) | yes (via `exp((X * log(base)) - offset)`) |

## Files to create

- `cpp/src/core/preprocessing/scatter/{snv.{hpp,c}, local_snv.{hpp,c}, robust_snv.{hpp,c}, area_normalization.{hpp,c}}`
- `cpp/src/core/preprocessing/scaling/{normalize.{hpp,c}, simple_scale.{hpp,c}, log_transform.{hpp,c}}`
- `cpp/src/c_api/c_api_preprocessing.cpp` — extern "C" wrappers for all 7 operators.
- `cpp/include/chemometrics4all/c4a.h` — append "Phase 2 — Stateless scatter / scaling" section.
- `cpp/tests/test_preprocessing_stateless.cpp` — per-operator parity tests vs nirs4all + numpy + prospectr.
- `parity/python_generator/scripts/generate_phase2_fixtures.py` — fixture generator.
- `parity/fixtures/snv_v1.json`, `lsnv_v1.json`, `rnv_v1.json`, `area_norm_v1.json`, `normalize_v1.json`, `simple_scale_v1.json`, `log_transform_v1.json`.
- `docs/algorithms/{snv,lsnv,rnv,area_normalization,normalize,simple_scale,log_transform}.md`.

## ABI surface added

Pattern per operator (using SNV as example):
```c
typedef struct c4a_pp_snv_state_t c4a_pp_snv_state_t;
C4A_API c4a_status_t c4a_pp_snv_create(c4a_pp_snv_state_t** out,
                                       int with_mean, int with_std, int ddof);
C4A_API void         c4a_pp_snv_destroy(c4a_pp_snv_state_t* state);
C4A_API c4a_status_t c4a_pp_snv_transform(c4a_pp_snv_state_t* state,
                                          c4a_matrix_view_t X, c4a_array_t* out);
```

3 symbols × 7 operators = **21 new ABI symbols**. Total: 36 → 57.

## Parity references

| Operator | Primary ref | Tolerance |
|----------|-------------|-----------|
| SNV | nirs4all.operators.transforms.scalers.StandardNormalVariate + numpy mean/std | 1e-12 abs / 1e-13 rel |
| LocalSNV | nirs4all.operators.transforms.scalers.LocalStandardNormalVariate + numpy sliding window | 1e-12 / 1e-13 |
| RobustSNV | nirs4all.operators.transforms.scalers.RobustStandardNormalVariate + numpy median/percentile | 1e-12 / 1e-13 |
| AreaNormalization | nirs4all.operators.transforms.nirs.AreaNormalization + numpy sum/max/norm | 1e-12 / 1e-13 |
| Normalize | sklearn.preprocessing.normalize | 1e-12 / 1e-13 |
| SimpleScale | nirs4all.operators.transforms.scalers.SimpleScale + numpy min/max | 1e-12 / 1e-13 |
| LogTransform | nirs4all.operators.transforms.nirs.LogTransform + numpy log | 1e-12 / 1e-13 |

R cross-check for SNV (optional): `prospectr::standardNormalVariate` — 1e-10 abs (sufficient given prospectr uses sample-population std with `ddof=1`).

## Acceptance criteria

- ✅ Build clean: `cmake --build --preset dev-debug` → 0 warnings, 0 errors.
- ✅ 7 parity test files pass; total tests: 20 → 20 + 7 = 27.
- ✅ ABI symbol list updated: 36 → 57 symbols.
- ✅ `c4a_cli --selfcheck` exits 0.
- ✅ Opus post-review accepts.
- ✅ Push to GitHub, CI green.

## Implementation notes

- All 7 operators are short C functions (~20-50 LOC each). State struct holds only constructor parameters.
- LocalSNV needs a sliding-window helper. Vendor a small `c4a_common_sliding_mean_std` in `cpp/src/core/common/sliding.c`.
- For padded boundaries: `reflect`, `edge`, `constant`. Match scipy.ndimage convention.
- SNV uses `ddof` (delta degrees of freedom) for std normalization. Default 0 (matches numpy); nirs4all uses 0 too. `ddof=1` is sample std.
- AreaNormalization: nirs4all's default is `method="sum"` which sums absolute values. Implement `sum, abs_sum, max, l1, l2`.
- LogTransform: default `offset=0` (caller's responsibility to ensure X > 0). Base 0 means natural log.

## Verification

```bash
cd /home/delete/nirs4all/chemometrics4all
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests
```

## Next phase

[Phase 3 — Stateful scatter](phase-3-stateful-scatter.md): MSC, EMSC, Baseline (centering), Derivate.
