# Phase 8 — Orthogonalization preprocessings (OSC + EPO)

## Goal

Ship the two stateful orthogonalization preprocessings that nirs4all 0.8.x
exposes under `operators/transforms/orthogonalization.py`, keeping the
Phase 0 / 1 ban on embedded PLS models. Both operators learn a projection
at fit time and replay it at transform time; the Y-predictive direction
inside OSC's "PLS-like deflation" step is implemented via the SVD-fallback
path (which for a univariate `y` is bit-equal to a single NIPALS PLS
iteration — see `docs/algorithms/osc.md`).

## Operators (2 stateful ops)

| Operator | Algorithm | Type |
|----------|-----------|------|
| **OSC** (`c4a_pp_osc_*`) | Direct OSC (DOSC variant, Westerhuis 2001). `_fit(X, y)` learns `n_components` orthogonal-to-`y` directions via Gram-Schmidt projection in the loadings space, `_transform(X)` deflates new spectra. Not invertible. | iterative loadings deflation |
| **EPO** (`c4a_pp_epo_*`) | External Parameter Orthogonalization (Roger 2003). `_fit(X, d)` learns the per-feature regression of `d` on `X`. `_transform(X)` is a documented pass-through (the projection requires `d`, which is unknown at transform time). | per-feature linear regression |

## Design rules

* **No embedded PLS.** OSC's PLS-weight extraction step is the SVD
  fallback `w = X^T y / ||X^T y||` (the leading right-singular vector
  of the rank-1 matrix `y x^T`). Documented and verified against
  nirs4all in `docs/algorithms/osc.md`.
* **EPO supervisor.** `c4a_pp_epo_fit(handle, X, d, d_len)` adds a
  `const double* d` parameter not present in the other Phase 2-5
  stateful ops. The wrapper validates `d_len == X.rows`.
* **OSC inverse_transform.** Returns `C4A_ERR_UNSUPPORTED`. The
  orthogonal-component deflation is many-to-one; the inverse is not
  unique.
* **EPO inverse_transform.** Returns `C4A_ERR_UNSUPPORTED`. The
  projection is lossy by design.

## C ABI surface added (§16)

2 ops × 6 symbols (`_create / _destroy / _fit / _transform /
_inverse_transform / _is_fitted`) = **12 new symbols**.
Total: 126 → **138 symbols**. ABI 1.6.0 → **1.7.0**.

```c
/* §16 — Phase 8 Orthogonalization preprocessings */

typedef struct c4a_pp_osc_handle_t c4a_pp_osc_handle_t;

C4A_API c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
C4A_API void         c4a_pp_osc_destroy(c4a_pp_osc_handle_t* h);
C4A_API c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* y, int64_t y_len);
C4A_API c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_inverse_transform(
    const c4a_pp_osc_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);   /* returns C4A_ERR_UNSUPPORTED */
C4A_API c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* h,
                                           int* out_fitted);

typedef struct c4a_pp_epo_handle_t c4a_pp_epo_handle_t;

C4A_API c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale);
C4A_API void         c4a_pp_epo_destroy(c4a_pp_epo_handle_t* h);
C4A_API c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* d, int64_t d_len);
C4A_API c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_epo_inverse_transform(
    const c4a_pp_epo_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);   /* returns C4A_ERR_UNSUPPORTED */
C4A_API c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* h,
                                           int* out_fitted);
```

## Parity tolerances

| Operator | Reference | Abs tol | Rel tol |
|----------|-----------|---------|---------|
| OSC | Frozen NumPy ref | 1e-9 | 1e-10 |
| EPO | Frozen NumPy ref | 1e-9 | 1e-10 |

The frozen NumPy reference matches nirs4all 0.8.x at ≤ 2 ULPs
(`max_abs ≤ 4.4e-16` across all 8 parameter combinations in the
Phase 8 fixtures). The c4a engine then matches the frozen reference
bit-for-bit on the parity tests (the looser 1e-9 / 1e-10 tolerance is
documentary headroom for future refactors).

## Files to create

### Operators
- `cpp/src/core/preprocessing/orthogonalization/osc.{c,h}`
- `cpp/src/core/preprocessing/orthogonalization/epo.{c,h}`

### C ABI wrappers
- `cpp/src/c_api/c_api_orthogonalization.cpp` (NEW file — first
  category outside `c_api_preprocessing.cpp`).

### Frozen Python references
- `parity/python_generator/src/c4a_parity_orthog_ref/__init__.py`
- `parity/python_generator/src/c4a_parity_orthog_ref/osc.py`
- `parity/python_generator/src/c4a_parity_orthog_ref/epo.py`

### Tests + fixtures + docs
- `cpp/tests/test_preprocessing_orthogonalization.cpp` — 6 new tests
  (2 smoke + 2 parity + 2 refit).
- `parity/python_generator/scripts/generate_phase8_fixtures.py`
- `parity/fixtures/osc_v1.json`, `parity/fixtures/epo_v1.json`
  (4 cases each, including fit_y_hex / fit_d_hex supervisor fields).
- `docs/algorithms/osc.md`, `docs/algorithms/epo.md`

### Process
- `cpp/include/chemometrics4all/c4a.h` — append §16 (Phase 8
  Orthogonalization). User integrates centrally.
- `cpp/include/chemometrics4all/c4a_version.h` — bump
  `C4A_ABI_VERSION_MINOR` 6 → 7. User integrates centrally.
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` — add the 12
  new `c4a_pp_{osc,epo}_*` symbols. User integrates centrally.
- `cpp/tests/main.cpp` — add
  `register_preprocessing_orthogonalization_tests(r)` call. User
  integrates centrally.
- `CHANGELOG.md` — `[0.1.0+abi.1.7.0]` section.

## Acceptance criteria

- Build clean (no new warnings).
- Tests: 82 → **88/88** (2 smoke + 2 parity + 2 refit).
- ABI: 126 → **138 symbols**. ABI bump 1.6.0 → 1.7.0.
- Frozen NumPy ref under `c4a_parity_orthog_ref/` matches nirs4all 0.8.x
  at ≤ 2 ULPs (verified via the side-by-side `check_phase8_parity.py`
  during fixture generation).
- OSC `inverse_transform` returns `C4A_ERR_UNSUPPORTED`.
- EPO `inverse_transform` returns `C4A_ERR_UNSUPPORTED`.
- Phase 0 / 1 PLS ban respected — no NIPALS or kernel-PLS iteration
  enters the OSC engine; we only use the SVD-fallback `X^T y / ||...||`
  direction which is the leading-singular vector of a rank-1 matrix.

## Verification

```bash
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests   # 88/88
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.7.0 \
    | awk '$2=="T" {print $3}' | sort -u | wc -l   # 138
```

## Next phase

[Phase 9 — Wavelet decomposition + denoising](phase-9-wavelets.md):
`Haar`, `WaveletDenoise`, et al.
