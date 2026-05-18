# Phase 4 — Derivatives & smoothing preprocessings

## Goal

Implement 5 stateless smoothing / derivative operators that complete the spectral-preprocessing toolkit alongside Phase 2/3 scatter operators.

## Operators

| Operator | Algorithm | Parameters | nirs4all reference |
|----------|-----------|------------|---|
| **SavitzkyGolay** | Polynomial least-squares smoothing & differentiation along axis=1 | `window_length: odd int >= 3`, `polyorder: int < window_length`, `deriv: int >= 0`, `delta: double > 0`, `mode: enum {mirror, constant, nearest, wrap, interp}`, `cval: double` | `scipy.signal.savgol_filter` via `operators/transforms/nirs.py` SavitzkyGolay class |
| **FirstDerivative** | `np.gradient(X, delta, axis=1)` — central differences with shape preservation | `delta: double > 0` (default 1.0) | `operators/transforms/nirs.py` FirstDerivative |
| **SecondDerivative** | Two-pass gradient: `np.gradient(np.gradient(X, delta, axis=1), delta, axis=1)` | `delta: double > 0` | `operators/transforms/nirs.py` SecondDerivative |
| **NorrisWilliams** | Smooth-then-difference: SG-smooth with `polyorder=2, window=segment`, then finite-difference with gap | `segment: odd int >= 3`, `gap: int >= 1`, `derivative_order: 1 or 2` | `operators/transforms/norris_williams.py` |
| **Gaussian** | `scipy.ndimage.gaussian_filter1d(X, sigma, order, axis=1, mode, cval, truncate)` | `sigma: double > 0`, `order: int >= 0`, `mode: enum`, `cval: double`, `truncate: double` (kernel radius) | `operators/transforms/signal.py` Gaussian |

All 5 are stateless — `_create / _transform / _destroy` (no `_fit`).

## Implementation notes

### Savitzky-Golay coefficients

The reference uses `scipy.signal.savgol_filter` which builds polynomial coefficients via `scipy.signal.savgol_coeffs(window_length, polyorder, deriv, delta, pos=None, use='conv')`. Internally that:

1. Builds a `(window_length, polyorder + 1)` Vandermonde matrix `V` where `V[i, k] = (i - (window_length-1)/2)^k`.
2. Computes the pseudo-inverse via SVD-based lstsq.
3. The `deriv`-th row of `V^+` scaled by `factorial(deriv) / delta^deriv` gives the convolution coefficients.

For our C engine, we precompute the coefficients **at `_create` time** (so transform is just a 1D convolution per row). The Vandermonde matrix is small (`window x (polyorder + 1)`); QR-based pseudoinverse is sufficient. The result: an array of `window_length` doubles cached in the handle.

### Boundary modes

`scipy.signal.savgol_filter` supports modes:
- `mirror` (default): reflect indices `[-1, -2, ...]` → `[1, 2, ...]`.
- `constant`: pad with `cval`.
- `nearest`: pad with edge value.
- `wrap`: cyclic.
- `interp`: fit the polynomial to the boundary segment using positions outside the window — this is the most complex mode. For v1 we may defer `interp` to a follow-up and return `C4A_ERR_NOT_IMPLEMENTED`.

`scipy.ndimage.gaussian_filter1d` supports modes:
- `reflect`: `[a b c d]` → `[b a | a b c d | d c]` (mirror without repeating edge).
- `constant`: pad with `cval`.
- `nearest`: replicate edge.
- `mirror`: `[a b c d]` → `[c b | a b c d | c b]` (mirror through edge, repeats edge).
- `wrap`: cyclic.

We replicate both sets exactly using a common `c4a_common_pad_1d` helper in `cpp/src/core/common/padding.c`.

### Gaussian kernel construction

For a 1D Gaussian kernel of sigma σ with truncation T (kernel half-width = `lw = int(T * σ + 0.5)`):
- Coefficients: `g[i] = exp(-i² / (2σ²))` for `i ∈ [-lw, lw]`.
- For `order > 0`: differentiate the kernel analytically.
- Normalize: divide by sum (for `order == 0`).

This matches `scipy.ndimage._filters._gaussian_kernel1d` exactly.

### NorrisWilliams

Two-pass: SG-smooth then finite-difference. Reuse the SG coefficient computation. The "difference" step is `(x[i+gap] - x[i-gap]) / (2*gap*delta)` for order 1, or `(x[i+gap] - 2*x[i] + x[i-gap]) / (gap*delta)^2` for order 2.

## C ABI surface added

5 operators × 3 symbols (`_create / _destroy / _transform`) = **15 new symbols**. Plus enum types for SG mode + Gaussian mode (already covered by SG enum since modes overlap).

Total ABI: 79 → **94 symbols**. ABI 1.3.0 → 1.4.0 (additive).

```c
/* SavitzkyGolay */
typedef struct c4a_pp_savgol_handle_t c4a_pp_savgol_handle_t;
typedef enum c4a_pp_savgol_mode_t {
    C4A_PP_SAVGOL_MIRROR   = 0,
    C4A_PP_SAVGOL_CONSTANT = 1,
    C4A_PP_SAVGOL_NEAREST  = 2,
    C4A_PP_SAVGOL_WRAP     = 3,
    C4A_PP_SAVGOL_INTERP   = 4
} c4a_pp_savgol_mode_t;
C4A_API c4a_status_t c4a_pp_savgol_create(c4a_pp_savgol_handle_t** out,
                                           int32_t window_length,
                                           int32_t polyorder,
                                           int32_t deriv,
                                           double delta,
                                           c4a_pp_savgol_mode_t mode,
                                           double cval);
C4A_API void         c4a_pp_savgol_destroy(c4a_pp_savgol_handle_t* h);
C4A_API c4a_status_t c4a_pp_savgol_transform(const c4a_pp_savgol_handle_t* h,
                                              c4a_matrix_view_t X,
                                              c4a_matrix_view_t out);

/* FirstDerivative / SecondDerivative */
typedef struct c4a_pp_first_derivative_handle_t  c4a_pp_first_derivative_handle_t;
typedef struct c4a_pp_second_derivative_handle_t c4a_pp_second_derivative_handle_t;
C4A_API c4a_status_t c4a_pp_first_derivative_create(c4a_pp_first_derivative_handle_t** out, double delta);
C4A_API void         c4a_pp_first_derivative_destroy(c4a_pp_first_derivative_handle_t* h);
C4A_API c4a_status_t c4a_pp_first_derivative_transform(...);  /* same shape as savgol */
/* ... same for SecondDerivative ... */

/* NorrisWilliams */
typedef struct c4a_pp_norris_williams_handle_t c4a_pp_norris_williams_handle_t;
C4A_API c4a_status_t c4a_pp_norris_williams_create(c4a_pp_norris_williams_handle_t** out,
                                                    int32_t segment, int32_t gap,
                                                    int32_t derivative_order);
/* ... destroy + transform ... */

/* Gaussian */
typedef struct c4a_pp_gaussian_handle_t c4a_pp_gaussian_handle_t;
typedef enum c4a_pp_gaussian_mode_t {
    C4A_PP_GAUSSIAN_REFLECT  = 0,
    C4A_PP_GAUSSIAN_CONSTANT = 1,
    C4A_PP_GAUSSIAN_NEAREST  = 2,
    C4A_PP_GAUSSIAN_MIRROR   = 3,
    C4A_PP_GAUSSIAN_WRAP     = 4
} c4a_pp_gaussian_mode_t;
C4A_API c4a_status_t c4a_pp_gaussian_create(c4a_pp_gaussian_handle_t** out,
                                             double sigma, int32_t order,
                                             c4a_pp_gaussian_mode_t mode,
                                             double cval, double truncate);
/* ... destroy + transform ... */
```

## Files to create

- `cpp/src/core/preprocessing/derivatives/{savitzky_golay,first_derivative,second_derivative,norris_williams}.{c,h}`
- `cpp/src/core/preprocessing/smoothing/gaussian.{c,h}` (new subdir)
- `cpp/src/core/common/padding.{c,h}` — shared boundary padding helper (used by SG, Gaussian, NorrisWilliams)
- `cpp/src/c_api/c_api_preprocessing.cpp` — append 15 wrappers
- `cpp/tests/test_preprocessing_smoothing.cpp` — 5 × 2 (smoke + parity) = 10 tests
- `parity/python_generator/scripts/generate_phase4_fixtures.py` — extends previous pattern
- `parity/fixtures/{savgol,first_derivative,second_derivative,norris_williams,gaussian}_v1.json`
- `docs/algorithms/{savitzky_golay,first_derivative,second_derivative,norris_williams,gaussian}.md`

## Parity references

| Operator | Reference | Tolerance |
|----------|-----------|-----------|
| SavitzkyGolay | `scipy.signal.savgol_filter` (scipy 1.11.4) | 1e-10 abs / 1e-11 rel (SVD-based pseudoinverse vs QR-based may differ a few ULPs on ill-conditioned Vandermonde) |
| FirstDerivative | `np.gradient(X, delta, axis=1)` | 1e-12 / 1e-13 |
| SecondDerivative | Two-pass `np.gradient` | 1e-12 / 1e-13 (linear ops compose) |
| NorrisWilliams | nirs4all source: SG-smooth + finite-diff | 1e-10 / 1e-11 |
| Gaussian | `scipy.ndimage.gaussian_filter1d` | 1e-10 / 1e-11 (kernel construction order matters) |

## Acceptance criteria

- ✅ Build clean.
- ✅ All 5 operators implement stateless lifecycle.
- ✅ Tests: 51 → 51 + 10 = **61/61**.
- ✅ ABI: 79 → **94 symbols**. ABI bump 1.3.0 → 1.4.0.
- ✅ Opus post-review at `docs/reviews/phase-4/opus-post.md`.
- ✅ **Codex Phases 3+4 combined review** at `docs/reviews/phase-4/codex-post.md` (every-2-phases cadence per user directive).
- ✅ Push to GitHub, CI green.

## Verification

```bash
cd /home/delete/nirs4all/chemometrics4all
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests   # 61/61
./build/dev-debug/cpp/cli/chemometrics4all_cli --selfcheck
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.4.0 | awk '$2=="T" {print $3}' | sort -u | wc -l  # 94
```

## Next phase

[Phase 5 — Baseline correction (AsLS family)](phase-5-baseline-correction.md): Detrend, AirPLS, ArPLS, AsLS, IModPoly, ModPoly, SNIP, RollingBall, IAsLS, BEADS. First phase to introduce the banded LDLT solver (`cpp/src/core/common/banded_solver.{c,h}`).
