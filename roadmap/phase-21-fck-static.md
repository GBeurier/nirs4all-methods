# Phase 21 — FCKStaticTransformer (fractional convolutional kernels, static bank)

## Goal

Ship the first specialised feature-extraction operator in nirs4all-methods: a
small, fixed bank of fractional convolutional kernels applied as 1-D
convolutions across the wavelength axis. The bank is the Cartesian product of
a list of fractional orders (`alphas`) and Gaussian widths (`sigmas`) at a
single, fixed kernel length. The transformer is stateless on its input —
the kernel bank is data-independent and constructed entirely from the
constructor hyperparameters.

## Operator (1 new)

| Operator | Algorithm | Type |
|----------|-----------|------|
| **FCKStaticTransformer** | Cartesian-product bank of `fractional_kernel_1d(alpha, sigma, K)` filters, applied row-wise via `scipy.ndimage.convolve1d(..., mode='reflect')`. Output shape `(n, L * p)` with `L = n_orders * n_scales`. | non-iterative, stateless |

## Reference

- `nirs4all.operators.transforms.fck_static.FCKStaticTransformer` (Python
  reference — see `fck_static.py`).
- `nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d` (canonical
  spatial-domain kernel construction — mirrored bit-exactly by
  `n4m_fck_kernel_1d`).

The C ABI follows the simpler `(alpha, sigma, K)` interface of
`fractional_kernel_1d` rather than the Python transformer's
`(alpha, scale, K, sigma)` 4-parameter form: `scale` and `sigma` are
collapsed into a single `sigma` axis on the C side (Python `scale * sigma`
is equivalent to scaling the Gaussian's `sigma` directly), and `kernel_size`
is a single scalar rather than a vector. This keeps the bank size
deterministic at `n_orders * n_scales` and the per-kernel storage uniform.

## C ABI surface added

4 new symbols (3 wrappers + 1 shape-helper):

```c
/* New public header — split off from n4m.h to keep the canonical
 * preprocessing surface compact. Include alongside n4m.h. */
#include "n4m/n4m_fck.h"

typedef struct n4m_pp_fck_static_handle_t n4m_pp_fck_static_handle_t;

N4M_API n4m_status_t n4m_pp_fck_static_create(
    n4m_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders,   /* alphas, length n_orders */
    int32_t n_orders,
    const double* filter_scales,   /* sigmas, length n_scales */
    int32_t n_scales);

N4M_API void n4m_pp_fck_static_destroy(n4m_pp_fck_static_handle_t* h);

N4M_API n4m_status_t n4m_pp_fck_static_transform(
    const n4m_pp_fck_static_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* Helper: output_cols = n_kernels * n_features, with int32_t overflow check. */
N4M_API n4m_status_t n4m_pp_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out);
```

Total: 126 → **130 symbols**.

## Internal helpers added

- `cpp/src/core/common/fck_kernel.{c,h}` — `n4m_fck_kernel_1d(alpha, sigma,
  kernel_size, out)`. Pure C; constructs the spatial-domain kernel
  (Gaussian × signed fractional power × optional zero-mean × L1
  normalisation) directly without any FFT/DFT. No external dependency.
- `cpp/src/core/preprocessing/specialized/fck_static.{c,h}` — engine state +
  `_state_new` / `_state_free` / `_state_apply`. Precomputes the L kernels
  at `_new` time and stores them reversed so the apply loop is a plain
  cross-correlation matching `scipy.ndimage.convolve1d` semantics with
  `mode='reflect'`.

## Test surface added

A new standalone test executable `n4m_tests_fck` (registered as
the second ctest binary alongside `n4m_tests`) covers:

1. `fck_smoke` — `_create / _transform / _destroy` lifecycle, shape helper,
   argument-validation error paths, NULL handling.
2. `fck_kernel_direct` — exercises `n4m_fck_kernel_1d` for the
   `alpha=0` (pure Gaussian, even-symmetric), `alpha=1.0` (zero-mean,
   antisymmetric), and `alpha=1.5` (zero-mean, antisymmetric) branches;
   verifies symmetry, zero-mean, L1 normalisation contracts.
3. `fck_output_cols` — overflow / NULL-pointer / negative-size paths of
   the shape helper.
4. `fck_parity` — loads `parity/fixtures/fck_static_v1.json` (5 cases) and
   verifies 1e-12 abs / 1e-13 rel parity against the canonical Python
   reference.

The new executable bundles `cpp/src/core/common/fck_kernel.c` so the
kernel-construction-direct test can reach `n4m_fck_kernel_1d` without
exposing the helper on the public ABI (it stays `hidden` visibility in
`libn4m`).

## Parity tolerances

| Operator | Abs tol | Rel tol | Notes |
|----------|---------|---------|-------|
| FCKStaticTransformer | 1e-12 | 1e-13 | Pure arithmetic — one Gaussian + one `pow` + L1 reduction per kernel build, then dot-products for the apply phase. No SVD or iterative solver. |

## Boundary mode

Default and only mode: `N4M_PAD_REFLECT` (matches
`scipy.ndimage.convolve1d(..., mode='reflect')`). Other boundary modes are
intentionally out of scope for the first FCK landing — they can be added in
a follow-up if a user needs them.

## Files created (Phase 21)

```
cpp/include/n4m/n4m_fck.h
cpp/src/core/common/fck_kernel.{c,h}
cpp/src/core/preprocessing/specialized/fck_static.{c,h}
cpp/src/c_api/c_api_fck.cpp
cpp/tests/test_fck.cpp
docs/algorithms/fck_static.md
parity/fixtures/fck_static_v1.json
parity/python_generator/scripts/generate_phase21_fixtures.py
roadmap/phase-21-fck-static.md   (this file)
```

## Constraints honoured (worktree isolation)

- **Did not modify**: `cpp/include/n4m/n4m.h`,
  `cpp/include/n4m/n4m_version.h`, top-level `CMakeLists.txt`,
  `cpp/tests/main.cpp`, `cpp/abi/expected_symbols_*.txt`.
- **Did modify**: `cpp/src/CMakeLists.txt` (added 3 new source files
  to the source lists), `cpp/tests/CMakeLists.txt` (added the new
  `n4m_tests_fck` executable + ctest registration), and
  `CHANGELOG.md` (Phase 21 entry).

The parent project's merger needs to:
1. Append the 4 new symbols to `cpp/abi/expected_symbols_{linux,macos,windows}.txt`.
2. Optionally fold `n4m_fck.h` into `n4m.h` §13 if a single-header surface is
   preferred (the current split keeps the two files independent).
3. Bump `N4M_ABI_VERSION_MINOR` from 6 to 7 (additive change, no
   compatibility break).
4. Update `docs/reviews/PHASES.md` row 21 with the post-review verdict.
