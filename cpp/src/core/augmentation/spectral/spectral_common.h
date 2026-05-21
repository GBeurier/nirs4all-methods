/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal helpers shared by the Phase 16 spectral augmenters.
 *
 *   - n4m_aug_gauss_kernel_build : normalized 1-D Gaussian of fixed odd width.
 *   - n4m_aug_convolve_reflect    : per-row reflect-padded 1-D convolution.
 *   - n4m_aug_uniform             : draw a uniform double from PCG64.
 *   - n4m_aug_randint             : draw a uniform int in [lo, hi) from PCG64.
 *
 * The kernel construction matches nirs4all's `_get_gaussian_kernel`:
 *
 *     x[i]   = i - (width - 1) / 2   for i in [0, width)
 *     k[i]   = exp(- x[i]^2 / (2 sigma^2))
 *     k[i]  /= sum(k)
 *
 * The convolution matches `scipy.ndimage.convolve1d(X, kernel, axis=1,
 * mode='reflect')` for symmetric Gaussian kernels: scipy reverses the kernel
 * once internally so it ends up correlation; on a symmetric kernel that is
 * identical to convolution. The boundary handling uses ndimage's "reflect"
 * convention (no edge repeat), routed through `n4m_pad_resolve_index`.
 *
 * Uniform draws use NumPy's `np.random.default_rng(seed).uniform(a, b)`
 * recipe: take a `[0, 1)` double from PCG64 via `(uint64 >> 11) * 2^-53`,
 * then scale `a + (b - a) * u`. This is bit-exact against NumPy 1.26.4.
 *
 * Integer draws use a single 64-bit accept/reject Lemire scheme to mirror
 * NumPy's `Generator.integers(low, high)` (default endpoint=False) — see
 * n4m_aug_randint for the byte-for-byte equivalence with NumPy's
 * `random_bounded_uint64`.
 */
#ifndef N4M_CORE_AUG_SPECTRAL_COMMON_H
#define N4M_CORE_AUG_SPECTRAL_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build a normalized Gaussian kernel of length `width` (must be odd, >= 1).
 * `out` must have space for `width` doubles. `sigma` must be > 0. */
void n4m_aug_gauss_kernel_build(double sigma, int32_t width, double* out);

/* Apply a per-row 1-D convolution with reflect padding.
 *   - `kernel` has `klen` entries (any positive length; usually odd).
 *   - `X` and `out` are (rows, cols) row-major; in-place safe (X == out).
 *   - Boundary handling: ndimage reflect (no edge repeat).
 * Allocates a single row-sized scratch buffer once. Returns N4M_OK or
 * N4M_ERR_OUT_OF_MEMORY. */
n4m_status_t n4m_aug_convolve_reflect(
    const double* X, int64_t rows, int64_t cols,
    const double* kernel, int64_t klen,
    double* out);

/* Same convolution as above, but with NumPy `np.pad(..., mode="reflect")`
 * no-edge-repeat boundaries. nirs4all's GaussianSmoothingJitter uses this
 * path through `_convolve_1d`, while other spectral smoothers use ndimage
 * reflect via `n4m_aug_convolve_reflect`. */
n4m_status_t n4m_aug_convolve_mirror(
    const double* X, int64_t rows, int64_t cols,
    const double* kernel, int64_t klen,
    double* out);

/* Draw a uniform double in [lo, hi) from PCG64. Equivalent to
 * `rng.uniform(lo, hi)` on NumPy 1.26.4. */
double n4m_aug_uniform(n4m_rng_pcg64* rng, double lo, double hi);

/* Buffered RNG state for `n4m_aug_randint`. NumPy 1.26 uses a 32-bit
 * Lemire path for `rng.integers(low, high)` whenever `high - low` fits
 * in uint32; that path buffers each uint64 draw as two 32-bit halves
 * (low half consumed first, then high half). The buffer survives across
 * `randint` calls, so we expose it as an explicit state that callers
 * reset at the top of each `_apply` and thread through their inner
 * loops. */
typedef struct n4m_aug_randint_state_t {
    n4m_rng_pcg64* rng;          /* not owned */
    uint64_t       buf_value;
    int            buf_has_high; /* 1 if the high 32 bits are still pending */
} n4m_aug_randint_state_t;

void n4m_aug_randint_state_reset(n4m_aug_randint_state_t* state,
                                  n4m_rng_pcg64* rng);

/* Draw a uniform integer in [lo, hi) from PCG64. Equivalent to
 * `rng.integers(lo, hi, endpoint=False)` on NumPy 1.26.4 — uses a 32-bit
 * Lemire path when `hi - lo <= 2^32`, the 64-bit path otherwise. Both
 * `lo` and `hi` must fit in int64_t and `hi > lo`. */
int64_t n4m_aug_randint(n4m_aug_randint_state_t* state,
                         int64_t lo, int64_t hi);

/* Linear interpolation: for each i in [0, n_query), set
 *   out[i] = lerp at x_query[i] into the table (x_known[j], y_known[j]),
 *            assuming x_known is sorted ascending. Matches np.interp:
 *     - x_query[i] <= x_known[0]      -> y_known[0]
 *     - x_query[i] >= x_known[N-1]    -> y_known[N-1]
 *     - otherwise linear interp between bracketing samples.
 * `n_known` >= 1, `n_query` >= 0. */
void n4m_aug_interp_linear(
    const double* x_query, int64_t n_query,
    const double* x_known, const double* y_known, int64_t n_known,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_SPECTRAL_COMMON_H */
