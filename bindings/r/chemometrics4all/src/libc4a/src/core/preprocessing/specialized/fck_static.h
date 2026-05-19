/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the FCKStaticTransformer operator (Phase 21).
 *
 * The transformer applies a precomputed bank of L = n_orders * n_scales
 * fractional convolutional kernels to each spectrum row. For an input of
 * shape (n, p), the output has shape (n, L * p): the L convolved signals
 * are horizontally concatenated.
 *
 * Each kernel is built once in `_create` via c4a_fck_kernel_1d(alpha,
 * sigma, kernel_size) over the Cartesian product of the alphas vector
 * and the sigmas vector (alpha varies slowest in memory, matching the
 * (i_order, i_scale) flattening so that the output ordering is stable
 * and predictable).
 *
 * Convolution semantics match scipy.ndimage.convolve1d with mode='reflect'
 * (the Phase 4 padding helper's C4A_PAD_REFLECT mode — edge-repeat
 * reflection). The kernel is reversed before being applied as
 * cross-correlation, matching scipy's convolve1d contract:
 *
 *   out[i, l, j] = sum_{k=0}^{K-1} reverse(h_l)[k] * X[i, j + k - lw]
 *
 * where K = kernel_size and lw = (K - 1) / 2.
 *
 * Output layout:
 *   out[row, l, j] = out[row * (L * p) + l * p + j]
 *
 * Lifecycle: stateless on input — the kernel bank is precomputed in
 * `_create` and is constant per-handle.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SPECIALIZED_FCK_STATIC_H
#define CHEMOMETRICS4ALL_CORE_PP_SPECIALIZED_FCK_STATIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_fck_static_state_t c4a_pp_fck_static_state_t;

/* Build the bank from the Cartesian product of `alphas` and `sigmas`. Returns
 * NULL on allocation failure or on invalid parameters (caller is expected to
 * have already validated parameter ranges in the C ABI wrapper). */
c4a_pp_fck_static_state_t* c4a_pp_fck_static_state_new(
    int32_t kernel_size,
    const double* alphas, int32_t n_orders,
    const double* sigmas, int32_t n_scales);

void c4a_pp_fck_static_state_free(c4a_pp_fck_static_state_t* state);

/* Apply the precomputed bank to `X` of shape (rows, cols) row-major.
 * `out` must point at a contiguous buffer of `rows * n_kernels * cols`
 * doubles where n_kernels == c4a_pp_fck_static_state_n_kernels(state). */
int c4a_pp_fck_static_state_apply(const c4a_pp_fck_static_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols,
                                   double* out);

/* Number of kernels in the bank. Returns 0 if state == NULL. */
int32_t c4a_pp_fck_static_state_n_kernels(
    const c4a_pp_fck_static_state_t* state);

/* Kernel length. Returns 0 if state == NULL. */
int32_t c4a_pp_fck_static_state_kernel_size(
    const c4a_pp_fck_static_state_t* state);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SPECIALIZED_FCK_STATIC_H */
