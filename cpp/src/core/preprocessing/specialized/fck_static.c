/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FCKStaticTransformer reference implementation.
 *
 * Builds L = n_orders * n_scales kernels at `_create` time, then applies them
 * row-by-row to the input matrix. For each (alpha_a, sigma_s) pair the kernel
 * is computed once via c4a_fck_kernel_1d; the L kernels are stored in a
 * single contiguous (L, kernel_size) double buffer with row-major layout:
 *
 *   kernels[l * kernel_size + k]    where l = a * n_scales + s
 *
 * (i.e., alpha varies slowest, matching the typical nested-loop ordering
 * `for a in alphas: for s in sigmas: emit kernel(a, s, K)`.)
 *
 * Convolution uses scipy.ndimage.convolve1d semantics with mode='reflect':
 *   - The kernel is reversed (h_rev[k] = h[K - 1 - k]) so that the
 *     correlation we apply implements a true convolution.
 *   - Out-of-range source indices are reflected via c4a_pad_resolve_index
 *     with C4A_PAD_REFLECT (no edge-repeat reflection).
 *
 * The output layout is (n_samples, n_kernels, n_features) flattened as
 * (n_samples, n_kernels * n_features) along the inner-axis-contiguous form
 *
 *   out[i * (L * p) + l * p + j] = sum_{k=0}^{K-1} h_l_rev[k] *
 *                                     X[i * p + idx(j + k - lw)]
 *
 * where lw = (K - 1) / 2 and idx() applies the reflect padding.
 */

#include "fck_static.h"

#include <stdlib.h>
#include <string.h>

#include "core/common/fck_kernel.h"
#include "core/common/padding.h"

struct c4a_pp_fck_static_state_t {
    int32_t kernel_size;     /* K */
    int32_t n_kernels;       /* L = n_orders * n_scales */
    double* kernels;         /* reversed kernels, shape (L, K) row-major */
};

c4a_pp_fck_static_state_t* c4a_pp_fck_static_state_new(
    int32_t kernel_size,
    const double* alphas, int32_t n_orders,
    const double* sigmas, int32_t n_scales) {
    if (kernel_size <= 0 || n_orders <= 0 || n_scales <= 0 ||
        alphas == NULL || sigmas == NULL) {
        return NULL;
    }
    const int64_t n_kernels64 = (int64_t)n_orders * (int64_t)n_scales;
    if (n_kernels64 > (int64_t)INT32_MAX) {
        return NULL;
    }
    const int32_t n_kernels = (int32_t)n_kernels64;

    c4a_pp_fck_static_state_t* s =
        (c4a_pp_fck_static_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->kernel_size = kernel_size;
    s->n_kernels   = n_kernels;
    s->kernels     = (double*)malloc((size_t)n_kernels *
                                      (size_t)kernel_size * sizeof(double));
    if (s->kernels == NULL) {
        free(s);
        return NULL;
    }

    /* Build each kernel into a scratch buffer, then reverse-copy into the
     * destination so the apply loop can use a plain correlation sum. */
    double* scratch = (double*)malloc((size_t)kernel_size * sizeof(double));
    if (scratch == NULL) {
        free(s->kernels);
        free(s);
        return NULL;
    }

    for (int32_t a = 0; a < n_orders; ++a) {
        for (int32_t v = 0; v < n_scales; ++v) {
            const int32_t l = a * n_scales + v;
            if (c4a_fck_kernel_1d(alphas[a], sigmas[v], kernel_size,
                                   scratch) != 0) {
                free(scratch);
                free(s->kernels);
                free(s);
                return NULL;
            }
            double* dst = s->kernels + (size_t)l * (size_t)kernel_size;
            /* Reverse so the apply loop is a cross-correlation that
             * implements convolution against the original kernel. */
            for (int32_t k = 0; k < kernel_size; ++k) {
                dst[k] = scratch[kernel_size - 1 - k];
            }
        }
    }
    free(scratch);
    return s;
}

void c4a_pp_fck_static_state_free(c4a_pp_fck_static_state_t* state) {
    if (state == NULL) return;
    free(state->kernels);
    free(state);
}

int32_t c4a_pp_fck_static_state_n_kernels(
    const c4a_pp_fck_static_state_t* state) {
    return state == NULL ? 0 : state->n_kernels;
}

int32_t c4a_pp_fck_static_state_kernel_size(
    const c4a_pp_fck_static_state_t* state) {
    return state == NULL ? 0 : state->kernel_size;
}

int c4a_pp_fck_static_state_apply(const c4a_pp_fck_static_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols,
                                   double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return -1;
    }
    if (rows < 0 || cols < 0) {
        return -2;
    }
    if (rows == 0 || cols == 0) {
        return 0;
    }
    const int32_t K = state->kernel_size;
    const int32_t L = state->n_kernels;
    const int32_t lw = (K - 1) / 2;
    const double* kernels = state->kernels;

    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in  = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)L * (size_t)cols;
        for (int32_t l = 0; l < L; ++l) {
            const double* h_rev = kernels + (size_t)l * (size_t)K;
            double* out_band = row_out + (size_t)l * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                double acc = 0.0;
                for (int32_t k = 0; k < K; ++k) {
                    const int64_t src_idx = j + (int64_t)k - (int64_t)lw;
                    int64_t resolved;
                    if (src_idx >= 0 && src_idx < cols) {
                        resolved = src_idx;
                    } else {
                        resolved = c4a_pad_resolve_index(src_idx, cols,
                                                          C4A_PAD_REFLECT);
                    }
                    acc += h_rev[k] * row_in[resolved];
                }
                out_band[j] = acc;
            }
        }
    }
    return 0;
}
