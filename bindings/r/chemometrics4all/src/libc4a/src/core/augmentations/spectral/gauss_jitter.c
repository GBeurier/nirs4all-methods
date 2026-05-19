/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * GaussianSmoothingJitter — implementation.
 *
 * Sigma values are drawn once via rng.uniform(lo, hi, size=n_samples) before
 * any convolution work, matching the Python reference's batched RNG order.
 */

#include "gauss_jitter.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_gauss_jitter_state_t {
    c4a_rng_pcg64* rng;
    double         sigma_lo;
    double         sigma_hi;
    int32_t        kernel_width;  /* odd, >= 1 */
};

c4a_aug_gauss_jitter_state_t* c4a_aug_gauss_jitter_state_new(
    c4a_rng_pcg64* rng,
    double sigma_lo, double sigma_hi,
    int32_t kernel_width) {
    if (rng == NULL) return NULL;
    if (!(sigma_lo > 0.0)) return NULL;
    if (!(sigma_hi >= sigma_lo)) return NULL;
    if (kernel_width <= 0) return NULL;
    int32_t w = kernel_width;
    if ((w % 2) == 0) ++w;  /* Python helper enforces odd width too */
    c4a_aug_gauss_jitter_state_t* s =
        (c4a_aug_gauss_jitter_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng          = rng;
    s->sigma_lo     = sigma_lo;
    s->sigma_hi     = sigma_hi;
    s->kernel_width = w;
    return s;
}

void c4a_aug_gauss_jitter_state_free(c4a_aug_gauss_jitter_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_gauss_jitter_state_apply(
    c4a_aug_gauss_jitter_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    const int32_t w = state->kernel_width;
    double* sigmas = (double*)malloc((size_t)rows * sizeof(double));
    double* kernel = (double*)malloc((size_t)w * sizeof(double));
    if (sigmas == NULL || kernel == NULL) {
        free(sigmas); free(kernel);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        sigmas[i] = c4a_aug_uniform(state->rng,
                                     state->sigma_lo, state->sigma_hi);
    }
    /* Process row by row — kernel changes per sample, so the convolve
     * helper is reused with rows=1. */
    for (int64_t i = 0; i < rows; ++i) {
        c4a_aug_gauss_kernel_build(sigmas[i], w, kernel);
        const c4a_status_t st = c4a_aug_convolve_reflect(
            X + (size_t)i * (size_t)cols, 1, cols,
            kernel, w,
            out + (size_t)i * (size_t)cols);
        if (st != C4A_OK) {
            free(sigmas); free(kernel);
            return st;
        }
    }
    free(sigmas);
    free(kernel);
    return C4A_OK;
}
