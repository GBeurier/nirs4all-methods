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

#include "core/augmentation/spectral/spectral_common.h"

struct n4m_aug_gauss_jitter_state_t {
    n4m_rng_pcg64* rng;
    double         sigma_lo;
    double         sigma_hi;
    int32_t        kernel_width;  /* odd, >= 1 */
};

n4m_aug_gauss_jitter_state_t* n4m_aug_gauss_jitter_state_new(
    n4m_rng_pcg64* rng,
    double sigma_lo, double sigma_hi,
    int32_t kernel_width) {
    if (rng == NULL) return NULL;
    if (!(sigma_lo > 0.0)) return NULL;
    if (!(sigma_hi >= sigma_lo)) return NULL;
    if (kernel_width <= 0) return NULL;
    int32_t w = kernel_width;
    if ((w % 2) == 0) ++w;  /* Python helper enforces odd width too */
    n4m_aug_gauss_jitter_state_t* s =
        (n4m_aug_gauss_jitter_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng          = rng;
    s->sigma_lo     = sigma_lo;
    s->sigma_hi     = sigma_hi;
    s->kernel_width = w;
    return s;
}

void n4m_aug_gauss_jitter_state_free(n4m_aug_gauss_jitter_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_gauss_jitter_state_apply(
    n4m_aug_gauss_jitter_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    const int32_t w = state->kernel_width;
    double* sigmas = (double*)malloc((size_t)rows * sizeof(double));
    double* kernel = (double*)malloc((size_t)w * sizeof(double));
    if (sigmas == NULL || kernel == NULL) {
        free(sigmas); free(kernel);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        sigmas[i] = n4m_aug_uniform(state->rng,
                                     state->sigma_lo, state->sigma_hi);
    }
    /* Process row by row — kernel changes per sample, so the convolve
     * helper is reused with rows=1. */
    for (int64_t i = 0; i < rows; ++i) {
        n4m_aug_gauss_kernel_build(sigmas[i], w, kernel);
        const n4m_status_t st = n4m_aug_convolve_mirror(
            X + (size_t)i * (size_t)cols, 1, cols,
            kernel, w,
            out + (size_t)i * (size_t)cols);
        if (st != N4M_OK) {
            free(sigmas); free(kernel);
            return st;
        }
    }
    free(sigmas);
    free(kernel);
    return N4M_OK;
}
