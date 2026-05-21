/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * UnsharpSpectralMask — implementation.
 *
 * Stream order matches the Python reference exactly:
 *   1. (deterministic) build smoothing kernel + batch-smooth.
 *   2. rng.uniform(amount_lo, amount_hi, size=(n_samples, 1)).
 *   3. row-wise fused multiply-add.
 */

#include "unsharp_mask.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct n4m_aug_unsharp_mask_state_t {
    n4m_rng_pcg64* rng;
    double         amount_lo;
    double         amount_hi;
    double         sigma;
    int32_t        kernel_width;
    double*        kernel;       /* precomputed at _new */
};

n4m_aug_unsharp_mask_state_t* n4m_aug_unsharp_mask_state_new(
    n4m_rng_pcg64* rng,
    double amount_lo, double amount_hi,
    double sigma, int32_t kernel_width) {
    if (rng == NULL) return NULL;
    if (!(amount_hi >= amount_lo)) return NULL;
    if (!(sigma > 0.0)) return NULL;
    if (kernel_width <= 0) return NULL;
    int32_t w = kernel_width;
    if ((w % 2) == 0) ++w;

    n4m_aug_unsharp_mask_state_t* s =
        (n4m_aug_unsharp_mask_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng          = rng;
    s->amount_lo    = amount_lo;
    s->amount_hi    = amount_hi;
    s->sigma        = sigma;
    s->kernel_width = w;
    s->kernel       = (double*)malloc((size_t)w * sizeof(double));
    if (s->kernel == NULL) {
        free(s);
        return NULL;
    }
    n4m_aug_gauss_kernel_build(sigma, w, s->kernel);
    return s;
}

void n4m_aug_unsharp_mask_state_free(n4m_aug_unsharp_mask_state_t* state) {
    if (state == NULL) return;
    free(state->kernel);
    free(state);
}

n4m_status_t n4m_aug_unsharp_mask_state_apply(
    n4m_aug_unsharp_mask_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    /* Smooth into a scratch buffer so we don't overwrite X before the FMA. */
    const size_t n_elems = (size_t)rows * (size_t)cols;
    double* smoothed = (double*)malloc(n_elems * sizeof(double));
    if (smoothed == NULL) return N4M_ERR_OUT_OF_MEMORY;
    const n4m_status_t st = n4m_aug_convolve_reflect(
        X, rows, cols, state->kernel, state->kernel_width, smoothed);
    if (st != N4M_OK) {
        free(smoothed);
        return st;
    }
    /* Per-sample amounts: rng.uniform(amount_lo, amount_hi, size=(rows, 1)). */
    double* amounts = (double*)malloc((size_t)rows * sizeof(double));
    if (amounts == NULL) {
        free(smoothed);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        amounts[i] = n4m_aug_uniform(state->rng,
                                      state->amount_lo, state->amount_hi);
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double a = amounts[i];
        const double* x  = X + (size_t)i * (size_t)cols;
        const double* sm = smoothed + (size_t)i * (size_t)cols;
        double*       y  = out + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            y[j] = x[j] + a * (x[j] - sm[j]);
        }
    }
    free(amounts);
    free(smoothed);
    return N4M_OK;
}
