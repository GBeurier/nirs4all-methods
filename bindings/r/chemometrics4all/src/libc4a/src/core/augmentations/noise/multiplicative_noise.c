/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * MultiplicativeNoise — reference implementation.
 *
 * Draws one Gaussian sample per row, scales by `sigma_gain`, and broadcasts
 * the resulting per-sample factor across the row:
 *
 *     standard_normal_fill(rng, eps, n_samples)
 *     out[i, j] = X[i, j] * (1 + sigma_gain * eps[i])
 */

#include "multiplicative_noise.h"

#include <stdlib.h>

struct c4a_aug_multiplicative_noise_state_t {
    c4a_rng_pcg64* rng;   /* borrowed; not owned. */
    double sigma_gain;
};

c4a_aug_multiplicative_noise_state_t* c4a_aug_multiplicative_noise_state_new(
    c4a_rng_pcg64* rng, double sigma_gain) {
    c4a_aug_multiplicative_noise_state_t* s =
        (c4a_aug_multiplicative_noise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng        = rng;
    s->sigma_gain = sigma_gain;
    return s;
}

void c4a_aug_multiplicative_noise_state_free(
    c4a_aug_multiplicative_noise_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_multiplicative_noise_state_apply(
    c4a_aug_multiplicative_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return C4A_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return C4A_OK;

    double* eps = (double*)malloc((size_t)rows * sizeof(double));
    if (eps == NULL) return C4A_ERR_OUT_OF_MEMORY;
    c4a_pcg64_engine_standard_normal_fill(state->rng, eps, (size_t)rows);

    const double sigma_gain = state->sigma_gain;
    for (int64_t i = 0; i < rows; ++i) {
        const double gain = 1.0 + sigma_gain * eps[(size_t)i];
        const size_t off = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            out[off + (size_t)j] = X[off + (size_t)j] * gain;
        }
    }

    free(eps);
    return C4A_OK;
}
