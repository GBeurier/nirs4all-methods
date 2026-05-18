/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * HeteroscedasticNoiseAugmenter — reference implementation.
 *
 * Computes a (rows * cols) noise matrix in a single bulk
 * `standard_normal_fill` call, then scales each entry by the local sigma
 * (`base + signal_dep * |X[i, j]|`) and adds to X.
 */

#include "hetero_noise.h"

#include <math.h>
#include <stdlib.h>

struct c4a_aug_hetero_noise_state_t {
    c4a_rng_pcg64* rng;   /* borrowed; not owned. */
    double noise_base;
    double noise_signal_dep;
};

c4a_aug_hetero_noise_state_t* c4a_aug_hetero_noise_state_new(
    c4a_rng_pcg64* rng,
    double noise_base,
    double noise_signal_dep) {
    c4a_aug_hetero_noise_state_t* s =
        (c4a_aug_hetero_noise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng              = rng;
    s->noise_base       = noise_base;
    s->noise_signal_dep = noise_signal_dep;
    return s;
}

void c4a_aug_hetero_noise_state_free(c4a_aug_hetero_noise_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_hetero_noise_state_apply(
    c4a_aug_hetero_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return C4A_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return C4A_OK;

    const size_t n_total = (size_t)rows * (size_t)cols;
    double* noise = (double*)malloc(n_total * sizeof(double));
    if (noise == NULL) return C4A_ERR_OUT_OF_MEMORY;
    c4a_pcg64_engine_standard_normal_fill(state->rng, noise, n_total);

    const double base = state->noise_base;
    const double dep  = state->noise_signal_dep;
    for (size_t k = 0; k < n_total; ++k) {
        const double xv    = X[k];
        const double sigma = base + dep * fabs(xv);
        out[k] = xv + noise[k] * sigma;
    }

    free(noise);
    return C4A_OK;
}
