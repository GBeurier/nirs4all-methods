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

struct n4m_aug_hetero_noise_state_t {
    n4m_rng_pcg64* rng;   /* borrowed; not owned. */
    double noise_base;
    double noise_signal_dep;
};

n4m_aug_hetero_noise_state_t* n4m_aug_hetero_noise_state_new(
    n4m_rng_pcg64* rng,
    double noise_base,
    double noise_signal_dep) {
    n4m_aug_hetero_noise_state_t* s =
        (n4m_aug_hetero_noise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng              = rng;
    s->noise_base       = noise_base;
    s->noise_signal_dep = noise_signal_dep;
    return s;
}

void n4m_aug_hetero_noise_state_free(n4m_aug_hetero_noise_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_hetero_noise_state_apply(
    n4m_aug_hetero_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return N4M_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return N4M_OK;

    const size_t n_total = (size_t)rows * (size_t)cols;
    double* noise = (double*)malloc(n_total * sizeof(double));
    if (noise == NULL) return N4M_ERR_OUT_OF_MEMORY;
    n4m_pcg64_engine_standard_normal_fill(state->rng, noise, n_total);

    const double base = state->noise_base;
    const double dep  = state->noise_signal_dep;
    for (size_t k = 0; k < n_total; ++k) {
        const double xv    = X[k];
        const double sigma = base + dep * fabs(xv);
        out[k] = xv + noise[k] * sigma;
    }

    free(noise);
    return N4M_OK;
}
