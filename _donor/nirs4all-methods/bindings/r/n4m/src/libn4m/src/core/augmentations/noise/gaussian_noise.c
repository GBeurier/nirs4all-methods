/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * GaussianAdditiveNoise — reference implementation.
 *
 * std-per-row computation matches NumPy's biased estimator (ddof=0):
 *
 *     mu_i  = (1/p) * sum_j X[i, j]
 *     var_i = (1/p) * sum_j (X[i, j] - mu_i)^2
 *     std_i = sqrt(var_i)
 *
 * which is the two-pass form NumPy uses for `np.std(X, axis=1)`. The single
 * `standard_normal_fill(rows*cols)` call produces the noise matrix in
 * row-major order; multiplying by `std_i * sigma` yields the per-sample
 * noise term and the final `out[i, j] = X[i, j] + noise[i, j]`.
 */

#include "gaussian_noise.h"

#include <math.h>
#include <stdlib.h>

struct n4m_aug_gaussian_noise_state_t {
    n4m_rng_pcg64* rng;   /* borrowed; not owned. */
    double sigma;
};

n4m_aug_gaussian_noise_state_t* n4m_aug_gaussian_noise_state_new(
    n4m_rng_pcg64* rng, double sigma) {
    n4m_aug_gaussian_noise_state_t* s =
        (n4m_aug_gaussian_noise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng   = rng;
    s->sigma = sigma;
    return s;
}

void n4m_aug_gaussian_noise_state_free(n4m_aug_gaussian_noise_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_gaussian_noise_state_apply(
    n4m_aug_gaussian_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return N4M_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return N4M_OK;

    const size_t n_total = (size_t)rows * (size_t)cols;

    /* 1. Per-row std (biased estimator, two-pass form). */
    double* stds = (double*)malloc((size_t)rows * sizeof(double));
    if (stds == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (int64_t i = 0; i < rows; ++i) {
        double mu = 0.0;
        const double* row = X + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            mu += row[(size_t)j];
        }
        mu /= (double)cols;
        double var = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            const double d = row[(size_t)j] - mu;
            var += d * d;
        }
        var /= (double)cols;
        stds[(size_t)i] = sqrt(var);
    }

    /* 2. Draw the noise matrix (row-major) with one bulk call. */
    double* noise = (double*)malloc(n_total * sizeof(double));
    if (noise == NULL) {
        free(stds);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_pcg64_engine_standard_normal_fill(state->rng, noise, n_total);

    /* 3. out[i, j] = X[i, j] + noise[i, j] * stds[i] * sigma. */
    const double sigma = state->sigma;
    for (int64_t i = 0; i < rows; ++i) {
        const double scale = stds[(size_t)i] * sigma;
        const size_t off   = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            out[off + (size_t)j] =
                X[off + (size_t)j] + noise[off + (size_t)j] * scale;
        }
    }

    free(noise);
    free(stds);
    return N4M_OK;
}
