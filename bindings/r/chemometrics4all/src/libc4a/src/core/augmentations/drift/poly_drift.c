/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PolynomialBaselineDrift — reference implementation.
 *
 * Wavelength normalisation: `lambdas[j] = -1 + 2 * j / (cols - 1)` (the
 * `np.linspace(-1, 1, cols)` form). For `cols == 1` we use `lambdas[0] = -1`,
 * matching `np.linspace(-1, 1, 1) == np.array([-1.0])` in NumPy 1.26.
 *
 * Coefficient draws are iterated `for k in range(degree+1): for i in range(rows)`
 * — i.e. order-major — to mirror nirs4all's
 * `rng.uniform(lo, hi, size=(n_samples, 1))` inside the order loop.
 * That gives a deterministic RNG stream order matched by the Python ref.
 */

#include "poly_drift.h"

#include <stdlib.h>
#include <string.h>

struct c4a_aug_poly_drift_state_t {
    c4a_rng_pcg64* rng;   /* borrowed; not owned. */
    int32_t  degree;
    double*  coeff_min;   /* owned, length degree+1 */
    double*  coeff_max;   /* owned, length degree+1 */
};

c4a_aug_poly_drift_state_t* c4a_aug_poly_drift_state_new(
    c4a_rng_pcg64* rng,
    int32_t degree,
    const double* coeff_min,
    const double* coeff_max) {
    if (degree < 0)                              return NULL;
    if (coeff_min == NULL || coeff_max == NULL) return NULL;
    c4a_aug_poly_drift_state_t* s =
        (c4a_aug_poly_drift_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng       = rng;
    s->degree    = degree;
    const size_t n_coef = (size_t)degree + 1;
    s->coeff_min = (double*)malloc(n_coef * sizeof(double));
    s->coeff_max = (double*)malloc(n_coef * sizeof(double));
    if (s->coeff_min == NULL || s->coeff_max == NULL) {
        free(s->coeff_min);
        free(s->coeff_max);
        free(s);
        return NULL;
    }
    memcpy(s->coeff_min, coeff_min, n_coef * sizeof(double));
    memcpy(s->coeff_max, coeff_max, n_coef * sizeof(double));
    /* Validate intervals — caller must supply lo <= hi for every k. */
    for (size_t k = 0; k < n_coef; ++k) {
        if (!(s->coeff_max[k] >= s->coeff_min[k])) {
            free(s->coeff_min);
            free(s->coeff_max);
            free(s);
            return NULL;
        }
    }
    return s;
}

void c4a_aug_poly_drift_state_free(c4a_aug_poly_drift_state_t* state) {
    if (state == NULL) return;
    free(state->coeff_min);
    free(state->coeff_max);
    free(state);
}

c4a_status_t c4a_aug_poly_drift_state_apply(
    c4a_aug_poly_drift_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return C4A_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0)               return C4A_OK;

    const size_t n_coef = (size_t)state->degree + 1;

    /* 1. Build the lambda axis. linspace(-1, 1, cols).
     *    Special case cols == 1: np.linspace(-1, 1, 1) -> [-1.0]. */
    double* lambdas = (double*)malloc((size_t)cols * sizeof(double));
    if (lambdas == NULL) return C4A_ERR_OUT_OF_MEMORY;
    if (cols == 1) {
        lambdas[0] = -1.0;
    } else {
        const double step = 2.0 / (double)(cols - 1);
        for (int64_t j = 0; j < cols; ++j) {
            lambdas[(size_t)j] = -1.0 + step * (double)j;
        }
    }

    /* 2. Draw (rows * n_coef) uniform coefficients in order-major form.
     *    coeffs[k * rows + i] = coeff for sample i, order k. */
    double* coeffs = (double*)malloc(n_coef * (size_t)rows * sizeof(double));
    if (coeffs == NULL) {
        free(lambdas);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (size_t k = 0; k < n_coef; ++k) {
        const double lo = state->coeff_min[k];
        const double hi = state->coeff_max[k];
        const double span = hi - lo;
        double* dst = coeffs + k * (size_t)rows;
        for (int64_t i = 0; i < rows; ++i) {
            const double u = c4a_pcg64_engine_next_double(state->rng);
            dst[(size_t)i] = lo + span * u;
        }
    }

    /* 3. Accumulate drift[i, j] = sum_k coeffs[k, i] * lambda[j]^k, then add. */
    for (int64_t i = 0; i < rows; ++i) {
        const size_t off = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            const double lj = lambdas[(size_t)j];
            double drift = 0.0;
            double lj_pow = 1.0;  /* lj^0 */
            for (size_t k = 0; k < n_coef; ++k) {
                const double cik = coeffs[k * (size_t)rows + (size_t)i];
                drift += cik * lj_pow;
                lj_pow *= lj;
            }
            out[off + (size_t)j] = X[off + (size_t)j] + drift;
        }
    }

    free(coeffs);
    free(lambdas);
    return C4A_OK;
}
