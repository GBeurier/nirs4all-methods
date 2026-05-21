/* SPDX-License-Identifier: CECILL-2.1 */
#include "rotate_translate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct n4m_aug_rotate_translate_state_t {
    double p_range;
    double y_factor;
};

n4m_aug_rotate_translate_state_t* n4m_aug_rotate_translate_state_new(
    double p_range, double y_factor) {
    if (y_factor == 0.0) return NULL;
    n4m_aug_rotate_translate_state_t* s =
        (n4m_aug_rotate_translate_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->p_range  = p_range;
    s->y_factor = y_factor;
    return s;
}

void n4m_aug_rotate_translate_state_free(
    n4m_aug_rotate_translate_state_t* state) {
    free(state);
}

static double uniform(n4m_rng_pcg64* rng, double lo, double hi) {
    return lo + (hi - lo) * n4m_pcg64_engine_next_double(rng);
}

n4m_status_t n4m_aug_rotate_translate_state_apply(
    const n4m_aug_rotate_translate_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL || rng_void == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    /* x_range = linspace(0, 1, cols). */
    double* xr = (double*)malloc((size_t)cols * sizeof(double));
    if (xr == NULL) return N4M_ERR_OUT_OF_MEMORY;
    if (cols == 1) {
        xr[0] = 0.0;
    } else {
        for (int64_t j = 0; j < cols; ++j) {
            xr[j] = (double)j / (double)(cols - 1);
        }
    }
    const double pr5 = state->p_range / 5.0;
    /* Python draws p1, p2, xI then yI_uniform — all (n_samples,) vectors.
     * Mirror this draw order so the RNG stream is deterministic and
     * documented. */
    double* p1 = (double*)malloc((size_t)rows * sizeof(double));
    double* p2 = (double*)malloc((size_t)rows * sizeof(double));
    double* xI = (double*)malloc((size_t)rows * sizeof(double));
    double* uy = (double*)malloc((size_t)rows * sizeof(double));
    if (p1 == NULL || p2 == NULL || xI == NULL || uy == NULL) {
        free(xr); free(p1); free(p2); free(xI); free(uy);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) p1[i] = uniform(rng, -pr5, pr5);
    for (int64_t i = 0; i < rows; ++i) p2[i] = uniform(rng, -pr5, pr5);
    for (int64_t i = 0; i < rows; ++i) xI[i] = uniform(rng, 0.0, 1.0);
    for (int64_t i = 0; i < rows; ++i) uy[i] = uniform(rng, 0.0, 1.0);
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        /* Per-row max and std. */
        double mx = xrow[0];
        double sum = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            if (xrow[j] > mx) mx = xrow[j];
            sum += xrow[j];
        }
        const double mean = sum / (double)cols;
        double ssq = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            const double d = xrow[j] - mean;
            ssq += d * d;
        }
        const double sd = sqrt(ssq / (double)cols);
        const double yI_upper = (mx / state->y_factor > 0.0)
                                   ? mx / state->y_factor
                                   : 0.0;
        const double yI = uy[i] * yI_upper;
        for (int64_t j = 0; j < cols; ++j) {
            const double dxj = xr[j] - xI[i];
            const double distor = (xr[j] <= xI[i])
                                    ? (p1[i] * dxj + yI)
                                    : (p2[i] * dxj + yI);
            orow[j] = xrow[j] + distor * sd;
        }
    }
    free(xr); free(p1); free(p2); free(xI); free(uy);
    return N4M_OK;
}
