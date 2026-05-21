/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Detrend reference implementation.
 *
 * For each row x of length n:
 *   positions = [0, 1, ..., n-1] (float64)
 *   Build Vandermonde V (n, polyorder+1) with descending powers:
 *     V[i, j] = positions[i] ** (polyorder - j)
 *   Solve V c ~= x via Householder QR (QR shared with EMSC/SG).
 *   baseline[i] = sum_j c[j] * V[i, j]
 *   out[i] = x[i] - baseline[i]
 *
 * The descending-powers Vandermonde matches `np.vander(positions, polyorder + 1)`,
 * so the coefficients are returned in NumPy `polyfit` order (highest degree
 * first, identical to `np.polyfit(positions, x, polyorder)`).
 *
 * polyorder = 0 is a special case (constant baseline = mean(x)).
 */

#include "detrend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"

struct n4m_pp_detrend_state_t {
    int32_t polyorder;
};

n4m_pp_detrend_state_t* n4m_pp_detrend_state_new(int32_t polyorder) {
    if (polyorder < 0) {
        return NULL;
    }
    n4m_pp_detrend_state_t* s = (n4m_pp_detrend_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->polyorder = polyorder;
    return s;
}

void n4m_pp_detrend_state_free(n4m_pp_detrend_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_detrend_state_apply(const n4m_pp_detrend_state_t* state,
                                         const double* X,
                                         int64_t rows, int64_t cols,
                                         double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    const int32_t deg = state->polyorder;
    const int64_t p   = (int64_t)(deg + 1);   /* number of coefficients */
    if (cols < p) {
        /* Need at least deg+1 points to fit a degree-deg polynomial. */
        return N4M_ERR_INVALID_ARGUMENT;
    }

    /* Build the Vandermonde matrix V (cols, p) once: it depends only on the
     * column count and polyorder, not on the row data. */
    double* V_orig = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* V      = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* tau    = (double*)malloc((size_t)p * sizeof(double));
    double* qt_y   = (double*)malloc((size_t)cols * sizeof(double));
    double* coefs  = (double*)malloc((size_t)p * sizeof(double));
    if (V_orig == NULL || V == NULL || tau == NULL || qt_y == NULL || coefs == NULL) {
        free(V_orig); free(V); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < cols; ++i) {
        /* Descending powers to match np.vander default. */
        const double xi = (double)i;
        double power = 1.0;
        /* Fill from rightmost column first (lowest power). */
        for (int32_t j = (int32_t)p - 1; j >= 0; --j) {
            V_orig[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= xi;
        }
    }

    /* Factor V once. */
    memcpy(V, V_orig, (size_t)cols * (size_t)p * sizeof(double));
    const n4m_status_t qrst = n4m_householder_qr(V, cols, p, tau);
    if (qrst != N4M_OK) {
        free(V_orig); free(V); free(tau); free(qt_y); free(coefs);
        return qrst;
    }

    /* Per-row solve + subtract. */
    for (int64_t r = 0; r < rows; ++r) {
        const double* xrow = X + (size_t)r * (size_t)cols;
        memcpy(qt_y, xrow, (size_t)cols * sizeof(double));
        const n4m_status_t qst = n4m_apply_qt(V, cols, p, tau, qt_y);
        if (qst != N4M_OK) {
            free(V_orig); free(V); free(tau); free(qt_y); free(coefs);
            return qst;
        }
        const n4m_status_t bst = n4m_back_solve_R(V, cols, p, qt_y, coefs);
        if (bst != N4M_OK) {
            free(V_orig); free(V); free(tau); free(qt_y); free(coefs);
            return bst;
        }
        /* baseline[i] = sum_j coefs[j] * V_orig[i, j];  out[i] = x[i] - baseline */
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            double bval = 0.0;
            for (int32_t j = 0; j < (int32_t)p; ++j) {
                bval += coefs[j] * V_orig[(size_t)i * (size_t)p + (size_t)j];
            }
            orow[i] = xrow[i] - bval;
        }
    }

    free(V_orig);
    free(V);
    free(tau);
    free(qt_y);
    free(coefs);
    return N4M_OK;
}
