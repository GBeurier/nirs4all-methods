/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * IModPoly — Gan, Ruan, Mo 2006 improved modified polynomial baseline.
 *
 * Internal parity fixture: parity/python_generator/src/c4a_parity_pybaselines_ref/imodpoly.py
 *
 * For each row y of length n:
 *
 *   z := polyfit(positions, y, polyorder)
 *   devr := std(y - z, ddof=0)
 *   for iter in 0..max_iter:
 *       threshold := z + devr
 *       y := where(y > threshold, threshold, y)
 *       z := polyfit(positions, y, polyorder)
 *       devr_new := std(y - z, ddof=0)
 *       conv := |devr_new - devr| / max(devr_new, DBL_MIN)
 *       if conv < tol: break
 *       devr := devr_new
 *   out := original_y - z
 *
 * Differences vs ModPoly: peak-clipping uses (z + devr) not z; convergence
 * test is on devr rather than baseline L2. This matches the IModPoly
 * algorithm and the pybaselines implementation (which uses num_std=1.0 by
 * default). The mask_initial_peaks pre-step is disabled (matches num_std=1
 * default behaviour for clean spectra and keeps the parity surface compact).
 */

#include "imodpoly.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"

struct c4a_pp_imodpoly_state_t {
    int32_t polyorder;
    int32_t max_iter;
    double  tol;
};

c4a_pp_imodpoly_state_t* c4a_pp_imodpoly_state_new(int32_t polyorder,
                                                     int32_t max_iter,
                                                     double tol) {
    if (polyorder < 0 || max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_imodpoly_state_t* s =
        (c4a_pp_imodpoly_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->polyorder = polyorder;
    s->max_iter  = max_iter;
    s->tol       = tol;
    return s;
}

void c4a_pp_imodpoly_state_free(c4a_pp_imodpoly_state_t* state) {
    free(state);
}

/* Compute std(y - z, ddof=0) without an extra buffer. */
static double residual_std_pop(const double* y, const double* z, int64_t n) {
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        mean += y[i] - z[i];
    }
    mean /= (double)n;
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (y[i] - z[i]) - mean;
        ss += d * d;
    }
    return sqrt(ss / (double)n);
}

c4a_status_t c4a_pp_imodpoly_state_apply(const c4a_pp_imodpoly_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    const int32_t deg = state->polyorder;
    const int64_t p   = (int64_t)(deg + 1);
    if (cols < p) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    double* V_orig = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* V_qr   = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* tau    = (double*)malloc((size_t)p * sizeof(double));
    double* qt_y   = (double*)malloc((size_t)cols * sizeof(double));
    double* coefs  = (double*)malloc((size_t)p * sizeof(double));
    double* y_buf  = (double*)malloc((size_t)cols * sizeof(double));
    double* z      = (double*)malloc((size_t)cols * sizeof(double));
    if (!V_orig || !V_qr || !tau || !qt_y || !coefs || !y_buf || !z) {
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < cols; ++i) {
        const double xi = (double)i;
        double power = 1.0;
        for (int32_t j = (int32_t)p - 1; j >= 0; --j) {
            V_orig[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= xi;
        }
    }
    memcpy(V_qr, V_orig, (size_t)cols * (size_t)p * sizeof(double));
    const c4a_status_t qrst = c4a_householder_qr(V_qr, cols, p, tau);
    if (qrst != C4A_OK) {
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z);
        return qrst;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y0 = X + (size_t)r * (size_t)cols;
        memcpy(y_buf, y0, (size_t)cols * sizeof(double));

        /* Initial baseline + residual stdev. */
        memcpy(qt_y, y_buf, (size_t)cols * sizeof(double));
        c4a_status_t st = c4a_apply_qt(V_qr, cols, p, tau, qt_y);
        if (st != C4A_OK) { goto fail; }
        st = c4a_back_solve_R(V_qr, cols, p, qt_y, coefs);
        if (st != C4A_OK) { goto fail; }
        for (int64_t i = 0; i < cols; ++i) {
            double bval = 0.0;
            for (int32_t j = 0; j < (int32_t)p; ++j) {
                bval += coefs[j] * V_orig[(size_t)i * (size_t)p + (size_t)j];
            }
            z[i] = bval;
        }
        double devr = residual_std_pop(y_buf, z, cols);

        for (int32_t it = 0; it < state->max_iter; ++it) {
            /* Clip y[i] := min(y[i], z[i] + devr). */
            for (int64_t i = 0; i < cols; ++i) {
                const double thresh = z[i] + devr;
                if (y_buf[i] > thresh) y_buf[i] = thresh;
            }
            memcpy(qt_y, y_buf, (size_t)cols * sizeof(double));
            st = c4a_apply_qt(V_qr, cols, p, tau, qt_y);
            if (st != C4A_OK) { goto fail; }
            st = c4a_back_solve_R(V_qr, cols, p, qt_y, coefs);
            if (st != C4A_OK) { goto fail; }
            for (int64_t i = 0; i < cols; ++i) {
                double bval = 0.0;
                for (int32_t j = 0; j < (int32_t)p; ++j) {
                    bval += coefs[j] * V_orig[(size_t)i * (size_t)p + (size_t)j];
                }
                z[i] = bval;
            }
            const double devr_new = residual_std_pop(y_buf, z, cols);
            const double denom = (devr_new > DBL_MIN) ? devr_new : DBL_MIN;
            const double conv = fabs(devr_new - devr) / denom;
            devr = devr_new;
            if (conv < state->tol) {
                break;
            }
        }

        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            orow[i] = y0[i] - z[i];
        }
        continue;
    fail:
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z);
        return st;
    }

    free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
    free(y_buf); free(z);
    return C4A_OK;
}
