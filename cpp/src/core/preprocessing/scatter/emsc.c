/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Extended Multiplicative Scatter Correction reference implementation.
 *
 * `_fit` learns:
 *   reference[j]   = mean(X[:, j])     # per-column mean, length p (= cols)
 *   wavelengths[j] = (double)j         # integer wavelength axis matching
 *                                       # nirs4all's `np.arange(p)`
 *
 * `_transform` solves a (p x m) least-squares per row with the basis
 *   B = [reference, wavelengths^1, ..., wavelengths^degree]
 * to get coefficients c (length m). m = degree + 1. The output row is
 *   x'_i = (x_i - sum_{d=1..degree} c[d] * wavelengths^d) / c[0]
 *
 * Numerical strategy: we factor B = Q R using Householder reflections once
 * at fit time. Q is implicitly represented by the unit vectors v_k stored
 * column-wise in the lower triangle of the in-place factor, and tau_k
 * scalars are stored alongside. R (m x m upper triangular) is also stored
 * in-place. At transform time, each row's coefficient solve is
 *   y = Q^T x_i        # length p, only the first m entries matter
 *   c = R^{-1} y[0:m]  # back-substitution
 *
 * QR is numerically much more stable than the normal-equations + Cholesky
 * shortcut: condition(R) ≈ condition(B), whereas condition(B^T B) ≈
 * condition(B)^2. For our raw-integer wavelengths up to ~200 and
 * polynomial degree 2-3 the Gram matrix conditioning would otherwise cost
 * 8-10 decimal digits.
 */

#include "emsc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"

struct n4m_pp_emsc_state_t {
    int32_t  degree;
    int      fitted;
    int64_t  cols;
    int32_t  m;             /* basis dimensionality = degree + 1 */
    double*  basis_qr;      /* p * m, row-major: in-place Householder factor */
    double*  tau;           /* m, Householder scaling factors */
    double*  poly_basis;    /* p * degree, row-major: cached wavelengths^d
                             * columns for the subtraction step. */
};

n4m_pp_emsc_state_t* n4m_pp_emsc_state_new(int32_t degree) {
    if (degree < 1) {
        return NULL;
    }
    n4m_pp_emsc_state_t* s = (n4m_pp_emsc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->degree     = degree;
    s->fitted     = 0;
    s->cols       = 0;
    s->m          = degree + 1;
    s->basis_qr   = NULL;
    s->tau        = NULL;
    s->poly_basis = NULL;
    return s;
}

void n4m_pp_emsc_state_free(n4m_pp_emsc_state_t* state) {
    if (state == NULL) return;
    free(state->basis_qr);
    free(state->tau);
    free(state->poly_basis);
    free(state);
}

int n4m_pp_emsc_state_is_fitted(const n4m_pp_emsc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

n4m_status_t n4m_pp_emsc_state_fit(n4m_pp_emsc_state_t* state,
                                    const double* X,
                                    int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < (int64_t)(state->degree + 2)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int32_t m = state->m;
    const int64_t p = cols;
    const int32_t degree = state->degree;

    double* basis = (double*)malloc((size_t)p * (size_t)m * sizeof(double));
    if (basis == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* poly = (double*)malloc((size_t)p * (size_t)degree * sizeof(double));
    if (poly == NULL) {
        free(basis);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* tau = (double*)malloc((size_t)m * sizeof(double));
    if (tau == NULL) {
        free(basis);
        free(poly);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    /* Column 0 = reference = mean(X, axis=0). */
    const double rows_d = (double)rows;
    for (int64_t j = 0; j < p; ++j) {
        double acc = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            acc += X[(size_t)i * (size_t)cols + (size_t)j];
        }
        basis[(size_t)j * (size_t)m + 0u] = acc / rows_d;
    }
    /* Columns 1..degree = wavelengths^d, wavelengths = 0, 1, ..., p-1. */
    for (int64_t j = 0; j < p; ++j) {
        const double w = (double)j;
        double power = 1.0;
        for (int32_t d = 1; d <= degree; ++d) {
            power *= w;
            basis[(size_t)j * (size_t)m + (size_t)d] = power;
            /* Cache for the subtraction step. */
            poly[(size_t)j * (size_t)degree + (size_t)(d - 1)] = power;
        }
    }

    /* In-place Householder QR. */
    {
        const n4m_status_t qst = n4m_householder_qr(basis, p, (int64_t)m, tau);
        if (qst != N4M_OK) {
            free(basis);
            free(poly);
            free(tau);
            return qst;
        }
    }

    /* Commit. */
    free(state->basis_qr);
    free(state->tau);
    free(state->poly_basis);
    state->cols       = cols;
    state->basis_qr   = basis;
    state->tau        = tau;
    state->poly_basis = poly;
    state->fitted     = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_emsc_state_apply(const n4m_pp_emsc_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    const int32_t m = state->m;
    const int32_t degree = state->degree;

    /* Working buffers. The Q^T application overwrites a copy of x_i;
     * coefficients c live in a separate length-m buffer. */
    double* qt_x = (double*)malloc((size_t)cols * sizeof(double));
    if (qt_x == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* c = (double*)malloc((size_t)m * sizeof(double));
    if (c == NULL) {
        free(qt_x);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + (size_t)i * (size_t)cols;
        memcpy(qt_x, xrow, (size_t)cols * sizeof(double));
        {
            const n4m_status_t qst = n4m_apply_qt(
                state->basis_qr, cols, (int64_t)m, state->tau, qt_x);
            if (qst != N4M_OK) {
                free(qt_x);
                free(c);
                return qst;
            }
        }

        const n4m_status_t st = n4m_back_solve_R(state->basis_qr, cols,
                                                  (int64_t)m, qt_x, c);
        if (st != N4M_OK) {
            free(qt_x);
            free(c);
            return st;
        }
        if (c[0] == 0.0) {
            free(qt_x);
            free(c);
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        /* Use true division by c[0] (not multiplication by 1/c[0]) to preserve
         * NumPy's per-element rounding sequence; matches the same discipline
         * applied across the Phase 2/3 scatter and derivative engines. */
        const double c0    = c[0];
        double*      orow  = out + (size_t)i * (size_t)cols;

        for (int64_t j = 0; j < cols; ++j) {
            const double* prow = state->poly_basis +
                                  (size_t)j * (size_t)degree;
            double poly_val = 0.0;
            for (int32_t d = 0; d < degree; ++d) {
                poly_val += c[d + 1] * prow[d];
            }
            orow[j] = (xrow[j] - poly_val) / c0;
        }
    }

    free(qt_x);
    free(c);
    return N4M_OK;
}
