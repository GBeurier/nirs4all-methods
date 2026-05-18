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

struct c4a_pp_emsc_state_t {
    int32_t  degree;
    int      fitted;
    int64_t  cols;
    int32_t  m;             /* basis dimensionality = degree + 1 */
    double*  basis_qr;      /* p * m, row-major: in-place Householder factor */
    double*  tau;           /* m, Householder scaling factors */
    double*  poly_basis;    /* p * degree, row-major: cached wavelengths^d
                             * columns for the subtraction step. */
};

c4a_pp_emsc_state_t* c4a_pp_emsc_state_new(int32_t degree) {
    if (degree < 1) {
        return NULL;
    }
    c4a_pp_emsc_state_t* s = (c4a_pp_emsc_state_t*)malloc(sizeof(*s));
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

void c4a_pp_emsc_state_free(c4a_pp_emsc_state_t* state) {
    if (state == NULL) return;
    free(state->basis_qr);
    free(state->tau);
    free(state->poly_basis);
    free(state);
}

int c4a_pp_emsc_state_is_fitted(const c4a_pp_emsc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

/* In-place Householder QR factorisation of a (p x m) row-major matrix A
 * with p >= m. After the call:
 *   - The upper triangle of A[0:m, 0:m] holds R (m x m upper triangular).
 *   - The lower triangle (rows m..p-1 in column k, plus the (k+1..m-1)
 *     entries of row k below the diagonal slot — actually, since this is
 *     row-major, the Householder vectors v_k for k=0..m-1 are stored in
 *     column k of A from row k+1 down: v_k[k] = 1 (implicit), v_k[i] =
 *     A[i, k] for i > k.
 *   - tau[k] = the Householder scaling for vector k.
 *
 * After QR, applying Q^T to a length-p vector b gives Q^T b in place.
 * Reflection k uses v_k stored as described, with v_k[k] = 1 implicit. */
static void householder_qr(double* A, int64_t p, int32_t m, double* tau) {
    for (int32_t k = 0; k < m; ++k) {
        /* Compute the norm of A[k:p, k]. */
        double sigma = 0.0;
        for (int64_t i = k; i < p; ++i) {
            const double v = A[(size_t)i * (size_t)m + (size_t)k];
            sigma += v * v;
        }
        sigma = sqrt(sigma);

        const double Akk = A[(size_t)k * (size_t)m + (size_t)k];
        const double alpha = (Akk >= 0.0) ? -sigma : sigma;

        if (sigma == 0.0) {
            tau[k] = 0.0;
            continue;
        }

        const double vk0 = Akk - alpha;            /* first component of v before scaling */
        const double v_norm_sq = vk0 * vk0 +
            (sigma * sigma - Akk * Akk);            /* = ||v||^2 */
        const double t = 2.0 / v_norm_sq * vk0 * vk0;
        /* We store the Householder vector v_k in column k from row k+1
         * onward; the (k, k) slot will hold R[k, k] = alpha. The v vector
         * has v[k] = vk0, but we encode it implicitly by storing v[i] /
         * vk0 in A[i, k] for i > k and we keep tau[k] = 2 * vk0^2 / ||v||^2.
         * Then a reflection on b is: b -= tau[k] * (v^T b / vk0^2) * v
         *                          = b -= tau[k] * (sum_i b[i]*v[i]/vk0) * (v/vk0)
         * with v[k]/vk0 = 1, v[i]/vk0 = A[i, k] (stored) for i > k. */
        const double inv_vk0 = 1.0 / vk0;
        A[(size_t)k * (size_t)m + (size_t)k] = alpha;
        for (int64_t i = k + 1; i < p; ++i) {
            A[(size_t)i * (size_t)m + (size_t)k] *= inv_vk0;
        }
        tau[k] = t;

        /* Apply this reflection to columns k+1..m-1 of A. For each such
         * column j: A[k:p, j] -= tau[k] * (v_normalised^T A[k:p, j]) * v_normalised
         * where v_normalised[k] = 1, v_normalised[i] = A[i, k] (i > k). */
        for (int32_t j = k + 1; j < m; ++j) {
            double dot = A[(size_t)k * (size_t)m + (size_t)j];  /* * 1 for v[k]=1 */
            for (int64_t i = k + 1; i < p; ++i) {
                dot += A[(size_t)i * (size_t)m + (size_t)k]
                     * A[(size_t)i * (size_t)m + (size_t)j];
            }
            const double coef = tau[k] * dot;
            A[(size_t)k * (size_t)m + (size_t)j] -= coef;
            for (int64_t i = k + 1; i < p; ++i) {
                A[(size_t)i * (size_t)m + (size_t)j] -=
                    coef * A[(size_t)i * (size_t)m + (size_t)k];
            }
        }
    }
}

/* Apply Q^T to a length-p vector b in place, where Q is represented by the
 * Householder vectors in the lower triangle of A (after householder_qr). */
static void apply_qt(const double* A, int64_t p, int32_t m,
                     const double* tau, double* b) {
    for (int32_t k = 0; k < m; ++k) {
        if (tau[k] == 0.0) continue;
        /* dot = v^T b with v[k]=1, v[i] = A[i, k] for i > k. */
        double dot = b[k];
        for (int64_t i = k + 1; i < p; ++i) {
            dot += A[(size_t)i * (size_t)m + (size_t)k] * b[i];
        }
        const double coef = tau[k] * dot;
        b[k] -= coef;
        for (int64_t i = k + 1; i < p; ++i) {
            b[i] -= coef * A[(size_t)i * (size_t)m + (size_t)k];
        }
    }
}

/* Back-substitute R c = y where R is the upper-triangular m x m factor in
 * the top-left block of A (row-major, stride m). */
static c4a_status_t back_substitute(const double* A, int32_t m,
                                     const double* y, double* c) {
    for (int32_t i = m - 1; i >= 0; --i) {
        double s = y[i];
        for (int32_t j = i + 1; j < m; ++j) {
            s -= A[(size_t)i * (size_t)m + (size_t)j] * c[j];
        }
        const double diag = A[(size_t)i * (size_t)m + (size_t)i];
        if (diag == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        c[i] = s / diag;
    }
    return C4A_OK;
}

c4a_status_t c4a_pp_emsc_state_fit(c4a_pp_emsc_state_t* state,
                                    const double* X,
                                    int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < (int64_t)(state->degree + 2)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int32_t m = state->m;
    const int64_t p = cols;
    const int32_t degree = state->degree;

    double* basis = (double*)malloc((size_t)p * (size_t)m * sizeof(double));
    if (basis == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    double* poly = (double*)malloc((size_t)p * (size_t)degree * sizeof(double));
    if (poly == NULL) {
        free(basis);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    double* tau = (double*)malloc((size_t)m * sizeof(double));
    if (tau == NULL) {
        free(basis);
        free(poly);
        return C4A_ERR_OUT_OF_MEMORY;
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
    householder_qr(basis, p, m, tau);

    /* Commit. */
    free(state->basis_qr);
    free(state->tau);
    free(state->poly_basis);
    state->cols       = cols;
    state->basis_qr   = basis;
    state->tau        = tau;
    state->poly_basis = poly;
    state->fitted     = 1;
    return C4A_OK;
}

c4a_status_t c4a_pp_emsc_state_apply(const c4a_pp_emsc_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    const int32_t m = state->m;
    const int32_t degree = state->degree;

    /* Working buffers. The Q^T application overwrites a copy of x_i;
     * coefficients c live in a separate length-m buffer. */
    double* qt_x = (double*)malloc((size_t)cols * sizeof(double));
    if (qt_x == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    double* c = (double*)malloc((size_t)m * sizeof(double));
    if (c == NULL) {
        free(qt_x);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + (size_t)i * (size_t)cols;
        memcpy(qt_x, xrow, (size_t)cols * sizeof(double));
        apply_qt(state->basis_qr, cols, m, state->tau, qt_x);

        const c4a_status_t st = back_substitute(state->basis_qr, m, qt_x, c);
        if (st != C4A_OK) {
            free(qt_x);
            free(c);
            return st;
        }
        if (c[0] == 0.0) {
            free(qt_x);
            free(c);
            return C4A_ERR_NUMERICAL_FAILURE;
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
    return C4A_OK;
}
