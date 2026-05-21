/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * External Parameter Orthogonalization (EPO) reference engine.
 *
 * See epo.h for the algorithm sketch. Univariate external parameter `d`
 * (length n_samples). At fit time we learn the projection
 *
 *     B[j] = (dc · Xc[:, j]) / (dc · dc),     j = 0..cols-1
 *
 * which is the scalar coefficient that minimises ||Xc[:, j] - dc * B[j]||^2
 * for each feature j. At `_transform` time we have no `d`, so we apply
 * centering + uncentering (a no-op) as documented by the nirs4all reference.
 * The training-time projection is exposed through `_apply_with_d` (an
 * internal helper used by the parity fixture's fit_transform path).
 */

#include "epo.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_pp_epo_state_t {
    int     fitted;
    int     scale;
    int64_t cols;
    int64_t rows;       /* row count seen at fit time */
    double* X_mean;     /* length cols (all-zeros when scale=0) */
    double  d_mean;
    double* B;          /* length cols — regression coefficient per feature */
};

static const double kEPO_VAR_FLOOR = 1e-20;

n4m_pp_epo_state_t* n4m_pp_epo_state_new(int scale) {
    n4m_pp_epo_state_t* s = (n4m_pp_epo_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fitted = 0;
    s->scale  = scale ? 1 : 0;
    s->cols   = 0;
    s->rows   = 0;
    s->X_mean = NULL;
    s->d_mean = 0.0;
    s->B      = NULL;
    return s;
}

void n4m_pp_epo_state_free(n4m_pp_epo_state_t* state) {
    if (state == NULL) return;
    free(state->X_mean);
    free(state->B);
    free(state);
}

int n4m_pp_epo_state_is_fitted(const n4m_pp_epo_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

n4m_status_t n4m_pp_epo_state_fit(n4m_pp_epo_state_t* state,
                                   const double* X,
                                   const double* d,
                                   int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL || d == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    double* X_mean = (double*)calloc((size_t)cols, sizeof(double));
    double* B      = (double*)malloc((size_t)cols * sizeof(double));
    if (X_mean == NULL || B == NULL) {
        free(X_mean); free(B);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    const double rows_d = (double)rows;

    /* ---- Compute X_mean and d_mean (skip centering when scale=0) -------- */
    double d_mean = 0.0;
    if (state->scale) {
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += X[(size_t)i * (size_t)cols + (size_t)j];
            }
            X_mean[j] = s / rows_d;
        }
        for (int64_t i = 0; i < rows; ++i) d_mean += d[i];
        d_mean /= rows_d;
    }

    /* ---- Compute denominator dc · dc ----------------------------------- */
    double dd = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double e = d[i] - d_mean;
        dd += e * e;
    }
    if (dd < kEPO_VAR_FLOOR) {
        free(X_mean); free(B);
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    /* ---- Compute B[j] = (dc · Xc[:, j]) / (dc · dc) -------------------- */
    for (int64_t j = 0; j < cols; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double xc = X[(size_t)i * (size_t)cols + (size_t)j] - X_mean[j];
            const double dc = d[i] - d_mean;
            s += dc * xc;
        }
        B[j] = s / dd;
    }

    /* Commit state. */
    free(state->X_mean);
    free(state->B);
    state->cols   = cols;
    state->rows   = rows;
    state->X_mean = X_mean;
    state->d_mean = d_mean;
    state->B      = B;
    state->fitted = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_epo_state_apply(const n4m_pp_epo_state_t* state,
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
    /* Without d at transform time, EPO reduces to centering + uncentering —
     * i.e. the output equals the input. We still take the centering /
     * uncentering walk so the algorithm is honest about its assumed
     * d = d_mean point. */
    if (state->scale) {
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                const double x = X[(size_t)i * (size_t)cols + (size_t)j];
                out[(size_t)i * (size_t)cols + (size_t)j] =
                    (x - state->X_mean[j]) + state->X_mean[j];
            }
        }
    } else {
        /* Pure pass-through when not scaling. */
        if (out != X) {
            memcpy(out, X,
                   (size_t)rows * (size_t)cols * sizeof(double));
        }
    }
    return N4M_OK;
}

n4m_status_t n4m_pp_epo_state_apply_with_d(const n4m_pp_epo_state_t* state,
                                            const double* X,
                                            const double* d,
                                            int64_t rows, int64_t cols,
                                            double* out) {
    if (state == NULL || X == NULL || d == NULL || out == NULL) {
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
    /* X_filtered = X - X_mean - outer(d - d_mean, B) + X_mean
     *            = X - outer(d - d_mean, B) */
    for (int64_t i = 0; i < rows; ++i) {
        const double dc = d[i] - state->d_mean;
        for (int64_t j = 0; j < cols; ++j) {
            const double x = X[(size_t)i * (size_t)cols + (size_t)j];
            out[(size_t)i * (size_t)cols + (size_t)j] = x - dc * state->B[j];
        }
    }
    return N4M_OK;
}
