/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Orthogonal Signal Correction (Direct OSC / DOSC) reference engine.
 *
 * See osc.h for the algorithm sketch. The implementation mirrors
 * nirs4all.operators.transforms.orthogonalization.OSC with one engineered
 * difference (the Phase 1 design ban on embedded PLS): the "PLS weight"
 * step is implemented as the leading right-singular vector of the rank-1
 * matrix `yc x^T` after deflation, which for univariate y reduces
 * arithmetically to `X^T y / ||X^T y||` — the same direction nirs4all
 * computes. We document this equivalence in docs/algorithms/osc.md.
 */

#include "osc.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_osc_state_t {
    int     fitted;
    int     scale;
    int32_t n_components_request;  /* what the user asked for */
    int32_t n_components_actual;   /* what we could fit (clipped) */
    int32_t n_components_storage;  /* stride of W/P along the component axis */
    int64_t cols;                   /* feature count from fit */
    double* X_mean;                /* length cols */
    double* X_std;                 /* length cols (all-ones when scale=0) */
    double  y_mean;                /* scalar (univariate y) */
    double  y_std;                 /* scalar */
    double* W_ortho;               /* (cols x n_components_storage) row-major */
    double* P_ortho;               /* (cols x n_components_storage) row-major */
};

static const double kOSC_EPS = 1e-10;
static const double kOSC_STD_FLOOR = 1e-10;

c4a_pp_osc_state_t* c4a_pp_osc_state_new(int32_t n_components, int scale) {
    if (n_components < 1) {
        return NULL;
    }
    c4a_pp_osc_state_t* s = (c4a_pp_osc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fitted = 0;
    s->scale = scale ? 1 : 0;
    s->n_components_request = n_components;
    s->n_components_actual = 0;
    s->n_components_storage = 0;
    s->cols = 0;
    s->X_mean = NULL;
    s->X_std = NULL;
    s->y_mean = 0.0;
    s->y_std = 1.0;
    s->W_ortho = NULL;
    s->P_ortho = NULL;
    return s;
}

void c4a_pp_osc_state_free(c4a_pp_osc_state_t* state) {
    if (state == NULL) return;
    free(state->X_mean);
    free(state->X_std);
    free(state->W_ortho);
    free(state->P_ortho);
    free(state);
}

int c4a_pp_osc_state_is_fitted(const c4a_pp_osc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

static double dot_n(const double* a, const double* b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

c4a_status_t c4a_pp_osc_state_fit(c4a_pp_osc_state_t* state,
                                   const double* X,
                                   const double* y,
                                   int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL || y == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Effective n_components = min(requested, cols-1, rows-2). */
    int32_t n_eff = state->n_components_request;
    const int64_t max_feasible_a = cols - 1;
    const int64_t max_feasible_b = rows - 2;
    int64_t max_feasible = (max_feasible_a < max_feasible_b) ? max_feasible_a
                                                              : max_feasible_b;
    if (max_feasible < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if ((int64_t)n_eff > max_feasible) {
        n_eff = (int32_t)max_feasible;
    }

    double* X_mean = (double*)calloc((size_t)cols, sizeof(double));
    double* X_std  = (double*)malloc((size_t)cols * sizeof(double));
    double* Xc     = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    double* yc     = (double*)malloc((size_t)rows * sizeof(double));
    double* w_pls  = (double*)malloc((size_t)cols * sizeof(double));
    double* W_ortho = (double*)calloc((size_t)cols * (size_t)n_eff, sizeof(double));
    double* P_ortho = (double*)calloc((size_t)cols * (size_t)n_eff, sizeof(double));
    double* w      = (double*)malloc((size_t)cols * sizeof(double));
    double* p      = (double*)malloc((size_t)cols * sizeof(double));
    double* w_ortho = (double*)malloc((size_t)cols * sizeof(double));
    double* p_ortho = (double*)malloc((size_t)cols * sizeof(double));
    double* t_score = (double*)malloc((size_t)rows * sizeof(double));
    double* t_ortho = (double*)malloc((size_t)rows * sizeof(double));
    if (!X_mean || !X_std || !Xc || !yc || !w_pls || !W_ortho || !P_ortho ||
        !w || !p || !w_ortho || !p_ortho || !t_score || !t_ortho) {
        free(X_mean); free(X_std); free(Xc); free(yc); free(w_pls);
        free(W_ortho); free(P_ortho); free(w); free(p); free(w_ortho);
        free(p_ortho); free(t_score); free(t_ortho);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    const double rows_d = (double)rows;
    const double ddof   = (double)(rows - 1);  /* sample std with ddof=1 */

    /* ---- Center + scale X (per column) -------------------------------- */
    if (state->scale) {
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += X[(size_t)i * (size_t)cols + (size_t)j];
            }
            X_mean[j] = s / rows_d;
        }
        for (int64_t j = 0; j < cols; ++j) {
            double s2 = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                const double d = X[(size_t)i * (size_t)cols + (size_t)j] - X_mean[j];
                s2 += d * d;
            }
            double std = (ddof > 0.0) ? sqrt(s2 / ddof) : 0.0;
            if (std < kOSC_STD_FLOOR) std = 1.0;
            X_std[j] = std;
        }
    } else {
        for (int64_t j = 0; j < cols; ++j) { X_mean[j] = 0.0; X_std[j] = 1.0; }
    }
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            const double x = X[(size_t)i * (size_t)cols + (size_t)j];
            Xc[(size_t)i * (size_t)cols + (size_t)j] = (x - X_mean[j]) / X_std[j];
        }
    }

    /* ---- Center + scale y --------------------------------------------- */
    double y_mean = 0.0, y_std = 1.0;
    if (state->scale) {
        for (int64_t i = 0; i < rows; ++i) y_mean += y[i];
        y_mean /= rows_d;
        double s2 = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double d = y[i] - y_mean;
            s2 += d * d;
        }
        y_std = (ddof > 0.0) ? sqrt(s2 / ddof) : 0.0;
        if (y_std < kOSC_STD_FLOOR) y_std = 1.0;
    } else {
        y_mean = 0.0;
        y_std  = 1.0;
    }
    for (int64_t i = 0; i < rows; ++i) {
        yc[i] = (y[i] - y_mean) / y_std;
    }

    /* ---- Initial Y-predictive direction w_pls -------------------------- */
    for (int64_t j = 0; j < cols; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            s += Xc[(size_t)i * (size_t)cols + (size_t)j] * yc[i];
        }
        w_pls[j] = s;
    }
    double norm = sqrt(dot_n(w_pls, w_pls, cols));
    if (norm < kOSC_EPS) {
        free(X_mean); free(X_std); free(Xc); free(yc); free(w_pls);
        free(W_ortho); free(P_ortho); free(w); free(p); free(w_ortho);
        free(p_ortho); free(t_score); free(t_ortho);
        return C4A_ERR_NUMERICAL_FAILURE;
    }
    for (int64_t j = 0; j < cols; ++j) w_pls[j] /= norm;

    /* ---- Iterate orthogonal-component extraction ---------------------- */
    int32_t n_extracted = 0;
    for (int32_t iter = 0; iter < n_eff; ++iter) {
        /* w = X_res^T yc / ||...|| */
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += Xc[(size_t)i * (size_t)cols + (size_t)j] * yc[i];
            }
            w[j] = s;
        }
        double w_norm = sqrt(dot_n(w, w, cols));
        if (w_norm < kOSC_EPS) break;
        for (int64_t j = 0; j < cols; ++j) w[j] /= w_norm;

        /* t = X_res @ w (length rows) */
        for (int64_t i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                s += Xc[(size_t)i * (size_t)cols + (size_t)j] * w[j];
            }
            t_score[i] = s;
        }
        const double t_norm_sq = dot_n(t_score, t_score, rows);
        if (t_norm_sq < kOSC_EPS) break;

        /* p = X_res^T @ t / t_norm_sq */
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += Xc[(size_t)i * (size_t)cols + (size_t)j] * t_score[i];
            }
            p[j] = s / t_norm_sq;
        }

        /* w_ortho = p - (p . w_pls) * w_pls; normalize */
        const double pw = dot_n(p, w_pls, cols);
        for (int64_t j = 0; j < cols; ++j) w_ortho[j] = p[j] - pw * w_pls[j];
        const double wo_norm = sqrt(dot_n(w_ortho, w_ortho, cols));
        if (wo_norm < kOSC_EPS) break;
        for (int64_t j = 0; j < cols; ++j) w_ortho[j] /= wo_norm;

        /* t_ortho = X_res @ w_ortho */
        for (int64_t i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                s += Xc[(size_t)i * (size_t)cols + (size_t)j] * w_ortho[j];
            }
            t_ortho[i] = s;
        }
        const double to_norm_sq = dot_n(t_ortho, t_ortho, rows);
        if (to_norm_sq < kOSC_EPS) break;

        /* p_ortho = X_res^T @ t_ortho / t_ortho_norm_sq */
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += Xc[(size_t)i * (size_t)cols + (size_t)j] * t_ortho[i];
            }
            p_ortho[j] = s / to_norm_sq;
        }

        /* Deflate: X_res -= outer(t_ortho, p_ortho) */
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                Xc[(size_t)i * (size_t)cols + (size_t)j] -= t_ortho[i] * p_ortho[j];
            }
        }

        /* Store column iter of W_ortho, P_ortho (row-major (cols x n_eff)). */
        for (int64_t j = 0; j < cols; ++j) {
            W_ortho[(size_t)j * (size_t)n_eff + (size_t)iter] = w_ortho[j];
            P_ortho[(size_t)j * (size_t)n_eff + (size_t)iter] = p_ortho[j];
        }
        ++n_extracted;
    }

    /* Free old state if any, then move new buffers in. We keep the
     * (cols x n_eff) storage even when n_extracted < n_eff; the storage
     * stride is recorded separately so the transform loops know to skip
     * the trailing columns. */
    free(state->X_mean);
    free(state->X_std);
    free(state->W_ortho);
    free(state->P_ortho);

    state->cols = cols;
    state->X_mean = X_mean;
    state->X_std = X_std;
    state->y_mean = y_mean;
    state->y_std = y_std;
    state->n_components_actual = n_extracted;
    state->n_components_storage = n_eff;
    if (n_extracted == 0) {
        free(W_ortho);
        free(P_ortho);
        state->W_ortho = NULL;
        state->P_ortho = NULL;
        state->n_components_storage = 0;
    } else {
        state->W_ortho = W_ortho;
        state->P_ortho = P_ortho;
    }
    state->fitted = 1;

    free(Xc);
    free(yc);
    free(w_pls);
    free(w);
    free(p);
    free(w_ortho);
    free(p_ortho);
    free(t_score);
    free(t_ortho);
    return C4A_OK;
}

c4a_status_t c4a_pp_osc_state_apply(const c4a_pp_osc_state_t* state,
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

    const int32_t nc = state->n_components_actual;
    const int32_t stride = state->n_components_storage;
    /* Stage 1: copy (X - mean) / std (or just X if scale=0) into out. */
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            const double x = X[(size_t)i * (size_t)cols + (size_t)j];
            out[(size_t)i * (size_t)cols + (size_t)j] =
                (x - state->X_mean[j]) / state->X_std[j];
        }
    }
    /* Stage 2: for each orthogonal component apply deflation. */
    if (nc > 0) {
        double* t_ortho = (double*)malloc((size_t)rows * sizeof(double));
        if (t_ortho == NULL) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        for (int32_t k = 0; k < nc; ++k) {
            for (int64_t i = 0; i < rows; ++i) {
                double s = 0.0;
                for (int64_t j = 0; j < cols; ++j) {
                    s += out[(size_t)i * (size_t)cols + (size_t)j]
                       * state->W_ortho[(size_t)j * (size_t)stride + (size_t)k];
                }
                t_ortho[i] = s;
            }
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t j = 0; j < cols; ++j) {
                    out[(size_t)i * (size_t)cols + (size_t)j] -=
                        t_ortho[i]
                      * state->P_ortho[(size_t)j * (size_t)stride + (size_t)k];
                }
            }
        }
        free(t_ortho);
    }
    /* Stage 3: rescale back to original units (if scale). */
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out[(size_t)i * (size_t)cols + (size_t)j] =
                out[(size_t)i * (size_t)cols + (size_t)j] * state->X_std[j]
              + state->X_mean[j];
        }
    }
    return C4A_OK;
}
