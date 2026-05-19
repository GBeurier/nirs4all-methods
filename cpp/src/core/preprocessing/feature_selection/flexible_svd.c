/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FlexibleSVD — TruncatedSVD with flexible component specification.
 *
 * The number of components is interpreted from a single double:
 *   - integer value >= 1.0 ("count mode")
 *   - value in (0.0, 1.0) ("variance-ratio mode")
 *
 * The fit pipeline matches sklearn TruncatedSVD modulo sign convention:
 *
 *   1. U, S, Vt = svd(X, full_matrices=False)
 *   2. signs canonicalised with u_based_decision=True (inside svd.c)
 *   3. X_proj = U * S
 *   4. explained_variance = var(X_proj, axis=0)
 *   5. full_var = sum(var(X, axis=0))
 *   6. evar_ratio = explained_variance / full_var
 *   7. select k' (count or ratio mode)
 *   8. components_ = Vt[:k', :]
 *
 * Transform applies X @ components_.T row-by-row (no centering).
 *
 * Refit semantics: calling _fit twice releases the prior arrays before
 * installing the new ones.
 */

#include "flexible_svd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/svd.h"

struct c4a_pp_flex_svd_state_t {
    double  n_components_param;

    int      fitted;
    int64_t  n_features_in;
    int64_t  n_components;
    double*  components;          /* k' x n_features_in, row-major */
};

static int use_truncated_svd(int64_t rows, int64_t cols,
                             int64_t k_full, int64_t k) {
    const int64_t min_large_elements = 32768;
    if (k < 1 || k >= k_full) return 0;
    return rows > 0 && cols > 0 && rows >= min_large_elements / cols;
}

static int use_dual_wide_svd(int64_t rows, int64_t cols,
                             int64_t k_full, int64_t k) {
    return use_truncated_svd(rows, cols, k_full, k) &&
           rows <= cols && rows <= 256;
}

c4a_pp_flex_svd_state_t* c4a_pp_flex_svd_state_new(double n_components_param) {
    if (!(n_components_param > 0.0)) {  /* catches NaN and <= 0 */
        return NULL;
    }
    c4a_pp_flex_svd_state_t* s = (c4a_pp_flex_svd_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->n_components_param = n_components_param;
    s->fitted             = 0;
    s->n_features_in      = 0;
    s->n_components       = 0;
    s->components         = NULL;
    return s;
}

void c4a_pp_flex_svd_state_free(c4a_pp_flex_svd_state_t* state) {
    if (state == NULL) return;
    free(state->components);
    free(state);
}

int c4a_pp_flex_svd_state_is_fitted(const c4a_pp_flex_svd_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

int64_t c4a_pp_flex_svd_state_n_components(const c4a_pp_flex_svd_state_t* state) {
    return (state != NULL && state->fitted) ? state->n_components : 0;
}

/* Per-column variance, matching numpy.var(axis=0) with default ddof=0:
 *   var[j] = mean(X[:, j]^2) - mean(X[:, j])^2
 * Use a single pass over rows to limit memory traffic. */
static void column_variance(const double* X, int64_t rows, int64_t cols,
                            double* out_var) {
    const double rows_d = (double)rows;
    for (int64_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double v = X[(size_t)i * (size_t)cols + (size_t)j];
            sum    += v;
            sum_sq += v * v;
        }
        const double mean = sum / rows_d;
        out_var[j] = sum_sq / rows_d - mean * mean;
    }
}

static int64_t flex_svd_select_k(double param, int64_t k_full,
                                  const double* var_ratio) {
    if (param >= 1.0) {
        int64_t k = (int64_t)param;
        if (k < 1) k = 1;
        if (k > k_full) k = k_full;
        return k;
    }
    double cum = 0.0;
    for (int64_t i = 0; i < k_full; ++i) {
        cum += var_ratio[i];
        if (cum >= param) {
            return i + 1;
        }
    }
    return k_full;
}

c4a_status_t c4a_pp_flex_svd_state_fit(c4a_pp_flex_svd_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int64_t k_full = (rows < cols) ? rows : cols;

    if (state->n_components_param >= 1.0) {
        const int64_t k_kept =
            flex_svd_select_k(state->n_components_param, k_full, NULL);
        if (use_truncated_svd(rows, cols, k_full, k_kept)) {
            double* U  =
                (double*)malloc((size_t)rows * (size_t)k_kept * sizeof(double));
            double* S  = (double*)malloc((size_t)k_kept * sizeof(double));
            double* Vt =
                (double*)malloc((size_t)k_kept * (size_t)cols * sizeof(double));
            if (U == NULL || S == NULL || Vt == NULL) {
                free(U); free(S); free(Vt);
                return C4A_ERR_OUT_OF_MEMORY;
            }
            const c4a_status_t svd_st =
                use_dual_wide_svd(rows, cols, k_full, k_kept)
                    ? c4a_svd_truncated_dual_wide(
                          X, rows, cols, k_kept, U, S, Vt)
                    : c4a_svd_truncated_randomized(
                          X, rows, cols, k_kept, U, S, Vt);
            free(U);
            free(S);
            if (svd_st != C4A_OK) {
                free(Vt);
                return svd_st;
            }
            double* components =
                (double*)malloc((size_t)k_kept * (size_t)cols * sizeof(double));
            if (components == NULL) {
                free(Vt);
                return C4A_ERR_OUT_OF_MEMORY;
            }
            memcpy(components, Vt,
                   (size_t)k_kept * (size_t)cols * sizeof(double));
            free(Vt);

            free(state->components);
            state->n_features_in = cols;
            state->n_components  = k_kept;
            state->components    = components;
            state->fitted        = 1;
            return C4A_OK;
        }
    }

    /* Build a working copy of X so the full SVD can consume it in place. */
    double* Xwork =
        (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (Xwork == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    memcpy(Xwork, X, (size_t)rows * (size_t)cols * sizeof(double));

    double* U  = (double*)malloc((size_t)rows * (size_t)k_full * sizeof(double));
    double* S  = (double*)malloc((size_t)k_full * sizeof(double));
    double* Vt = (double*)malloc((size_t)k_full * (size_t)cols * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(U); free(S); free(Vt); free(Xwork);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    const c4a_status_t svd_st = c4a_svd_compact(Xwork, rows, cols, U, S, Vt);
    free(Xwork);
    if (svd_st != C4A_OK) {
        free(U); free(S); free(Vt);
        return svd_st;
    }

    /* explained_variance[k] = var(U[:, k] * S[k]) computed across rows.
     * For unit-norm U columns: sum(U[:, k]^2) = 1, so
     *   var(U[:, k] * S[k]) = S[k]^2 * (1/m - mean(U[:, k])^2)
     * but we keep the explicit two-pass numpy-style computation for
     * byte-aligned parity with the reference. */
    double* X_proj_var = (double*)malloc((size_t)k_full * sizeof(double));
    if (X_proj_var == NULL) {
        free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const double rows_d = (double)rows;
    for (int64_t k = 0; k < k_full; ++k) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double v = U[(size_t)i * (size_t)k_full + (size_t)k] * S[k];
            sum    += v;
            sum_sq += v * v;
        }
        const double mean = sum / rows_d;
        X_proj_var[k] = sum_sq / rows_d - mean * mean;
    }

    /* full_var = sum of per-column variances of the input matrix. */
    double* col_var = (double*)malloc((size_t)cols * sizeof(double));
    if (col_var == NULL) {
        free(X_proj_var); free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    column_variance(X, rows, cols, col_var);
    double full_var = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        full_var += col_var[j];
    }
    free(col_var);

    /* Compute variance ratio with safe divide. */
    double* var_ratio = (double*)malloc((size_t)k_full * sizeof(double));
    if (var_ratio == NULL) {
        free(X_proj_var); free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    if (full_var > 0.0) {
        const double inv = 1.0 / full_var;
        for (int64_t i = 0; i < k_full; ++i) {
            var_ratio[i] = X_proj_var[i] * inv;
        }
    } else {
        for (int64_t i = 0; i < k_full; ++i) {
            var_ratio[i] = 0.0;
        }
    }
    free(X_proj_var);

    const int64_t k_kept =
        flex_svd_select_k(state->n_components_param, k_full, var_ratio);
    free(var_ratio);

    double* components =
        (double*)malloc((size_t)k_kept * (size_t)cols * sizeof(double));
    if (components == NULL) {
        free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    memcpy(components, Vt, (size_t)k_kept * (size_t)cols * sizeof(double));

    free(U);
    free(S);
    free(Vt);

    /* Commit. */
    free(state->components);
    state->n_features_in = cols;
    state->n_components  = k_kept;
    state->components    = components;
    state->fitted        = 1;
    return C4A_OK;
}

c4a_status_t c4a_pp_flex_svd_state_apply(const c4a_pp_flex_svd_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->n_features_in) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (out_cols != state->n_components) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0 || out_cols == 0) {
        return C4A_OK;
    }

    if (state->n_components <= 8) {
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * (size_t)cols;
            double* orow = out + (size_t)i * (size_t)out_cols;
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int64_t j = 0; j < cols; ++j) {
                const double xj = row[j];
                for (int64_t k = 0; k < state->n_components; ++k) {
                    acc[k] += xj * state->components[(size_t)k * (size_t)cols + (size_t)j];
                }
            }
            for (int64_t k = 0; k < state->n_components; ++k) {
                orow[k] = acc[k];
            }
        }
        return C4A_OK;
    }

    /* out[i, k] = sum_j X[i, j] * components[k, j]. */
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double*       orow = out + (size_t)i * (size_t)out_cols;
        for (int64_t k = 0; k < state->n_components; ++k) {
            const double* comp_k = state->components + (size_t)k * (size_t)cols;
            double acc = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                acc += row[j] * comp_k[j];
            }
            orow[k] = acc;
        }
    }
    return C4A_OK;
}
