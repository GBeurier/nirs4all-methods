/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FlexiblePCA — PCA with flexible component specification.
 *
 * The number of components is interpreted from a single double:
 *   - integer value >= 1.0 ("count mode") -> keep exactly that many
 *     components, clamped to k = min(m, n).
 *   - value in (0.0, 1.0) ("variance-ratio mode") -> keep the smallest
 *     k' such that the cumulative variance ratio is >= the value.
 *
 * The fit pipeline matches sklearn PCA with ``svd_solver='full'``:
 *
 *   1. mean = X.mean(axis=0)
 *   2. Xc   = X - mean
 *   3. U, S, Vt = svd(Xc, full_matrices=False)   # via Jacobi (common/svd.c)
 *   4. signs canonicalised with u_based_decision=True (inside svd.c)
 *   5. evar  = S^2 / (m - 1)
 *      evar_ratio = evar / sum(evar)
 *   6. select k' (count or ratio mode)
 *   7. components_ = Vt[:k', :]
 *
 * Transform applies (X - mean) @ components_.T row-by-row.
 *
 * Refit semantics: calling _fit twice releases the prior arrays before
 * installing the new ones.
 */

#include "flexible_pca.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/svd.h"

struct n4m_pp_flex_pca_state_t {
    /* Constructor parameter — preserved verbatim so refits stay
     * stable across changes in the surrounding data. */
    double  n_components_param;

    /* Fitted state. */
    int      fitted;
    int64_t  n_features_in;        /* training cols */
    int64_t  n_components;         /* k' actually kept */
    double*  mean;                 /* length n_features_in */
    double*  components;           /* k' x n_features_in, row-major */
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

static int use_tall_gram_svd(int64_t rows, int64_t cols,
                             int64_t k_full, int64_t k) {
    (void)k_full;
    return k >= 1 && k < cols && rows >= cols && cols <= 64;
}

n4m_pp_flex_pca_state_t* n4m_pp_flex_pca_state_new(double n_components_param) {
    if (!(n_components_param > 0.0)) {  /* catches NaN and <= 0 */
        return NULL;
    }
    n4m_pp_flex_pca_state_t* s = (n4m_pp_flex_pca_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->n_components_param = n_components_param;
    s->fitted             = 0;
    s->n_features_in      = 0;
    s->n_components       = 0;
    s->mean               = NULL;
    s->components         = NULL;
    return s;
}

void n4m_pp_flex_pca_state_free(n4m_pp_flex_pca_state_t* state) {
    if (state == NULL) return;
    free(state->mean);
    free(state->components);
    free(state);
}

int n4m_pp_flex_pca_state_is_fitted(const n4m_pp_flex_pca_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

int64_t n4m_pp_flex_pca_state_n_components(const n4m_pp_flex_pca_state_t* state) {
    return (state != NULL && state->fitted) ? state->n_components : 0;
}

/* Decide how many components to keep, given the constructor parameter,
 * the full-rank k = min(m, n) and the per-component variance ratio
 * vector. */
static int64_t flex_pca_select_k(double param, int64_t k_full,
                                  const double* var_ratio) {
    if (param >= 1.0) {
        /* Count mode (sklearn truncates float n_components >= 1 to int). */
        int64_t k = (int64_t)param;
        if (k < 1) k = 1;
        if (k > k_full) k = k_full;
        return k;
    }
    /* Variance-ratio mode. Find smallest k' s.t. cumsum >= param. The
     * sklearn implementation:
     *   n_comp = searchsorted(cumsum, ratio) + 1
     * with ``side='left'``. searchsorted on ascending cumsum returns
     * the index of the first element >= ratio; adding 1 gives the
     * smallest cardinality. */
    double cum = 0.0;
    for (int64_t i = 0; i < k_full; ++i) {
        cum += var_ratio[i];
        if (cum >= param) {
            return i + 1;
        }
    }
    return k_full;
}

n4m_status_t n4m_pp_flex_pca_state_fit(n4m_pp_flex_pca_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int64_t k_full = (rows < cols) ? rows : cols;

    /* Compute column means. */
    double* mean = (double*)malloc((size_t)cols * sizeof(double));
    if (mean == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) mean[j] = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            mean[j] += row[j];
        }
    }
    for (int64_t j = 0; j < cols; ++j) {
        mean[j] /= (double)rows;
    }

    /* Build centered matrix in scratch (SVD consumes it in-place). */
    double* Xc = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (Xc == NULL) {
        free(mean);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double* crow = Xc + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            crow[j] = row[j] - mean[j];
        }
    }

    if (state->n_components_param >= 1.0) {
        const int64_t k_kept =
            flex_pca_select_k(state->n_components_param, k_full, NULL);
        if (use_tall_gram_svd(rows, cols, k_full, k_kept) ||
            use_truncated_svd(rows, cols, k_full, k_kept)) {
            double* U  =
                (double*)malloc((size_t)rows * (size_t)k_kept * sizeof(double));
            double* S  = (double*)malloc((size_t)k_kept * sizeof(double));
            double* Vt =
                (double*)malloc((size_t)k_kept * (size_t)cols * sizeof(double));
            if (U == NULL || S == NULL || Vt == NULL) {
                free(U); free(S); free(Vt);
                free(Xc); free(mean);
                return N4M_ERR_OUT_OF_MEMORY;
            }
            const n4m_status_t svd_st =
                use_tall_gram_svd(rows, cols, k_full, k_kept)
                    ? n4m_svd_truncated_tall_gram(
                          Xc, rows, cols, k_kept, U, S, Vt)
                    : (use_dual_wide_svd(rows, cols, k_full, k_kept)
                           ? n4m_svd_truncated_dual_wide(
                                 Xc, rows, cols, k_kept, U, S, Vt)
                           : n4m_svd_truncated_randomized(
                                 Xc, rows, cols, k_kept, U, S, Vt));
            free(Xc);
            free(U);
            free(S);
            if (svd_st != N4M_OK) {
                free(Vt); free(mean);
                return svd_st;
            }
            free(state->mean);
            free(state->components);
            state->n_features_in = cols;
            state->n_components  = k_kept;
            state->mean          = mean;
            state->components    = Vt;
            state->fitted        = 1;
            return N4M_OK;
        }
    }

    /* Allocate SVD outputs. */
    double* U  = (double*)malloc((size_t)rows * (size_t)k_full * sizeof(double));
    double* S  = (double*)malloc((size_t)k_full * sizeof(double));
    double* Vt = (double*)malloc((size_t)k_full * (size_t)cols * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(U); free(S); free(Vt);
        free(Xc); free(mean);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    const n4m_status_t svd_st = n4m_svd_compact(Xc, rows, cols, U, S, Vt);
    free(Xc);  /* SVD has consumed it */
    if (svd_st != N4M_OK) {
        free(U); free(S); free(Vt); free(mean);
        return svd_st;
    }

    /* Explained variance and ratio. evar = S^2 / (n_samples - 1). */
    double* evar       = (double*)malloc((size_t)k_full * sizeof(double));
    double* var_ratio  = (double*)malloc((size_t)k_full * sizeof(double));
    if (evar == NULL || var_ratio == NULL) {
        free(evar); free(var_ratio);
        free(U); free(S); free(Vt); free(mean);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    const double inv_dof = 1.0 / (double)(rows - 1);
    double total = 0.0;
    for (int64_t i = 0; i < k_full; ++i) {
        evar[i] = S[i] * S[i] * inv_dof;
        total  += evar[i];
    }
    if (total > 0.0) {
        const double inv_total = 1.0 / total;
        for (int64_t i = 0; i < k_full; ++i) {
            var_ratio[i] = evar[i] * inv_total;
        }
    } else {
        for (int64_t i = 0; i < k_full; ++i) {
            var_ratio[i] = 0.0;
        }
    }

    /* Select k' and copy the top-k' rows of Vt as the components. */
    const int64_t k_kept =
        flex_pca_select_k(state->n_components_param, k_full, var_ratio);
    double* components =
        (double*)malloc((size_t)k_kept * (size_t)cols * sizeof(double));
    if (components == NULL) {
        free(evar); free(var_ratio);
        free(U); free(S); free(Vt); free(mean);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    memcpy(components, Vt, (size_t)k_kept * (size_t)cols * sizeof(double));

    free(evar);
    free(var_ratio);
    free(U);
    free(S);
    free(Vt);

    /* Commit. */
    free(state->mean);
    free(state->components);
    state->n_features_in = cols;
    state->n_components  = k_kept;
    state->mean          = mean;
    state->components    = components;
    state->fitted        = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_flex_pca_state_apply(const n4m_pp_flex_pca_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->n_features_in) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (out_cols != state->n_components) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0 || out_cols == 0) {
        return N4M_OK;
    }

    if (state->n_components == 5) {
        const double* comp0 = state->components;
        const double* comp1 = comp0 + (size_t)cols;
        const double* comp2 = comp1 + (size_t)cols;
        const double* comp3 = comp2 + (size_t)cols;
        const double* comp4 = comp3 + (size_t)cols;
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * (size_t)cols;
            double* orow = out + (size_t)i * (size_t)out_cols;
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0, acc4 = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                const double xj = row[j] - state->mean[j];
                acc0 += xj * comp0[j];
                acc1 += xj * comp1[j];
                acc2 += xj * comp2[j];
                acc3 += xj * comp3[j];
                acc4 += xj * comp4[j];
            }
            orow[0] = acc0;
            orow[1] = acc1;
            orow[2] = acc2;
            orow[3] = acc3;
            orow[4] = acc4;
        }
        return N4M_OK;
    }

    if (state->n_components <= 8) {
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * (size_t)cols;
            double* orow = out + (size_t)i * (size_t)out_cols;
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int64_t j = 0; j < cols; ++j) {
                const double xj = row[j] - state->mean[j];
                for (int64_t k = 0; k < state->n_components; ++k) {
                    acc[k] += xj * state->components[(size_t)k * (size_t)cols + (size_t)j];
                }
            }
            for (int64_t k = 0; k < state->n_components; ++k) {
                orow[k] = acc[k];
            }
        }
        return N4M_OK;
    }

    /* out[i, k] = sum_j (X[i, j] - mean[j]) * components[k, j]. */
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double*       orow = out + (size_t)i * (size_t)out_cols;
        for (int64_t k = 0; k < state->n_components; ++k) {
            const double* comp_k = state->components + (size_t)k * (size_t)cols;
            double acc = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                acc += (row[j] - state->mean[j]) * comp_k[j];
            }
            orow[k] = acc;
        }
    }
    return N4M_OK;
}
