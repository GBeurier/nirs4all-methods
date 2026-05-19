/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Minimal PCA wrapper over c4a_svd_compact. See pca_helper.h for the
 * contract.
 */

#include "pca_helper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/svd.h"

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

/* Apply the sklearn sign convention: per principal axis (row of components),
 * the entry with the largest absolute value is non-negative. If we need to
 * flip a row, we also flip the corresponding column of the scores buffer
 * (when provided) so X = scores @ components remains invariant. */
static void flip_signs_max_abs(double* components, double* scores,
                                int64_t k, int64_t cols, int64_t n_scores) {
    for (int64_t i = 0; i < k; ++i) {
        double* row = components + (size_t)i * (size_t)cols;
        int64_t arg = 0;
        double best = fabs(row[0]);
        for (int64_t j = 1; j < cols; ++j) {
            const double v = fabs(row[j]);
            if (v > best) {
                best = v;
                arg = j;
            }
        }
        if (row[arg] < 0.0) {
            for (int64_t j = 0; j < cols; ++j) row[j] = -row[j];
            if (scores != NULL) {
                for (int64_t r = 0; r < n_scores; ++r) {
                    scores[(size_t)r * (size_t)k + (size_t)i] =
                        -scores[(size_t)r * (size_t)k + (size_t)i];
                }
            }
        }
    }
}

c4a_status_t c4a_pca_fit(const double* X, int64_t rows, int64_t cols,
                          int64_t n_components, int center,
                          c4a_pca_fit_t* out) {
    if (out == NULL) return C4A_ERR_NULL_POINTER;
    memset(out, 0, sizeof(*out));
    if (rows < 2 || cols < 1) return C4A_ERR_INVALID_ARGUMENT;
    if (X == NULL) return C4A_ERR_NULL_POINTER;

    int64_t k_max = rows;
    if (cols < k_max) k_max = cols;
    int64_t k = (n_components > 0) ? n_components : k_max;
    if (k > k_max) k = k_max;
    if (k < 1) k = 1;

    double* mean = (double*)malloc((size_t)cols * sizeof(double));
    double* Xc   = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (mean == NULL || Xc == NULL) {
        free(mean); free(Xc);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    if (center) {
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
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * (size_t)cols;
            double* crow = Xc + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                crow[j] = row[j] - mean[j];
            }
        }
    } else {
        for (int64_t j = 0; j < cols; ++j) mean[j] = 0.0;
        memcpy(Xc, X, (size_t)rows * (size_t)cols * sizeof(double));
    }

    const int64_t k_full = (rows < cols) ? rows : cols;

    if (use_truncated_svd(rows, cols, k_full, k)) {
        double* U  = (double*)malloc((size_t)rows * (size_t)k * sizeof(double));
        double* S  = (double*)malloc((size_t)k * sizeof(double));
        double* Vt = (double*)malloc((size_t)k * (size_t)cols * sizeof(double));
        if (U == NULL || S == NULL || Vt == NULL) {
            free(mean); free(Xc); free(U); free(S); free(Vt);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        const c4a_status_t sst = use_dual_wide_svd(rows, cols, k_full, k)
            ? c4a_svd_truncated_dual_wide(Xc, rows, cols, k, U, S, Vt)
            : c4a_svd_truncated_randomized(Xc, rows, cols, k, U, S, Vt);
        if (sst != C4A_OK) {
            free(mean); free(Xc); free(U); free(S); free(Vt);
            return sst;
        }

        double* components =
            (double*)malloc((size_t)k * (size_t)cols * sizeof(double));
        double* ev = (double*)malloc((size_t)k * sizeof(double));
        double* scores_train =
            (double*)malloc((size_t)rows * (size_t)k * sizeof(double));
        if (components == NULL || ev == NULL || scores_train == NULL) {
            free(mean); free(Xc); free(U); free(S); free(Vt);
            free(components); free(ev); free(scores_train);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        memcpy(components, Vt, (size_t)k * (size_t)cols * sizeof(double));
        const double n_minus_1 = (double)(rows - 1);
        for (int64_t i = 0; i < k; ++i) {
            ev[i] = (n_minus_1 > 0.0) ? (S[i] * S[i] / n_minus_1) : 0.0;
            for (int64_t r = 0; r < rows; ++r) {
                scores_train[(size_t)r * (size_t)k + (size_t)i] =
                    U[(size_t)r * (size_t)k + (size_t)i] * S[i];
            }
        }
        flip_signs_max_abs(components, scores_train, k, cols, rows);

        out->rows = rows;
        out->cols = cols;
        out->k = k;
        out->mean = mean;
        out->components = components;
        out->scores = scores_train;
        out->explained_var = ev;

        free(Xc); free(U); free(S); free(Vt);
        return C4A_OK;
    }

    double* U  = (double*)malloc((size_t)rows * (size_t)k_full * sizeof(double));
    double* S  = (double*)malloc((size_t)k_full * sizeof(double));
    double* Vt = (double*)malloc((size_t)k_full * (size_t)cols * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(mean); free(Xc); free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const c4a_status_t sst = c4a_svd_compact(Xc, rows, cols, U, S, Vt);
    if (sst != C4A_OK) {
        free(mean); free(Xc); free(U); free(S); free(Vt);
        return sst;
    }

    /* Components: top-k rows of Vt -> (k, cols) */
    double* components = (double*)malloc((size_t)k * (size_t)cols * sizeof(double));
    double* ev = (double*)malloc((size_t)k * sizeof(double));
    /* Training scores: U[:, :k] * S[:k] -> (rows, k) */
    double* scores_train = (double*)malloc((size_t)rows * (size_t)k * sizeof(double));
    if (components == NULL || ev == NULL || scores_train == NULL) {
        free(mean); free(Xc); free(U); free(S); free(Vt);
        free(components); free(ev); free(scores_train);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            components[(size_t)i * (size_t)cols + (size_t)j] =
                Vt[(size_t)i * (size_t)cols + (size_t)j];
        }
        const double s_i = S[i];
        const double n_minus_1 = (double)(rows - 1);
        ev[i] = (n_minus_1 > 0.0) ? (s_i * s_i / n_minus_1) : 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            scores_train[(size_t)r * (size_t)k + (size_t)i] =
                U[(size_t)r * (size_t)k_full + (size_t)i] * s_i;
        }
    }
    /* sklearn-style sign convention on the components (anchor on the
     * component row, not on U). Adjust the training scores in lockstep so
     * the X = scores @ components decomposition stays consistent (the
     * scores buffer itself is local and only used to flip the components
     * here). */
    flip_signs_max_abs(components, scores_train, k, cols, rows);

    out->rows = rows;
    out->cols = cols;
    out->k = k;
    out->mean = mean;
    out->components = components;
    out->scores = scores_train;
    out->explained_var = ev;

    free(Xc); free(U); free(S); free(Vt);
    return C4A_OK;
}

void c4a_pca_fit_free(c4a_pca_fit_t* fit) {
    if (fit == NULL) return;
    free(fit->mean);          fit->mean = NULL;
    free(fit->components);    fit->components = NULL;
    free(fit->scores);        fit->scores = NULL;
    free(fit->explained_var); fit->explained_var = NULL;
    fit->rows = fit->cols = fit->k = 0;
}

c4a_status_t c4a_pca_transform(const c4a_pca_fit_t* fit,
                                const double* X_new, int64_t n_rows,
                                double* scores) {
    if (fit == NULL || scores == NULL) return C4A_ERR_NULL_POINTER;
    if (n_rows < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (n_rows == 0) return C4A_OK;
    if (X_new == NULL) return C4A_ERR_NULL_POINTER;
    const int64_t cols = fit->cols;
    const int64_t k = fit->k;
    if (k <= 8) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const double* row = X_new + (size_t)r * (size_t)cols;
            double* srow = scores + (size_t)r * (size_t)k;
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int64_t j = 0; j < cols; ++j) {
                const double xj = row[j] - fit->mean[j];
                for (int64_t i = 0; i < k; ++i) {
                    acc[i] += xj * fit->components[(size_t)i * (size_t)cols + (size_t)j];
                }
            }
            for (int64_t i = 0; i < k; ++i) {
                srow[i] = acc[i];
            }
        }
        return C4A_OK;
    }
    for (int64_t r = 0; r < n_rows; ++r) {
        const double* row = X_new + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < k; ++i) {
            const double* comp = fit->components + (size_t)i * (size_t)cols;
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                s += comp[j] * (row[j] - fit->mean[j]);
            }
            scores[(size_t)r * (size_t)k + (size_t)i] = s;
        }
    }
    return C4A_OK;
}

c4a_status_t c4a_pca_inverse_transform(const c4a_pca_fit_t* fit,
                                        const double* scores,
                                        int64_t n_rows,
                                        double* X_hat) {
    if (fit == NULL || X_hat == NULL) return C4A_ERR_NULL_POINTER;
    if (n_rows < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (n_rows == 0) return C4A_OK;
    if (scores == NULL) return C4A_ERR_NULL_POINTER;
    const int64_t cols = fit->cols;
    const int64_t k = fit->k;
    for (int64_t r = 0; r < n_rows; ++r) {
        const double* s_row = scores + (size_t)r * (size_t)k;
        double* x_row = X_hat + (size_t)r * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            double v = fit->mean[j];
            for (int64_t i = 0; i < k; ++i) {
                v += s_row[i] * fit->components[(size_t)i * (size_t)cols + (size_t)j];
            }
            x_row[j] = v;
        }
    }
    return C4A_OK;
}
