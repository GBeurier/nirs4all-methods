/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Local Outlier Factor — vendored minimal implementation. See header.
 *
 * State held after fit:
 *   - train_X    : copy of training matrix.
 *   - k_distance : per-training-sample distance to its k-th nearest neighbour.
 *   - lrd        : local reachability density per training sample.
 *
 * Scoring uses the same kNN search against the training set with the cached
 * lrd values.
 */

#include "lof.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_lof_t {
    double* train_X;
    int64_t rows;
    int64_t cols;
    int     k;
    double* k_distance;   /* (rows,) */
    double* lrd;          /* (rows,) */
};

/* Compute squared Euclidean distance between two rows of length cols. */
static double sq_eucl(const double* a, const double* b, int64_t cols) {
    double s = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

/* Partial selection (Hoare partition) — find the k smallest values in
 * ``dist[0..n)`` and reorder both ``dist`` and ``idx`` so the first k
 * positions hold them. After the call the first k positions are in
 * ascending order. Stable ties broken by sample index. O(n + k log k). */
static void partial_sort_k_with_indices(double* dist, int32_t* idx,
                                          int32_t n, int k) {
    if (k <= 0) return;
    if (k > n) k = n;
    /* Linear partial-selection: for each of the k slots find the smallest
     * remaining (with sample-index tie-break) and swap to the front. */
    for (int i = 0; i < k; ++i) {
        int best = i;
        double best_d = dist[i];
        int32_t best_idx = idx[i];
        for (int j = i + 1; j < n; ++j) {
            const double d = dist[j];
            const int32_t ij = idx[j];
            if (d < best_d || (d == best_d && ij < best_idx)) {
                best = j;
                best_d = d;
                best_idx = ij;
            }
        }
        if (best != i) {
            const double td = dist[i]; dist[i] = dist[best]; dist[best] = td;
            const int32_t ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        }
    }
}

n4m_lof_t* n4m_lof_new(void) {
    n4m_lof_t* l = (n4m_lof_t*)calloc(1, sizeof(n4m_lof_t));
    return l;
}

void n4m_lof_free(n4m_lof_t* lof) {
    if (lof == NULL) return;
    free(lof->train_X);
    free(lof->k_distance);
    free(lof->lrd);
    free(lof);
}

n4m_status_t n4m_lof_fit(n4m_lof_t* lof,
                          const double* X, int64_t rows, int64_t cols,
                          int k) {
    if (lof == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 2 || cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    if (X == NULL) return N4M_ERR_NULL_POINTER;
    if (k < 1) return N4M_ERR_INVALID_ARGUMENT;
    if ((int64_t)k > rows - 1) k = (int)(rows - 1);

    free(lof->train_X);
    free(lof->k_distance);
    free(lof->lrd);
    lof->train_X = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    lof->k_distance = (double*)malloc((size_t)rows * sizeof(double));
    lof->lrd = (double*)malloc((size_t)rows * sizeof(double));
    if (lof->train_X == NULL || lof->k_distance == NULL || lof->lrd == NULL) {
        free(lof->train_X);   lof->train_X = NULL;
        free(lof->k_distance);lof->k_distance = NULL;
        free(lof->lrd);       lof->lrd = NULL;
        return N4M_ERR_OUT_OF_MEMORY;
    }
    memcpy(lof->train_X, X, (size_t)rows * (size_t)cols * sizeof(double));
    lof->rows = rows;
    lof->cols = cols;
    lof->k = k;

    /* Build the k-distance + neighbour list for every training sample. */
    double* dist_buf = (double*)malloc((size_t)rows * sizeof(double));
    int32_t* idx_buf = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (dist_buf == NULL || idx_buf == NULL) {
        free(dist_buf); free(idx_buf);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    /* We store the kNN indices in a flat (rows, k) buffer so we can compute
     * lrd in two passes. */
    int32_t* knn = (int32_t*)malloc((size_t)rows * (size_t)k * sizeof(int32_t));
    if (knn == NULL) {
        free(dist_buf); free(idx_buf);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* xi = lof->train_X + (size_t)i * (size_t)cols;
        int32_t n_valid = 0;
        for (int64_t j = 0; j < rows; ++j) {
            if (j == i) continue;
            const double* xj = lof->train_X + (size_t)j * (size_t)cols;
            dist_buf[n_valid] = sqrt(sq_eucl(xi, xj, cols));
            idx_buf[n_valid] = (int32_t)j;
            ++n_valid;
        }
        partial_sort_k_with_indices(dist_buf, idx_buf, n_valid, k);
        lof->k_distance[i] = dist_buf[k - 1];
        for (int t = 0; t < k; ++t) {
            knn[(size_t)i * (size_t)k + (size_t)t] = idx_buf[t];
        }
    }

    /* lrd_i = 1 / mean reach-distance(i, j) for j in N_k(i). */
    for (int64_t i = 0; i < rows; ++i) {
        const double* xi = lof->train_X + (size_t)i * (size_t)cols;
        double sum = 0.0;
        for (int t = 0; t < k; ++t) {
            const int32_t j = knn[(size_t)i * (size_t)k + (size_t)t];
            const double* xj = lof->train_X + (size_t)j * (size_t)cols;
            const double dij = sqrt(sq_eucl(xi, xj, cols));
            const double reach = (lof->k_distance[j] > dij)
                                  ? lof->k_distance[j] : dij;
            sum += reach;
        }
        const double mean = sum / (double)k;
        lof->lrd[i] = (mean > 0.0) ? (1.0 / mean) : 0.0;
    }

    free(knn);
    free(dist_buf);
    free(idx_buf);
    return N4M_OK;
}

n4m_status_t n4m_lof_score(const n4m_lof_t* lof,
                            const double* X_new, int64_t rows_new,
                            int64_t cols,
                            double* scores) {
    if (lof == NULL || scores == NULL) return N4M_ERR_NULL_POINTER;
    if (rows_new < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows_new == 0) return N4M_OK;
    if (X_new == NULL) return N4M_ERR_NULL_POINTER;
    if (cols != lof->cols) return N4M_ERR_SHAPE_MISMATCH;
    if (lof->lrd == NULL) return N4M_ERR_NOT_FITTED;
    const int k = lof->k;
    const int64_t train_rows = lof->rows;
    double* dist_buf = (double*)malloc((size_t)train_rows * sizeof(double));
    int32_t* idx_buf = (int32_t*)malloc((size_t)train_rows * sizeof(int32_t));
    if (dist_buf == NULL || idx_buf == NULL) {
        free(dist_buf); free(idx_buf);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows_new; ++r) {
        const double* xi = X_new + (size_t)r * (size_t)cols;
        /* Compute all train distances; sort top-k. */
        for (int64_t j = 0; j < train_rows; ++j) {
            const double* xj = lof->train_X + (size_t)j * (size_t)cols;
            dist_buf[j] = sqrt(sq_eucl(xi, xj, cols));
            idx_buf[j] = (int32_t)j;
        }
        partial_sort_k_with_indices(dist_buf, idx_buf,
                                       (int32_t)train_rows, k);
        /* reach-distance(i, j) = max(k_distance(j), d(i, j)). */
        double sum = 0.0;
        for (int t = 0; t < k; ++t) {
            const int32_t j = idx_buf[t];
            const double d = dist_buf[t];
            const double reach = (lof->k_distance[j] > d)
                                  ? lof->k_distance[j] : d;
            sum += reach;
        }
        const double mean = sum / (double)k;
        const double lrd_i = (mean > 0.0) ? (1.0 / mean) : 0.0;
        /* LOF = mean lrd_j / lrd_i, j in N_k(i). */
        double lof_sum = 0.0;
        for (int t = 0; t < k; ++t) {
            const int32_t j = idx_buf[t];
            lof_sum += lof->lrd[j];
        }
        const double lof_mean = lof_sum / (double)k;
        scores[r] = (lrd_i > 0.0) ? (lof_mean / lrd_i) : HUGE_VAL;
    }
    free(dist_buf);
    free(idx_buf);
    return N4M_OK;
}
