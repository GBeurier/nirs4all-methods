/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SPXY K-Fold splitter — alternating max-min fold assignment.
 *
 * Implements the algorithm in nirs4all.operators.splitters.SPXYFold._assign_to_folds:
 *
 *   1. centroid_distances = D.mean(axis=1)           (length n_samples)
 *   2. init_indices = argsort(centroid_distances)[-n_splits:]
 *      (ascending order, so init_indices[k] is the (n_samples - n_splits + k)-th
 *      smallest centroid distance — i.e. the n_splits samples farthest
 *      from the dataset centroid).
 *   3. fold_assignment[init_indices[k]] = k for k in [0, n_splits).
 *   4. Loop: while there are unassigned samples, cycle through folds
 *      0..n_splits-1; for each fold pick the unassigned sample maximising
 *      its minimum distance to the current fold's members. Skip folds that
 *      have reached max_size = ceil(n / n_splits).
 *
 * Tie-breaking: NumPy's argmax / argsort use stable first-occurrence
 * ordering. We follow the same convention by iterating sample indices
 * ascending and using strict > / < comparisons.
 */

#include "spxy_fold.h"

#include <stdlib.h>
#include <string.h>

n4m_split_spxy_fold_state_t* n4m_split_spxy_fold_state_new(int32_t n_splits,
                                                            int32_t use_y) {
    if (n_splits < 2) return NULL;
    n4m_split_spxy_fold_state_t* s =
        (n4m_split_spxy_fold_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->n_splits = n_splits;
    s->use_y    = use_y;
    return s;
}

void n4m_split_spxy_fold_state_free(n4m_split_spxy_fold_state_t* state) {
    free(state);
}

/* Helper: argsort indices ascending by `values[i]`. Stable: ties resolve by
 * sample index (low first), matching NumPy's argsort with kind='stable'.
 * We use a simple insertion-style selection over an O(n^2) inner loop, which
 * is fine for the n typical in NIRS splits (<= a few thousands). For larger
 * n the loop dominates; we accept the overhead given the calibration size
 * regime. */
static void argsort_ascending(const double* values, int64_t n, int64_t* out_idx) {
    /* Initialise out_idx = [0, 1, ..., n-1]. */
    for (int64_t i = 0; i < n; ++i) out_idx[i] = i;
    /* Insertion sort by `values[out_idx[i]]`. Stable. */
    for (int64_t i = 1; i < n; ++i) {
        int64_t key = out_idx[i];
        double  vk  = values[key];
        int64_t j = i - 1;
        while (j >= 0 && values[out_idx[j]] > vk) {
            out_idx[j + 1] = out_idx[j];
            --j;
        }
        out_idx[j + 1] = key;
    }
}

/* Compute D (X-only or X+Y normalised sum) and store its mean over rows in
 * `centroid_distances`. The caller owns both buffers. */
static n4m_status_t compute_distance_and_centroid(
    const n4m_split_spxy_fold_state_t* s,
    const double* X, int64_t rows, int64_t cols_x,
    const double* Y, int64_t cols_y,
    double* D, double* centroid_distances) {

    n4m_status_t st = n4m_split_euclidean_dist(X, rows, cols_x, D);
    if (st != N4M_OK) return st;
    n4m_split_normalise_distance_matrix(D, rows);

    if (s->use_y == 1) {
        if (Y == NULL) return N4M_ERR_NULL_POINTER;
        double* DY = (double*)malloc((size_t)rows * (size_t)rows * sizeof(double));
        if (DY == NULL) return N4M_ERR_OUT_OF_MEMORY;
        st = n4m_split_euclidean_dist(Y, rows, cols_y, DY);
        if (st != N4M_OK) { free(DY); return st; }
        n4m_split_normalise_distance_matrix(DY, rows);
        const size_t mat = (size_t)rows * (size_t)rows;
        for (size_t i = 0; i < mat; ++i) D[i] += DY[i];
        free(DY);
    } else if (s->use_y == 2) {
        /* Hamming-on-Y: D_Y[i, j] = any(y[i, k] != y[j, k]) over k.
         * Already in [0, 1], no normalisation needed. */
        if (Y == NULL) return N4M_ERR_NULL_POINTER;
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = i + 1; j < rows; ++j) {
                int diff = 0;
                for (int64_t k = 0; k < cols_y; ++k) {
                    if (Y[i * cols_y + k] != Y[j * cols_y + k]) {
                        diff = 1;
                        break;
                    }
                }
                const double d = (double)diff;
                D[i * rows + j] += d;
                D[j * rows + i] += d;
            }
        }
    }

    for (int64_t i = 0; i < rows; ++i) {
        double s_row = 0.0;
        for (int64_t j = 0; j < rows; ++j) s_row += D[i * rows + j];
        centroid_distances[i] = s_row / (double)rows;
    }
    return N4M_OK;
}

n4m_status_t n4m_split_spxy_fold_assign(const n4m_split_spxy_fold_state_t* s,
                                         const double* X, int64_t rows,
                                         int64_t cols_x,
                                         const double* Y, int64_t cols_y,
                                         int32_t* fold_assignment) {
    if (s == NULL || X == NULL || fold_assignment == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (s->n_splits < 2 || rows < 2 || cols_x < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (s->n_splits > rows) {
        /* Match nirs4all: refuse rather than wrap. */
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int32_t K = s->n_splits;

    double* D = (double*)malloc((size_t)rows * (size_t)rows * sizeof(double));
    double* centroid = (double*)malloc((size_t)rows * sizeof(double));
    if (D == NULL || centroid == NULL) {
        free(D); free(centroid);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_status_t st = compute_distance_and_centroid(s, X, rows, cols_x,
                                                     Y, cols_y, D, centroid);
    if (st != N4M_OK) {
        free(D); free(centroid);
        return st;
    }

    int64_t* sorted_idx = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    char*    assigned   = (char*)calloc((size_t)rows, sizeof(char));
    int32_t* fold_sizes = (int32_t*)calloc((size_t)K, sizeof(int32_t));
    /* fold_members stores the sample indices currently in each fold.
     * Each fold can hold at most `max_size` members. */
    const int32_t target_size = (int32_t)(rows / K);
    const int32_t max_size    = target_size + ((rows % K) != 0 ? 1 : 0);
    int64_t* fold_members = (int64_t*)malloc((size_t)K * (size_t)max_size
                                             * sizeof(int64_t));
    /* For each remaining (unassigned) sample r, track min distance to
     * fold f's members. We refresh this lazily — when a new sample is
     * added to fold f, we update min_dist[f][r] for all unassigned r. */
    double* min_dist_to_fold = (double*)malloc((size_t)rows * (size_t)K
                                               * sizeof(double));

    if (sorted_idx == NULL || assigned == NULL || fold_sizes == NULL ||
        fold_members == NULL || min_dist_to_fold == NULL) {
        free(D); free(centroid); free(sorted_idx); free(assigned);
        free(fold_sizes); free(fold_members); free(min_dist_to_fold);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    /* Initialise fold_assignment to -1. */
    for (int64_t i = 0; i < rows; ++i) fold_assignment[i] = -1;

    /* Step 1: init folds with the K samples farthest from the centroid. */
    argsort_ascending(centroid, rows, sorted_idx);
    for (int32_t k = 0; k < K; ++k) {
        const int64_t sample_idx = sorted_idx[rows - K + k];
        fold_assignment[sample_idx] = k;
        fold_members[(size_t)k * (size_t)max_size + 0] = sample_idx;
        fold_sizes[k] = 1;
        assigned[sample_idx] = 1;
    }

    /* Initialise min_dist_to_fold for the unassigned samples. */
    for (int64_t r = 0; r < rows; ++r) {
        if (assigned[r]) continue;
        for (int32_t k = 0; k < K; ++k) {
            const int64_t m0 = fold_members[(size_t)k * (size_t)max_size + 0];
            min_dist_to_fold[(size_t)k * (size_t)rows + (size_t)r] = D[r * rows + m0];
        }
    }

    /* Step 2: alternating assignment, cycle through folds. */
    int64_t remaining = rows - K;
    while (remaining > 0) {
        for (int32_t k = 0; k < K && remaining > 0; ++k) {
            if (fold_sizes[k] >= max_size) continue;
            /* Find r* = argmax over unassigned r of min_dist_to_fold[k][r]. */
            int64_t best_r = -1;
            double  best_d = -1.0;
            const double* mk = &min_dist_to_fold[(size_t)k * (size_t)rows];
            for (int64_t r = 0; r < rows; ++r) {
                if (assigned[r]) continue;
                if (best_r == -1 || mk[r] > best_d) {
                    best_r = r;
                    best_d = mk[r];
                }
            }
            if (best_r < 0) break;
            fold_assignment[best_r] = k;
            fold_members[(size_t)k * (size_t)max_size + (size_t)fold_sizes[k]] = best_r;
            fold_sizes[k]++;
            assigned[best_r] = 1;
            /* Update min_dist_to_fold[k][r] for the newly added member. */
            for (int64_t r = 0; r < rows; ++r) {
                if (assigned[r]) continue;
                const double nd = D[best_r * rows + r];
                if (nd < min_dist_to_fold[(size_t)k * (size_t)rows + (size_t)r]) {
                    min_dist_to_fold[(size_t)k * (size_t)rows + (size_t)r] = nd;
                }
            }
            --remaining;
        }
        /* Safety: if all folds are full but remaining > 0 (shouldn't happen
         * because max_size * K >= rows), break to avoid infinite loop. */
        int any_room = 0;
        for (int32_t k = 0; k < K; ++k) {
            if (fold_sizes[k] < max_size) { any_room = 1; break; }
        }
        if (!any_room) break;
    }

    free(D); free(centroid); free(sorted_idx); free(assigned);
    free(fold_sizes); free(fold_members); free(min_dist_to_fold);
    return N4M_OK;
}

n4m_status_t n4m_split_spxy_fold_extract(const int32_t* fold_assignment,
                                          int64_t rows, int32_t fold_idx,
                                          n4m_split_result_t* out) {
    if (fold_assignment == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || fold_idx < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    int64_t n_test = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (fold_assignment[i] == fold_idx) ++n_test;
    }
    int64_t n_train = rows - n_test;
    if (n_train < 1 || n_test < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_status_t st = n4m_split_result_alloc(out, n_train, n_test);
    if (st != N4M_OK) return st;
    int64_t ti = 0, te = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (fold_assignment[i] == fold_idx) {
            out->test_idx[te++] = i;
        } else {
            out->train_idx[ti++] = i;
        }
    }
    return N4M_OK;
}
