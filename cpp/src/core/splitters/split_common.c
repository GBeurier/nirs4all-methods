/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared engine helpers for the Phase 11 sample-partitioning splitters.
 *
 * The max-min selection mirrors the reference implementation in
 * nirs4all.operators.splitters._max_min_distance_split: the two farthest
 * points are picked first via argmax over the flattened distance matrix,
 * then the next sample is chosen to maximise the minimum distance to the
 * already-selected set. NumPy's argmax breaks ties at the first occurrence
 * — we do the same here so the integer index outputs are byte-identical to
 * the NumPy reference.
 */

#include "split_common.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Result allocation.
 *
 * We pack the train and test index arrays into one single allocation so the
 * public destroy function only has to call free() once.
 * ------------------------------------------------------------------------ */
c4a_status_t c4a_split_result_alloc(c4a_split_result_t* out,
                                    int64_t n_train, int64_t n_test) {
    if (out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n_train < 0 || n_test < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    out->train_idx = NULL;
    out->test_idx  = NULL;
    out->n_train   = 0;
    out->n_test    = 0;
    out->_owner    = NULL;

    const size_t total = (size_t)(n_train + n_test);
    if (total == 0) {
        return C4A_OK;
    }
    int64_t* buf = (int64_t*)malloc(total * sizeof(int64_t));
    if (buf == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    out->_owner    = buf;
    out->train_idx = (n_train > 0) ? buf : NULL;
    out->test_idx  = (n_test > 0) ? (buf + n_train) : NULL;
    out->n_train   = n_train;
    out->n_test    = n_test;
    return C4A_OK;
}

void c4a_split_result_clear(c4a_split_result_t* r) {
    if (r == NULL) return;
    r->train_idx = NULL;
    r->test_idx  = NULL;
    r->n_train   = 0;
    r->n_test    = 0;
    r->_owner    = NULL;
}

void c4a_split_result_free(c4a_split_result_t* r) {
    if (r == NULL) return;
    free(r->_owner);
    c4a_split_result_clear(r);
}

/* --------------------------------------------------------------------------
 * test_size → (n_train, n_test).
 *
 * sklearn's `_validate_shuffle_split` rounds n_test = ceil(test_size * N).
 * We replicate the float branch (we never accept an integer test_size in
 * the C ABI for simplicity).
 * ------------------------------------------------------------------------ */
c4a_status_t c4a_split_compute_train_test_count(int64_t n_samples,
                                                double test_size,
                                                int64_t* out_n_train,
                                                int64_t* out_n_test) {
    if (out_n_train == NULL || out_n_test == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    *out_n_train = 0;
    *out_n_test  = 0;
    if (n_samples < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (!(test_size > 0.0) || !(test_size < 1.0) || !isfinite(test_size)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const double prod = test_size * (double)n_samples;
    int64_t n_test = (int64_t)ceil(prod);
    if (n_test < 1) n_test = 1;
    if (n_test >= n_samples) n_test = n_samples - 1;
    int64_t n_train = n_samples - n_test;
    if (n_train < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    *out_n_train = n_train;
    *out_n_test  = n_test;
    return C4A_OK;
}

/* --------------------------------------------------------------------------
 * Pairwise Euclidean distance matrix.
 *
 * We compute D[i, j] = sqrt(sum_k (X[i, k] - X[j, k])^2) for all (i, j)
 * pairs. Symmetric, with D[i, i] = 0. Loop ordering is upper-triangular
 * to halve the work; the diagonal is explicitly zeroed.
 * ------------------------------------------------------------------------ */
c4a_status_t c4a_split_euclidean_dist(const double* X, int64_t rows,
                                       int64_t cols, double* D) {
    if (X == NULL || D == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; i < rows; ++i) {
        D[i * rows + i] = 0.0;
        for (int64_t j = i + 1; j < rows; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < cols; ++k) {
                const double d = X[i * cols + k] - X[j * cols + k];
                s += d * d;
            }
            const double dist = sqrt(s);
            D[i * rows + j] = dist;
            D[j * rows + i] = dist;
        }
    }
    return C4A_OK;
}

c4a_status_t c4a_split_euclidean_dist_to_point(const double* X,
                                                int64_t rows, int64_t cols,
                                                const double* r, double* out) {
    if (X == NULL || r == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int64_t k = 0; k < cols; ++k) {
            const double d = X[i * cols + k] - r[k];
            s += d * d;
        }
        out[i] = sqrt(s);
    }
    return C4A_OK;
}

void c4a_split_normalise_distance_matrix(double* D, int64_t n) {
    if (D == NULL || n <= 0) return;
    double dmax = 0.0;
    const size_t total = (size_t)n * (size_t)n;
    for (size_t i = 0; i < total; ++i) {
        if (D[i] > dmax) dmax = D[i];
    }
    if (dmax <= 0.0) return;
    const double inv = 1.0 / dmax;
    for (size_t i = 0; i < total; ++i) {
        D[i] *= inv;
    }
}

double c4a_split_rng_uniform01(c4a_rng_pcg64* rng) {
    /* NumPy convention: (u64 >> 11) * 2^-53. The literal 9007199254740992.0
     * == 2^53. */
    const uint64_t u = c4a_pcg64_engine_next_uint64(rng);
    return (double)(u >> 11) * (1.0 / 9007199254740992.0);
}

/* --------------------------------------------------------------------------
 * Max-min subset selection.
 *
 * Replicates the algorithm in nirs4all.operators.splitters.KennardStoneSplitter
 * (also reused by SPXYSplitter). Internally:
 *
 *   1. Find (i*, j*) = argmax of D (NumPy unravel_index order: row-major
 *      flat index of the first maximum). Add both to the train set.
 *   2. For step k = 2..train_size - 1:
 *      min_distances[j] = min over picked p of D[p, j] for each j not yet
 *      in the train set; pick the j with the largest min_distance.
 *
 * Tie-breaking: NumPy's argmax / argmin returns the first occurrence on
 * ties. We follow the same convention (`>` and `<` strictly, never `>=`).
 *
 * We keep an `in_train[i]` boolean array and a running `min_dist_to_train[i]`
 * vector so each iteration is O(n) rather than O(train_size * n).
 * ------------------------------------------------------------------------ */
c4a_status_t c4a_split_max_min_selection(const double* D, int64_t n,
                                          int64_t train_size,
                                          int64_t* train_idx,
                                          int64_t* test_idx) {
    if (D == NULL || train_idx == NULL || test_idx == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 2 || train_size < 2 || train_size > n) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Step 1: pick the two globally farthest points (first occurrence of
     * the max in flat row-major order, matching np.unravel_index of
     * np.argmax(D.ravel())). */
    int64_t flat_arg = 0;
    {
        double best = D[0];
        const size_t total = (size_t)n * (size_t)n;
        for (size_t k = 1; k < total; ++k) {
            if (D[k] > best) {
                best = D[k];
                flat_arg = (int64_t)k;
            }
        }
    }
    const int64_t first  = flat_arg / n;
    const int64_t second = flat_arg % n;
    if (first == second) {
        /* All zeros — fall back to (0, 1) so the algorithm can still run. */
        train_idx[0] = 0;
        train_idx[1] = 1;
    } else {
        train_idx[0] = first;
        train_idx[1] = second;
    }

    char*    in_train          = (char*)calloc((size_t)n, sizeof(char));
    double*  min_dist_to_train = (double*)malloc((size_t)n * sizeof(double));
    if (in_train == NULL || min_dist_to_train == NULL) {
        free(in_train);
        free(min_dist_to_train);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    in_train[train_idx[0]] = 1;
    in_train[train_idx[1]] = 1;
    /* Initialise min_dist_to_train[j] = min(D[first, j], D[second, j]). */
    for (int64_t j = 0; j < n; ++j) {
        double d1 = D[train_idx[0] * n + j];
        double d2 = D[train_idx[1] * n + j];
        min_dist_to_train[j] = (d1 < d2) ? d1 : d2;
    }

    /* Step 2: greedy max-min selection. */
    for (int64_t k = 2; k < train_size; ++k) {
        int64_t best_j  = -1;
        double  best_d  = -1.0;
        for (int64_t j = 0; j < n; ++j) {
            if (in_train[j]) continue;
            if (best_j == -1 || min_dist_to_train[j] > best_d) {
                best_j = j;
                best_d = min_dist_to_train[j];
            }
        }
        if (best_j < 0) {
            free(in_train);
            free(min_dist_to_train);
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        train_idx[k] = best_j;
        in_train[best_j] = 1;
        /* Update running min distances against the newly added point. */
        for (int64_t j = 0; j < n; ++j) {
            if (in_train[j]) continue;
            const double nd = D[best_j * n + j];
            if (nd < min_dist_to_train[j]) min_dist_to_train[j] = nd;
        }
    }

    /* Step 3: fill the test set with everything not in train, in
     * ascending order (matches the NumPy reference where index_test starts
     * as np.arange(N) and elements are removed by position). */
    int64_t t = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (!in_train[j]) {
            test_idx[t++] = j;
        }
    }

    free(in_train);
    free(min_dist_to_train);
    return C4A_OK;
}
