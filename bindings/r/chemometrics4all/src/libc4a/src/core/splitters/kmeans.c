/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * KMeans-based splitter — vendored k-means++ + Lloyd iterations.
 *
 * The C engine reproduces the algorithm bit-for-bit against a NumPy
 * reference that uses `numpy.random.default_rng(seed)` (i.e. PCG64) for
 * the random draws. The integer index outputs are therefore byte-equal
 * across the C and Python implementations for any fixed seed.
 *
 * Vendored k-means++ details:
 *   - First centroid: pick sample at index `rng.uniform(0, 1) * N` floor.
 *     We use `_uint64 >> 11` * 2^-53 for the uniform draw (matches
 *     NumPy's next_double convention) and floor to get the integer index.
 *   - Subsequent k-1 centroids: weighted-by-sq-dist sampling. We draw
 *     `u = rng.uniform()` and find the first cumulative weight >= u * total.
 *   - Lloyd: assign samples to closest centroid (argmin on Euclidean
 *     distance; first occurrence on ties); recompute centroids as column-
 *     wise mean per cluster; repeat until centroids unchanged or max_iter.
 *
 * Empty clusters (which can happen when n_train > n_distinct_points) get
 * a re-seed: we pick the sample that is currently farthest from any
 * existing centroid. This avoids the divide-by-zero in mean computation.
 */

#include "kmeans.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

c4a_split_kmeans_state_t* c4a_split_kmeans_state_new(double test_size,
                                                      uint64_t seed,
                                                      int32_t max_iter) {
    c4a_split_kmeans_state_t* s =
        (c4a_split_kmeans_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    s->seed      = seed;
    s->max_iter  = max_iter > 0 ? max_iter : 300;
    return s;
}

void c4a_split_kmeans_state_free(c4a_split_kmeans_state_t* state) {
    free(state);
}

/* Squared Euclidean distance from X[i, :] to centroid `c`. */
static double sq_dist(const double* X, int64_t i, int64_t cols, const double* c) {
    double s = 0.0;
    for (int64_t k = 0; k < cols; ++k) {
        const double d = X[i * cols + k] - c[k];
        s += d * d;
    }
    return s;
}

/* k-means++ initialisation. Writes k * cols doubles into `centroids`.
 * Updates `closest_sq[i]` = squared distance from sample i to its closest
 * already-chosen centroid. Returns C4A_OK or an error code. */
static c4a_status_t kmeans_plus_plus_init(const double* X, int64_t rows,
                                           int64_t cols, int64_t k,
                                           c4a_rng_pcg64* rng,
                                           double* centroids,
                                           double* closest_sq) {
    /* Centroid 0: uniform random sample. */
    const int64_t first = (int64_t)floor(c4a_split_rng_uniform01(rng) * (double)rows);
    const int64_t first_clamped = (first >= rows) ? rows - 1 : first;
    memcpy(centroids, X + first_clamped * cols, (size_t)cols * sizeof(double));
    for (int64_t i = 0; i < rows; ++i) {
        closest_sq[i] = sq_dist(X, i, cols, centroids);
    }

    /* Centroids 1..k-1: weighted by closest_sq. */
    for (int64_t c = 1; c < k; ++c) {
        double total = 0.0;
        for (int64_t i = 0; i < rows; ++i) total += closest_sq[i];
        const double u = c4a_split_rng_uniform01(rng) * total;
        double acc = 0.0;
        int64_t pick = rows - 1;
        for (int64_t i = 0; i < rows; ++i) {
            acc += closest_sq[i];
            if (acc >= u) {
                pick = i;
                break;
            }
        }
        memcpy(centroids + c * cols, X + pick * cols,
               (size_t)cols * sizeof(double));
        /* Update closest_sq with the new centroid. */
        for (int64_t i = 0; i < rows; ++i) {
            const double d = sq_dist(X, i, cols, centroids + c * cols);
            if (d < closest_sq[i]) closest_sq[i] = d;
        }
    }
    return C4A_OK;
}

/* Lloyd iteration: assign each sample to its closest centroid, recompute
 * centroids as column means. Returns 1 if assignments changed, 0 if stable.
 *
 * `new_centroids` (k * cols doubles) and `counts` (k int64_t) are scratch
 * buffers owned by the caller. They are zeroed at entry on every iteration
 * so the caller does not pay the malloc/free cost per Lloyd step. Mirrors
 * the AsLS L_buf / D_buf hoisting pattern. */
static int lloyd_step(const double* X, int64_t rows, int64_t cols,
                       int64_t k, double* centroids,
                       int64_t* assignments,
                       double*  new_centroids,
                       int64_t* counts) {
    int changed = 0;
    for (int64_t i = 0; i < rows; ++i) {
        int64_t best_c = 0;
        double  best_d = sq_dist(X, i, cols, centroids);
        for (int64_t c = 1; c < k; ++c) {
            const double d = sq_dist(X, i, cols, centroids + c * cols);
            if (d < best_d) { best_d = d; best_c = c; }
        }
        if (assignments[i] != best_c) {
            assignments[i] = best_c;
            changed = 1;
        }
    }
    /* Recompute centroids. Empty clusters: leave centroid in place. */
    memset(new_centroids, 0, (size_t)k * (size_t)cols * sizeof(double));
    memset(counts, 0, (size_t)k * sizeof(int64_t));
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t c = assignments[i];
        for (int64_t kk = 0; kk < cols; ++kk) {
            new_centroids[c * cols + kk] += X[i * cols + kk];
        }
        ++counts[c];
    }
    for (int64_t c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            const double inv = 1.0 / (double)counts[c];
            for (int64_t kk = 0; kk < cols; ++kk) {
                centroids[c * cols + kk] = new_centroids[c * cols + kk] * inv;
            }
        }
        /* Otherwise leave centroid unchanged. */
    }
    return changed;
}

c4a_status_t c4a_split_kmeans_apply(const c4a_split_kmeans_state_t* state,
                                     const double* X, int64_t rows, int64_t cols,
                                     c4a_split_result_t* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    int64_t n_train = 0, n_test = 0;
    c4a_status_t st = c4a_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != C4A_OK) return st;

    const int64_t k = n_train;
    /* Hoist all per-iteration scratch out of the Lloyd loop: centroids,
     * closest-sq, assignments, plus the `new_centroids` / `counts` accumulators
     * that lloyd_step zeroes-and-fills on every step. Saves 2*max_iter
     * malloc/free calls per row across the Phase 11 KMeans path. */
    double*  centroids     = (double*)malloc((size_t)k * (size_t)cols
                                              * sizeof(double));
    double*  closest_sq    = (double*)malloc((size_t)rows * sizeof(double));
    int64_t* assignments   = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    double*  new_centroids = (double*)malloc((size_t)k * (size_t)cols
                                              * sizeof(double));
    int64_t* counts        = (int64_t*)malloc((size_t)k * sizeof(int64_t));
    if (centroids == NULL || closest_sq == NULL || assignments == NULL ||
        new_centroids == NULL || counts == NULL) {
        free(centroids); free(closest_sq); free(assignments);
        free(new_centroids); free(counts);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) assignments[i] = -1;

    c4a_rng_pcg64 rng;
    c4a_pcg64_engine_seed(&rng, state->seed);

    st = kmeans_plus_plus_init(X, rows, cols, k, &rng, centroids, closest_sq);
    if (st != C4A_OK) {
        free(centroids); free(closest_sq); free(assignments);
        free(new_centroids); free(counts);
        return st;
    }
    for (int32_t it = 0; it < state->max_iter; ++it) {
        if (!lloyd_step(X, rows, cols, k, centroids, assignments,
                        new_centroids, counts)) break;
    }

    /* For each centroid, pick the closest sample (argmin, first on ties).
     * Then deduplicate (np.unique returns sorted). */
    int64_t* picked = (int64_t*)malloc((size_t)k * sizeof(int64_t));
    if (picked == NULL) {
        free(centroids); free(closest_sq); free(assignments);
        free(new_centroids); free(counts);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t c = 0; c < k; ++c) {
        int64_t best_i = 0;
        double  best_d = sq_dist(X, 0, cols, centroids + c * cols);
        for (int64_t i = 1; i < rows; ++i) {
            const double d = sq_dist(X, i, cols, centroids + c * cols);
            if (d < best_d) { best_d = d; best_i = i; }
        }
        picked[c] = best_i;
    }
    /* Deduplicate via flag bitmap. */
    char* in_train = (char*)calloc((size_t)rows, sizeof(char));
    if (in_train == NULL) {
        free(centroids); free(closest_sq); free(assignments);
        free(new_centroids); free(counts); free(picked);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t c = 0; c < k; ++c) in_train[picked[c]] = 1;
    int64_t actual_n_train = 0;
    for (int64_t i = 0; i < rows; ++i) if (in_train[i]) ++actual_n_train;
    /* nirs4all caps train at n_train (which is k, but after dedup
     * actual_n_train may be < k). We mirror the behaviour: train = the
     * sorted unique set as-is (may be smaller than k); test = complement. */
    int64_t actual_n_test = rows - actual_n_train;

    st = c4a_split_result_alloc(out, actual_n_train, actual_n_test);
    if (st != C4A_OK) {
        free(centroids); free(closest_sq); free(assignments);
        free(new_centroids); free(counts); free(picked); free(in_train);
        return st;
    }
    int64_t ti = 0, te = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (in_train[i]) out->train_idx[ti++] = i;
        else             out->test_idx[te++]  = i;
    }

    free(centroids); free(closest_sq); free(assignments);
    free(new_centroids); free(counts); free(picked); free(in_train);
    return C4A_OK;
}
