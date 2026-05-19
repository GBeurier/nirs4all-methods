/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SPlit splitter — data twinning (Vakayil & Joseph 2022).
 *
 * Bit-exact equivalent of nirs4all.operators.splitters._twin with the
 * starting index drawn from PCG64 (instead of np.random.randint) so the
 * C engine and the Python reference fixture (which mirrors this exact
 * PCG64 draw) reproduce the same twin indices.
 */

#include "split_splitter.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

c4a_split_splt_state_t* c4a_split_splt_state_new(double test_size,
                                                  uint64_t seed) {
    c4a_split_splt_state_t* s =
        (c4a_split_splt_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    s->seed      = seed;
    return s;
}

void c4a_split_splt_state_free(c4a_split_splt_state_t* state) {
    free(state);
}

c4a_status_t c4a_split_splt_apply(const c4a_split_splt_state_t* state,
                                   const double* X, int64_t rows, int64_t cols,
                                   c4a_split_result_t* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 4 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (!(state->test_size > 0.0) || !(state->test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int64_t r = (int64_t)(1.0 / state->test_size);
    if (r < 2 || r > rows / 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Step 1: preprocess X — drop constant columns, z-score (population). */
    char* keep_col = (char*)malloc((size_t)cols * sizeof(char));
    if (keep_col == NULL) return C4A_ERR_OUT_OF_MEMORY;
    int64_t kept = 0;
    for (int64_t c = 0; c < cols; ++c) {
        const double v0 = X[c];
        int constant = 1;
        for (int64_t i = 1; i < rows; ++i) {
            if (X[i * cols + c] != v0) { constant = 0; break; }
        }
        keep_col[c] = (char)(constant ? 0 : 1);
        if (!constant) ++kept;
    }
    if (kept == 0) {
        /* All constant — twin = first ceil(N/r) indices in order (since all
         * distances are zero, ties break to first occurrence everywhere). */
        free(keep_col);
        const int64_t n_test = (rows + r - 1) / r;
        const int64_t n_train = rows - n_test;
        c4a_status_t st = c4a_split_result_alloc(out, n_train, n_test);
        if (st != C4A_OK) return st;
        for (int64_t i = 0; i < n_test; ++i) out->test_idx[i] = i;
        for (int64_t i = 0; i < n_train; ++i) out->train_idx[i] = n_test + i;
        return C4A_OK;
    }

    double* Xz = (double*)malloc((size_t)rows * (size_t)kept * sizeof(double));
    if (Xz == NULL) { free(keep_col); return C4A_ERR_OUT_OF_MEMORY; }
    {
        int64_t k = 0;
        for (int64_t c = 0; c < cols; ++c) {
            if (!keep_col[c]) continue;
            double mean = 0.0;
            for (int64_t i = 0; i < rows; ++i) mean += X[i * cols + c];
            mean /= (double)rows;
            double var = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                const double d = X[i * cols + c] - mean;
                var += d * d;
            }
            const double sd = sqrt(var / (double)rows);
            const double inv = (sd > 0.0) ? 1.0 / sd : 1.0;
            for (int64_t i = 0; i < rows; ++i) {
                Xz[i * kept + k] = (X[i * cols + c] - mean) * inv;
            }
            ++k;
        }
    }
    free(keep_col);

    /* Step 2: pick starting index u1 = floor(uniform01 * N). */
    c4a_rng_pcg64 rng;
    c4a_pcg64_engine_seed(&rng, state->seed);
    const double u_start = c4a_split_rng_uniform01(&rng);
    int64_t current = (int64_t)floor(u_start * (double)rows);
    if (current < 0) current = 0;
    if (current >= rows) current = rows - 1;

    const int64_t n_twin = (rows + r - 1) / r;
    int64_t* twin_idx = (int64_t*)malloc((size_t)n_twin * sizeof(int64_t));
    char*    active   = (char*)malloc((size_t)rows * sizeof(char));
    double*  sq_dists = (double*)malloc((size_t)rows * sizeof(double));
    int64_t* active_list = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (twin_idx == NULL || active == NULL || sq_dists == NULL ||
        active_list == NULL) {
        free(Xz); free(twin_idx); free(active); free(sq_dists); free(active_list);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    memset(active, 1, (size_t)rows);
    int64_t n_active = rows;
    int64_t count = 0;

    while (n_active >= r) {
        /* Collect active_list. */
        int64_t alen = 0;
        for (int64_t i = 0; i < rows; ++i) if (active[i]) active_list[alen++] = i;

        /* sq_dists[a] = ||Xz[current] - Xz[active_list[a]]||^2. */
        for (int64_t a = 0; a < alen; ++a) {
            const int64_t j = active_list[a];
            double s = 0.0;
            for (int64_t c = 0; c < kept; ++c) {
                const double d = Xz[j * kept + c] - Xz[current * kept + c];
                s += d * d;
            }
            sq_dists[a] = s;
        }

        /* Find the r nearest active samples, sorted by distance ascending.
         * For ties, NumPy's argpartition followed by argsort within the top
         * group resolves by underlying index (since stable sort breaks
         * ties by position). Sort the first r entries via insertion sort
         * on (sq_dist, active_index). */
        /* Partial selection: find r smallest sq_dists. We do an O(r * alen)
         * scan via a small heap kept as a sorted array of size r. */
        int64_t* nearest = (int64_t*)malloc((size_t)r * sizeof(int64_t));
        double*  near_d  = (double*)malloc((size_t)r * sizeof(double));
        if (nearest == NULL || near_d == NULL) {
            free(nearest); free(near_d);
            free(Xz); free(twin_idx); free(active); free(sq_dists); free(active_list);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        for (int64_t k = 0; k < r; ++k) {
            nearest[k] = active_list[k];
            near_d[k]  = sq_dists[k];
        }
        /* Sort initial r ascending by (near_d, nearest). */
        for (int64_t i = 1; i < r; ++i) {
            int64_t ni = nearest[i];
            double  di = near_d[i];
            int64_t k = i - 1;
            while (k >= 0 && (near_d[k] > di ||
                              (near_d[k] == di && nearest[k] > ni))) {
                nearest[k + 1] = nearest[k];
                near_d[k + 1]  = near_d[k];
                --k;
            }
            nearest[k + 1] = ni;
            near_d[k + 1]  = di;
        }
        double worst = near_d[r - 1];
        int64_t worst_idx = nearest[r - 1];
        for (int64_t a = r; a < alen; ++a) {
            const double  d = sq_dists[a];
            const int64_t j = active_list[a];
            if (d < worst || (d == worst && j < worst_idx)) {
                /* Replace worst, then sift up. */
                nearest[r - 1] = j;
                near_d[r - 1]  = d;
                int64_t k = r - 2;
                while (k >= 0 && (near_d[k] > near_d[k + 1] ||
                                  (near_d[k] == near_d[k + 1] &&
                                   nearest[k] > nearest[k + 1]))) {
                    int64_t tn = nearest[k];
                    double  td = near_d[k];
                    nearest[k] = nearest[k + 1];
                    near_d[k]  = near_d[k + 1];
                    nearest[k + 1] = tn;
                    near_d[k + 1]  = td;
                    --k;
                }
                worst = near_d[r - 1];
                worst_idx = nearest[r - 1];
            }
        }

        /* Closest neighbour goes into the twin. */
        twin_idx[count++] = nearest[0];

        /* Deactivate all r nearest. */
        for (int64_t k = 0; k < r; ++k) active[nearest[k]] = 0;
        n_active -= r;
        const int64_t farthest = nearest[r - 1];
        free(nearest); free(near_d);

        /* Pick next current = nearest active to `farthest`. */
        if (n_active > 0) {
            int64_t best_j  = -1;
            double  best_d  = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                if (!active[i]) continue;
                double s = 0.0;
                for (int64_t c = 0; c < kept; ++c) {
                    const double d = Xz[i * kept + c] - Xz[farthest * kept + c];
                    s += d * d;
                }
                if (best_j == -1 || s < best_d ||
                    (s == best_d && i < best_j)) {
                    best_j = i;
                    best_d = s;
                }
            }
            current = best_j;
        }
    }
    /* Handle remainder: when N not divisible by r, append current. */
    if (count < n_twin) {
        twin_idx[count++] = current;
    }

    /* Build train (complement, sorted ascending) and test (twin in pick order). */
    char* in_test = (char*)calloc((size_t)rows, sizeof(char));
    if (in_test == NULL) {
        free(Xz); free(twin_idx); free(active); free(sq_dists); free(active_list);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < count; ++i) in_test[twin_idx[i]] = 1;
    const int64_t n_test_actual  = count;
    const int64_t n_train_actual = rows - count;
    c4a_status_t st = c4a_split_result_alloc(out, n_train_actual, n_test_actual);
    if (st != C4A_OK) {
        free(Xz); free(twin_idx); free(active); free(sq_dists);
        free(active_list); free(in_test);
        return st;
    }
    /* Test in pick order (matches Python's _twin returning twin_idx). */
    for (int64_t i = 0; i < count; ++i) out->test_idx[i] = twin_idx[i];
    /* Train as complement, ascending (matches np.delete(np.arange(N), idx)). */
    int64_t ti = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (!in_test[i]) out->train_idx[ti++] = i;
    }

    free(Xz); free(twin_idx); free(active); free(sq_dists);
    free(active_list); free(in_test);
    return C4A_OK;
}
