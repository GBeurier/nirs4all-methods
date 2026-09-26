/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LocalMixupAugmenter — pick a neighbor for each sample and mix it in.
 *
 * For each sample i:
 *   1. Compute its k+1 nearest neighbors via squared-Euclidean distance.
 *   2. Draw a uniform integer in [0, k) to pick which neighbor (excluding
 *      self at index 0).
 *   3. Draw Beta(alpha, alpha) for lambda.
 *   4. out[i] = lam * X[i] + (1 - lam) * X[neighbor_idx], and the same
 *      mix of the targets when they are given.
 *
 * The reference uses sklearn NearestNeighbors with default parameters which
 * is brute force for small inputs. We mirror that by computing the full
 * pairwise squared-distance matrix once and using partial selection.
 */

#include "local_mixup.h"
#include "mixup.h"

#include <stdlib.h>

#include "core/augmentation/aug_rng_utils.h"
#include "core/augmentation/spectral/spectral_common.h"

struct n4m_aug_local_mixup_state_t {
    double  alpha;
    int32_t k_neighbors;
};

n4m_aug_local_mixup_state_t* n4m_aug_local_mixup_state_new(double alpha,
                                                            int32_t k_neighbors) {
    if (!(alpha > 0.0) || k_neighbors <= 0) {
        return NULL;
    }
    n4m_aug_local_mixup_state_t* s =
        (n4m_aug_local_mixup_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->alpha = alpha;
    s->k_neighbors = k_neighbors;
    return s;
}

void n4m_aug_local_mixup_state_free(n4m_aug_local_mixup_state_t* s) {
    free(s);
}

/* Compute squared-Euclidean distance between two row vectors. */
static double sqeuclidean(const double* a, const double* b, int64_t cols) {
    double s = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

/* Partial selection: find the k smallest indices of `dist[0..n)`
 * and put them at the start of `idx[]` in ascending distance order. */
static void partial_sort_k(double* dist, int64_t* idx, int64_t n, int64_t k) {
    /* Stable selection sort over the first k positions for determinism
     * (matching sklearn NearestNeighbors deterministic tie-breaking). */
    for (int64_t i = 0; i < k && i < n; ++i) {
        int64_t best = i;
        for (int64_t j = i + 1; j < n; ++j) {
            if (dist[j] < dist[best]) {
                best = j;
            }
        }
        if (best != i) {
            const double td = dist[i]; dist[i] = dist[best]; dist[best] = td;
            const int64_t ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        }
    }
}

n4m_status_t n4m_aug_local_mixup_apply_impl(
    const n4m_aug_local_mixup_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* Y, int64_t y_cols,
    double* out, double* out_y) {
    if (state == NULL || rng == NULL || X == NULL || out == NULL ||
        (Y != NULL && out_y == NULL)) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0 || (Y != NULL && y_cols < 0)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    const int32_t k = state->k_neighbors;
    if ((int64_t)k + 1 > rows) {
        /* Not enough rows to provide k distinct non-self neighbors. */
        return N4M_ERR_INVALID_ARGUMENT;
    }

    /* Workspace: dist[rows], idx[rows] for each query row; the drawn
     * neighbour and weight of every row. */
    double*  dist    = (double*)malloc((size_t)rows * sizeof(double));
    int64_t* idx     = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int64_t* partner = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    double*  lam     = (double*)malloc((size_t)rows * sizeof(double));
    if (dist == NULL || idx == NULL || partner == NULL || lam == NULL) {
        free(dist); free(idx); free(partner); free(lam);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    n4m_aug_randint_state_t randint_state;
    n4m_aug_randint_state_reset(&randint_state, rng);

    /* For each row, find the k+1 nearest neighbors (including self at idx 0). */
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < rows; ++j) {
            idx[j]  = j;
            dist[j] = sqeuclidean(X + (size_t)i * (size_t)cols,
                                  X + (size_t)j * (size_t)cols, cols);
        }
        partial_sort_k(dist, idx, rows, (int64_t)k + 1);

        /* Pick one neighbor at random from indices[1..k+1) — uniform
         * over the k non-self neighbors. NumPy's
         * `rng.choice(indices[i, 1:])` advances the bit stream by one
         * integers(0, k) call. */
        const int64_t pick = n4m_aug_randint(&randint_state, 0, k);
        partner[i] = idx[1 + pick];
        lam[i] = n4m_aug_rng_beta(rng, state->alpha, state->alpha);
    }

    n4m_status_t st = n4m_aug_mix_rows(X, rows, cols, partner, lam, out);
    if (st == N4M_OK && Y != NULL && y_cols > 0) {
        st = n4m_aug_mix_rows(Y, rows, y_cols, partner, lam, out_y);
    }
    free(dist);
    free(idx);
    free(partner);
    free(lam);
    return st;
}
