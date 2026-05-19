/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SPXY splitter — joint X-Y max-min distance selection.
 *
 * Matches nirs4all.operators.splitters.SPXYSplitter with the default
 * metric (Euclidean) on both X and Y, no PCA reduction.
 */

#include "spxy.h"

#include <stdlib.h>

c4a_split_spxy_state_t* c4a_split_spxy_state_new(double test_size) {
    c4a_split_spxy_state_t* s = (c4a_split_spxy_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    return s;
}

void c4a_split_spxy_state_free(c4a_split_spxy_state_t* state) {
    free(state);
}

c4a_status_t c4a_split_spxy_apply(const c4a_split_spxy_state_t* state,
                                   const double* X, int64_t rows,
                                   int64_t cols_x,
                                   const double* Y, int64_t cols_y,
                                   c4a_split_result_t* out) {
    if (state == NULL || X == NULL || Y == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols_x < 1 || cols_y < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    int64_t n_train = 0, n_test = 0;
    c4a_status_t st = c4a_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != C4A_OK) return st;
    if (n_train < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    const size_t mat = (size_t)rows * (size_t)rows;
    double* DX = (double*)malloc(mat * sizeof(double));
    double* DY = (double*)malloc(mat * sizeof(double));
    if (DX == NULL || DY == NULL) {
        free(DX); free(DY);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    st = c4a_split_euclidean_dist(X, rows, cols_x, DX);
    if (st != C4A_OK) { free(DX); free(DY); return st; }
    st = c4a_split_euclidean_dist(Y, rows, cols_y, DY);
    if (st != C4A_OK) { free(DX); free(DY); return st; }
    c4a_split_normalise_distance_matrix(DX, rows);
    c4a_split_normalise_distance_matrix(DY, rows);
    for (size_t i = 0; i < mat; ++i) {
        DX[i] += DY[i];
    }
    free(DY);

    st = c4a_split_result_alloc(out, n_train, n_test);
    if (st != C4A_OK) {
        free(DX);
        return st;
    }
    st = c4a_split_max_min_selection(DX, rows, n_train,
                                      out->train_idx, out->test_idx);
    free(DX);
    if (st != C4A_OK) {
        c4a_split_result_free(out);
        return st;
    }
    return C4A_OK;
}
