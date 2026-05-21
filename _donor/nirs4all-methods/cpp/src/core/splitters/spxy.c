/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SPXY splitter — joint X-Y max-min distance selection.
 *
 * Matches nirs4all.operators.splitters.SPXYSplitter with the default
 * metric (Euclidean) on both X and Y, no PCA reduction.
 */

#include "spxy.h"

#include <stdlib.h>

n4m_split_spxy_state_t* n4m_split_spxy_state_new(double test_size) {
    n4m_split_spxy_state_t* s = (n4m_split_spxy_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    return s;
}

void n4m_split_spxy_state_free(n4m_split_spxy_state_t* state) {
    free(state);
}

n4m_status_t n4m_split_spxy_apply(const n4m_split_spxy_state_t* state,
                                   const double* X, int64_t rows,
                                   int64_t cols_x,
                                   const double* Y, int64_t cols_y,
                                   n4m_split_result_t* out) {
    if (state == NULL || X == NULL || Y == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols_x < 1 || cols_y < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    int64_t n_train = 0, n_test = 0;
    n4m_status_t st = n4m_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != N4M_OK) return st;
    if (n_train < 2) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const size_t mat = (size_t)rows * (size_t)rows;
    double* DX = (double*)malloc(mat * sizeof(double));
    double* DY = (double*)malloc(mat * sizeof(double));
    if (DX == NULL || DY == NULL) {
        free(DX); free(DY);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    st = n4m_split_euclidean_dist(X, rows, cols_x, DX);
    if (st != N4M_OK) { free(DX); free(DY); return st; }
    st = n4m_split_euclidean_dist(Y, rows, cols_y, DY);
    if (st != N4M_OK) { free(DX); free(DY); return st; }
    n4m_split_normalise_distance_matrix(DX, rows);
    n4m_split_normalise_distance_matrix(DY, rows);
    for (size_t i = 0; i < mat; ++i) {
        DX[i] += DY[i];
    }
    free(DY);

    st = n4m_split_result_alloc(out, n_train, n_test);
    if (st != N4M_OK) {
        free(DX);
        return st;
    }
    st = n4m_split_max_min_selection(DX, rows, n_train,
                                      out->train_idx, out->test_idx);
    free(DX);
    if (st != N4M_OK) {
        n4m_split_result_free(out);
        return st;
    }
    return N4M_OK;
}
