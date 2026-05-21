/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Kennard-Stone sample-partitioning splitter — max-min Euclidean distance.
 *
 * Matches nirs4all.operators.splitters.KennardStoneSplitter with the
 * default metric (Euclidean) and no PCA reduction (raw X distances).
 */

#include "kennard_stone.h"

#include <stdlib.h>

n4m_split_ks_state_t* n4m_split_ks_state_new(double test_size) {
    n4m_split_ks_state_t* s = (n4m_split_ks_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    return s;
}

void n4m_split_ks_state_free(n4m_split_ks_state_t* state) {
    free(state);
}

n4m_status_t n4m_split_ks_apply(const n4m_split_ks_state_t* state,
                                const double* X, int64_t rows, int64_t cols,
                                n4m_split_result_t* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    int64_t n_train = 0, n_test = 0;
    n4m_status_t st = n4m_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != N4M_OK) return st;
    if (n_train < 2) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double* D = (double*)malloc((size_t)rows * (size_t)rows * sizeof(double));
    if (D == NULL) return N4M_ERR_OUT_OF_MEMORY;
    st = n4m_split_euclidean_dist(X, rows, cols, D);
    if (st != N4M_OK) {
        free(D);
        return st;
    }

    st = n4m_split_result_alloc(out, n_train, n_test);
    if (st != N4M_OK) {
        free(D);
        return st;
    }
    st = n4m_split_max_min_selection(D, rows, n_train,
                                      out->train_idx, out->test_idx);
    free(D);
    if (st != N4M_OK) {
        n4m_split_result_free(out);
        return st;
    }
    return N4M_OK;
}
