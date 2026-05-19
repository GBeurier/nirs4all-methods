/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Kennard-Stone sample-partitioning splitter — max-min Euclidean distance.
 *
 * Matches nirs4all.operators.splitters.KennardStoneSplitter with the
 * default metric (Euclidean) and no PCA reduction (raw X distances).
 */

#include "kennard_stone.h"

#include <stdlib.h>

c4a_split_ks_state_t* c4a_split_ks_state_new(double test_size) {
    c4a_split_ks_state_t* s = (c4a_split_ks_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    return s;
}

void c4a_split_ks_state_free(c4a_split_ks_state_t* state) {
    free(state);
}

c4a_status_t c4a_split_ks_apply(const c4a_split_ks_state_t* state,
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
    if (n_train < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    double* D = (double*)malloc((size_t)rows * (size_t)rows * sizeof(double));
    if (D == NULL) return C4A_ERR_OUT_OF_MEMORY;
    st = c4a_split_euclidean_dist(X, rows, cols, D);
    if (st != C4A_OK) {
        free(D);
        return st;
    }

    st = c4a_split_result_alloc(out, n_train, n_test);
    if (st != C4A_OK) {
        free(D);
        return st;
    }
    st = c4a_split_max_min_selection(D, rows, n_train,
                                      out->train_idx, out->test_idx);
    free(D);
    if (st != C4A_OK) {
        c4a_split_result_free(out);
        return st;
    }
    return C4A_OK;
}
