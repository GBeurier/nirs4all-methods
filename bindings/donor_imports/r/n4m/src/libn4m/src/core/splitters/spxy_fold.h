/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SPXY K-Fold splitter (and pure Kennard-Stone
 * K-Fold via `use_y == 0`). Public ABI lives in n4m.h §15.
 *
 * Reference: nirs4all.operators.splitters.SPXYFold.
 *
 * Per-fold output: for each requested fold index, returns the train and
 * test sample indices where the test set is the samples assigned to that
 * fold by the alternating max-min algorithm.
 */
#ifndef N4M_CORE_SPLITTERS_SPXY_FOLD_H
#define N4M_CORE_SPLITTERS_SPXY_FOLD_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_split_spxy_fold_state_t {
    int32_t n_splits;
    int32_t use_y;   /* 0 = pure Kennard-Stone (X only), 1 = SPXY (X+Y), 2 = SPXY with hamming Y */
} n4m_split_spxy_fold_state_t;

n4m_split_spxy_fold_state_t* n4m_split_spxy_fold_state_new(int32_t n_splits,
                                                            int32_t use_y);
void n4m_split_spxy_fold_state_free(n4m_split_spxy_fold_state_t* s);

/* Compute the fold assignment vector (length rows) for the configured
 * n_splits / use_y on (X, Y). Output buffer must be provided by caller.
 * X must be (rows x cols_x); when use_y != 0 Y must be (rows x cols_y).
 *
 * fold_assignment[i] in [0, n_splits) on success. */
n4m_status_t n4m_split_spxy_fold_assign(const n4m_split_spxy_fold_state_t* s,
                                         const double* X, int64_t rows,
                                         int64_t cols_x,
                                         const double* Y, int64_t cols_y,
                                         int32_t* fold_assignment);

/* Produce the train/test split for one fold given a precomputed assignment
 * vector. The test set is the samples with fold_assignment[i] == fold_idx
 * (in ascending sample-index order); train is the complement. */
n4m_status_t n4m_split_spxy_fold_extract(const int32_t* fold_assignment,
                                          int64_t rows, int32_t fold_idx,
                                          n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_SPXY_FOLD_H */
