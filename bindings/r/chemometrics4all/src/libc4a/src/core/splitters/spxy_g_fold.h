/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the group-aware SPXY K-Fold splitter. Public ABI
 * lives in c4a.h §15.
 *
 * Reference: nirs4all.operators.splitters.SPXYGFold (n_splits >= 2 path).
 *
 * Algorithm:
 *   1. Aggregate samples by group label (groups[i] is an integer; first
 *      occurrence defines the group's position in the representative
 *      array). Representatives are either column-wise mean or median of
 *      the group's X (and Y when use_y != 0).
 *   2. Run the same alternating max-min fold assignment as SPXYFold on the
 *      n_groups x n_groups distance matrix.
 *   3. Expand the per-group fold assignment back to per-sample by mapping
 *      each sample to its group's fold index.
 *
 * NOTE: We require numeric (int64_t) group labels in the public ABI. The
 * Python reference supports arbitrary hashable group keys; we restrict to
 * int64_t here because (a) the C ABI doesn't carry generic hashable types
 * and (b) the canonical NIRS pipelines map group identifiers to int64
 * indices upstream of the splitter.
 */
#ifndef CHEMOMETRICS4ALL_CORE_SPLITTERS_SPXY_G_FOLD_H
#define CHEMOMETRICS4ALL_CORE_SPLITTERS_SPXY_G_FOLD_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_split_spxy_g_fold_state_t {
    int32_t n_splits;
    int32_t use_y;        /* 0 = X-only, 1 = SPXY euclidean Y, 2 = SPXY hamming Y */
    int32_t aggregation;  /* 0 = mean, 1 = median */
} c4a_split_spxy_g_fold_state_t;

c4a_split_spxy_g_fold_state_t* c4a_split_spxy_g_fold_state_new(int32_t n_splits,
                                                                int32_t use_y,
                                                                int32_t aggregation);
void c4a_split_spxy_g_fold_state_free(c4a_split_spxy_g_fold_state_t* s);

/* Compute the per-sample fold assignment for the configured K-fold and
 * grouping. groups[i] is the group label for sample i; samples sharing the
 * same label always end up in the same fold.
 *
 * fold_assignment must be a caller-owned int32_t buffer of length rows. */
c4a_status_t c4a_split_spxy_g_fold_assign(const c4a_split_spxy_g_fold_state_t* s,
                                           const double* X, int64_t rows,
                                           int64_t cols_x,
                                           const double* Y, int64_t cols_y,
                                           const int64_t* groups,
                                           int32_t* fold_assignment);

/* Convenience: produce the train/test split for one fold given a
 * precomputed assignment vector. Identical to spxy_fold_extract. */
c4a_status_t c4a_split_spxy_g_fold_extract(const int32_t* fold_assignment,
                                            int64_t rows, int32_t fold_idx,
                                            c4a_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_SPLITTERS_SPXY_G_FOLD_H */
