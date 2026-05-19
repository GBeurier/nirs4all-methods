/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the BinnedStratifiedGroupKFold splitter. Public
 * ABI lives in c4a.h §15.
 *
 * Reference: nirs4all.operators.splitters.BinnedStratifiedGroupKFold (which
 * delegates to KBinsDiscretizer + sklearn StratifiedGroupKFold). The C
 * engine ships its own simplified equivalent so the parity fixture
 * (using NumPy default_rng = PCG64) reproduces byte-exact integer indices.
 *
 * Algorithm:
 *   1. Bin y into n_bins bins (uniform or quantile strategy).
 *   2. Assign each group a single bin label = the bin of its first sample.
 *      (Simplification vs sklearn which uses majority/constraint-aware
 *      assignment; we mirror this in the Python fixture generator.)
 *   3. Per bin, optionally shuffle the groups (Fisher-Yates with PCG64),
 *      then assign them to folds round-robin (group at position p of the
 *      bin's sorted-or-shuffled order goes to fold (p % n_splits)).
 *   4. Per sample, fold_assignment[i] = fold of sample i's group.
 */
#ifndef CHEMOMETRICS4ALL_CORE_SPLITTERS_BINNED_STRAT_GROUP_KFOLD_H
#define CHEMOMETRICS4ALL_CORE_SPLITTERS_BINNED_STRAT_GROUP_KFOLD_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "kbins_stratified.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_split_bsgk_state_t {
    int32_t  n_splits;
    int32_t  n_bins;
    int32_t  strategy;
    int32_t  shuffle;
    uint64_t seed;
} c4a_split_bsgk_state_t;

c4a_split_bsgk_state_t* c4a_split_bsgk_state_new(int32_t n_splits,
                                                   int32_t n_bins,
                                                   int32_t strategy,
                                                   int32_t shuffle,
                                                   uint64_t seed);
void c4a_split_bsgk_state_free(c4a_split_bsgk_state_t* state);

c4a_status_t c4a_split_bsgk_assign(const c4a_split_bsgk_state_t* state,
                                    const double* y, int64_t rows,
                                    const int64_t* groups,
                                    int32_t* fold_assignment);

c4a_status_t c4a_split_bsgk_extract(const int32_t* fold_assignment,
                                     int64_t rows, int32_t fold_idx,
                                     c4a_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_SPLITTERS_BINNED_STRAT_GROUP_KFOLD_H */
