/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the KBins-stratified splitter. Public ABI lives in
 * n4m.h §15.
 *
 * Reference: nirs4all.operators.splitters.KBinsStratifiedSplitter (which
 * delegates to KBinsDiscretizer + sklearn StratifiedShuffleSplit). The C
 * engine mirrors that stack, including NumPy legacy RandomState(MT19937)
 * shuffling, so byte-exact integer indices match the external reference.
 *
 * Strategies:
 *   * uniform  — equal-width bins from y.min() to y.max().
 *   * quantile — equal-frequency bins (quantile-based edges over a
 *                sorted-y reference).
 *
 * Stratified shuffle split: allocate per-bin train/test counts with sklearn's
 * `_approximate_mode`, shuffle each bin with RandomState.permutation, then
 * apply the final train/test permutations. Output order intentionally matches
 * sklearn; it is not sorted.
 */
#ifndef N4M_CORE_SPLITTERS_KBINS_STRATIFIED_H
#define N4M_CORE_SPLITTERS_KBINS_STRATIFIED_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Strategy codes for the kbins discretiser are the public enumerators
 * `N4M_SPLIT_KBINS_UNIFORM` / `N4M_SPLIT_KBINS_QUANTILE` from n4m.h §17. */

typedef struct n4m_split_kbins_state_t {
    double   test_size;
    uint64_t seed;
    int32_t  n_bins;
    int32_t  strategy;
} n4m_split_kbins_state_t;

n4m_split_kbins_state_t* n4m_split_kbins_state_new(double test_size,
                                                    uint64_t seed,
                                                    int32_t n_bins,
                                                    int32_t strategy);
void n4m_split_kbins_state_free(n4m_split_kbins_state_t* state);

/* Split a (rows x 1) y vector into train/test indices. X is not used by
 * the algorithm but the public ABI requires its dimensions; pass NULL for
 * X and only y / rows.
 *
 * `y` must be length rows, F64. */
n4m_status_t n4m_split_kbins_apply(const n4m_split_kbins_state_t* state,
                                    const double* y, int64_t rows,
                                    n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_KBINS_STRATIFIED_H */
