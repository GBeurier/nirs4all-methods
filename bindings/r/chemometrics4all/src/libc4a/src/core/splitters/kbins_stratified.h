/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the KBins-stratified splitter. Public ABI lives in
 * c4a.h §15.
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
#ifndef CHEMOMETRICS4ALL_CORE_SPLITTERS_KBINS_STRATIFIED_H
#define CHEMOMETRICS4ALL_CORE_SPLITTERS_KBINS_STRATIFIED_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Strategy codes for the kbins discretiser are the public enumerators
 * `C4A_SPLIT_KBINS_UNIFORM` / `C4A_SPLIT_KBINS_QUANTILE` from c4a.h §17. */

typedef struct c4a_split_kbins_state_t {
    double   test_size;
    uint64_t seed;
    int32_t  n_bins;
    int32_t  strategy;
} c4a_split_kbins_state_t;

c4a_split_kbins_state_t* c4a_split_kbins_state_new(double test_size,
                                                    uint64_t seed,
                                                    int32_t n_bins,
                                                    int32_t strategy);
void c4a_split_kbins_state_free(c4a_split_kbins_state_t* state);

/* Split a (rows x 1) y vector into train/test indices. X is not used by
 * the algorithm but the public ABI requires its dimensions; pass NULL for
 * X and only y / rows.
 *
 * `y` must be length rows, F64. */
c4a_status_t c4a_split_kbins_apply(const c4a_split_kbins_state_t* state,
                                    const double* y, int64_t rows,
                                    c4a_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_SPLITTERS_KBINS_STRATIFIED_H */
