/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SystematicCircularSplitter. Public ABI lives
 * in c4a.h §15.
 *
 * Reference: nirs4all.operators.splitters.SystematicCircularSplitter.
 *
 * Algorithm:
 *   1. ordered_idx = argsort(y, axis=0).
 *   2. offset = floor(uniform01 * n_samples) drawn from PCG64(seed).
 *      (The Python reference uses random.randint(0, n_samples) which
 *      yields a value in [0, n_samples] inclusive; we use the half-open
 *      interval which is what np.roll uses modulo. The fixture generator
 *      mirrors the same draw to keep parity.)
 *   3. rotated_idx = np.roll(ordered_idx, offset).
 *   4. step = n_samples / n_train; indices = [round(step * i) for i in
 *      range(n_train)]; index_train = rotated_idx[indices];
 *      index_test = np.delete(rotated_idx, indices).
 *   5. Output train and test as taken (NOT sorted — the test set keeps the
 *      rotated order; we then sort both for stable output).
 *
 * The Python reference fixture generator uses the same PCG64 draw and
 * np.roll / np.delete sequence so the integer indices agree.
 */
#ifndef CHEMOMETRICS4ALL_CORE_SPLITTERS_SYSTEMATIC_CIRCULAR_H
#define CHEMOMETRICS4ALL_CORE_SPLITTERS_SYSTEMATIC_CIRCULAR_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_split_syscirc_state_t {
    double   test_size;
    uint64_t seed;
} c4a_split_syscirc_state_t;

c4a_split_syscirc_state_t* c4a_split_syscirc_state_new(double test_size,
                                                        uint64_t seed);
void c4a_split_syscirc_state_free(c4a_split_syscirc_state_t* state);

c4a_status_t c4a_split_syscirc_apply(const c4a_split_syscirc_state_t* state,
                                      const double* y, int64_t rows,
                                      c4a_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_SPLITTERS_SYSTEMATIC_CIRCULAR_H */
