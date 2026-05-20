/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SPXY (Sample set Partitioning based on
 * X-Y joint distance) splitter. Public ABI lives in n4m.h §15.
 *
 * Reference: nirs4all.operators.splitters.SPXYSplitter
 *
 * Algorithm:
 *   D = D_X / max(D_X) + D_Y / max(D_Y)
 * then max-min selection as in Kennard-Stone.
 */
#ifndef N4M_CORE_SPLITTERS_SPXY_H
#define N4M_CORE_SPLITTERS_SPXY_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "split_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_split_spxy_state_t {
    double test_size;
} n4m_split_spxy_state_t;

n4m_split_spxy_state_t* n4m_split_spxy_state_new(double test_size);
void                    n4m_split_spxy_state_free(n4m_split_spxy_state_t* s);

/* Run SPXY selection on a (rows x cols_x) X matrix and a (rows x cols_y)
 * y matrix (both row-major F64). cols_y is typically 1 for regression and
 * may be > 1 for multi-output targets. */
n4m_status_t n4m_split_spxy_apply(const n4m_split_spxy_state_t* state,
                                   const double* X, int64_t rows,
                                   int64_t cols_x,
                                   const double* Y, int64_t cols_y,
                                   n4m_split_result_t* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_SPLITTERS_SPXY_H */
