/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Area Normalization — per-row division by an area metric.
 *
 * Reference: nirs4all.operators.transforms.nirs.AreaNormalization
 *   for each row i:
 *     - method='sum'    : area = sum(X[i])
 *     - method='abs_sum': area = sum(|X[i]|)
 *     - method='trapz'  : area = trapezoidal integral (unit spacing) =
 *                                 (X[i, 0] + X[i, -1]) / 2 + sum(X[i, 1:-1])
 *   if |area| < 1e-10 → area = 1.0
 *   X[i] /= area
 */
#ifndef N4M_CORE_PP_SCATTER_AREA_NORMALIZATION_H
#define N4M_CORE_PP_SCATTER_AREA_NORMALIZATION_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* n4m_pp_area_method_t is declared in n4m/n4m.h (public). */

typedef struct n4m_pp_area_state_t n4m_pp_area_state_t;

n4m_pp_area_state_t* n4m_pp_area_state_new(n4m_pp_area_method_t method);

void n4m_pp_area_state_free(n4m_pp_area_state_t* state);

n4m_status_t n4m_pp_area_apply_method(n4m_pp_area_method_t method,
                                      const double* X, int64_t rows,
                                      int64_t cols, double* out);

n4m_status_t n4m_pp_area_apply(const n4m_pp_area_state_t* state,
                               const double* X, int64_t rows, int64_t cols,
                               double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCATTER_AREA_NORMALIZATION_H */
