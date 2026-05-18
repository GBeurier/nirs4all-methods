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
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCATTER_AREA_NORMALIZATION_H
#define CHEMOMETRICS4ALL_CORE_PP_SCATTER_AREA_NORMALIZATION_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* c4a_pp_area_method_t is declared in chemometrics4all/c4a.h (public). */

typedef struct c4a_pp_area_state_t c4a_pp_area_state_t;

c4a_pp_area_state_t* c4a_pp_area_state_new(c4a_pp_area_method_t method);

void c4a_pp_area_state_free(c4a_pp_area_state_t* state);

c4a_status_t c4a_pp_area_apply(const c4a_pp_area_state_t* state,
                               const double* X, int64_t rows, int64_t cols,
                               double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCATTER_AREA_NORMALIZATION_H */
