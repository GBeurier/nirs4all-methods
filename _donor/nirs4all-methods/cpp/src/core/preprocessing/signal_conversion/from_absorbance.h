/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FromAbsorbance — element-wise inverse of ToAbsorbance.
 *
 * Reference: nirs4all.operators.transforms.signal_conversion.FromAbsorbance
 *   R_or_T = 10**(-X)
 *   if is_percent:
 *       R_or_T = R_or_T * 100.0
 *
 * `is_percent` reflects the `target_type`: when the target signal is
 * `reflectance%` or `transmittance%` the post-power result is rescaled by
 * 100. The operator carries no other parameters.
 */
#ifndef N4M_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H
#define N4M_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_from_absorbance_state_t n4m_pp_from_absorbance_state_t;

/* Allocate a FromAbsorbance state. `is_percent` rescales the result by 100. */
n4m_pp_from_absorbance_state_t* n4m_pp_from_absorbance_state_new(int is_percent);

void n4m_pp_from_absorbance_state_free(n4m_pp_from_absorbance_state_t* state);

n4m_status_t n4m_pp_from_absorbance_apply(
    const n4m_pp_from_absorbance_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H */
