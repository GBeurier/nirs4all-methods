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
#ifndef CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H
#define CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_from_absorbance_state_t c4a_pp_from_absorbance_state_t;

/* Allocate a FromAbsorbance state. `is_percent` rescales the result by 100. */
c4a_pp_from_absorbance_state_t* c4a_pp_from_absorbance_state_new(int is_percent);

void c4a_pp_from_absorbance_state_free(c4a_pp_from_absorbance_state_t* state);

c4a_status_t c4a_pp_from_absorbance_apply(
    const c4a_pp_from_absorbance_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FROM_ABSORBANCE_H */
