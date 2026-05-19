/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ToAbsorbance — element-wise log10 conversion from reflectance/transmittance
 * to absorbance.
 *
 * Reference: nirs4all.operators.transforms.signal_conversion.ToAbsorbance
 *   if is_percent:
 *       X = X / 100.0
 *   if clip_negative:
 *       X = np.clip(X, epsilon, None)        # negatives -> epsilon
 *   else:
 *       X = np.maximum(X, epsilon)           # negatives -> epsilon, same effect
 *   A = -log10(X)
 *
 * The two clip paths produce the same numerical result for our scalar inputs
 * (a one-sided clamp to >= epsilon); the duplication exists in nirs4all only
 * because numpy.clip with `None` upper bound mirrors numpy.maximum exactly.
 * We keep both flags in the state for ABI symmetry but the kernel takes the
 * one-sided clamp branch regardless.
 *
 * `is_percent` is captured at create time (set by the wrapper from the
 * SignalType enum on the Python side; in C we expose it as a boolean flag).
 * `epsilon` must be > 0.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_TO_ABSORBANCE_H
#define CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_TO_ABSORBANCE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_to_absorbance_state_t c4a_pp_to_absorbance_state_t;

/* Allocate a ToAbsorbance state. `is_percent` divides by 100 first when set
 * to a non-zero value; `epsilon` is the positive clamp target (must be > 0);
 * `clip_negative` is preserved for API symmetry but does not change the
 * kernel result (see header banner). */
c4a_pp_to_absorbance_state_t* c4a_pp_to_absorbance_state_new(
    int is_percent, double epsilon, int clip_negative);

void c4a_pp_to_absorbance_state_free(c4a_pp_to_absorbance_state_t* state);

c4a_status_t c4a_pp_to_absorbance_apply(
    const c4a_pp_to_absorbance_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_TO_ABSORBANCE_H */
