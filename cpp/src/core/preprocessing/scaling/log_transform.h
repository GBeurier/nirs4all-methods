/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LogTransform — element-wise logarithm with optional safety offset.
 *
 * Reference: nirs4all.operators.transforms.nirs.LogTransform
 *   X_temp = X + offset
 *   if auto_offset and min(X_temp) <= 0:
 *       fitted_offset = offset + (min_value - min(X_temp))
 *   else:
 *       fitted_offset = offset
 *   X' = X + fitted_offset                # ← uses ORIGINAL X, not X_temp
 *   X' = where(X' <= 0, min_value, X')
 *   X' = log(X') / log(base)              # natural log if base == e
 *
 * The fit/transform split in nirs4all evaluates min() on the fit data, then
 * applies that fitted offset to subsequent calls. Since our parity fixture
 * captures `fit_transform(X)`, the C engine computes the fitted offset
 * inline from the same X.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCALING_LOG_TRANSFORM_H
#define CHEMOMETRICS4ALL_CORE_PP_SCALING_LOG_TRANSFORM_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_log_state_t c4a_pp_log_state_t;

/* `base` is the log base; pass 0.0 (or e) to select the natural logarithm.
 * `offset` is the manual additive offset. `auto_offset` enables the
 * fit-time safety offset that keeps the post-offset minimum >= `min_value`.
 */
c4a_pp_log_state_t* c4a_pp_log_state_new(double base, double offset,
                                          int auto_offset, double min_value);

void c4a_pp_log_state_free(c4a_pp_log_state_t* state);

c4a_status_t c4a_pp_log_apply(const c4a_pp_log_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCALING_LOG_TRANSFORM_H */
