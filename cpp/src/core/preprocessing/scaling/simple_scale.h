/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SimpleScale — per-column (axis=0) min-max scaling to the [0, 1] range.
 *
 * Reference: nirs4all.operators.transforms.scalers.SimpleScale
 *   min_ = np.min(X, axis=0)
 *   max_ = np.max(X, axis=0)
 *   X'   = (X - min_) / (max_ - min_)
 *
 * The state carries no tunable parameters; the operator is fully data-driven.
 * We keep a placeholder state struct for API symmetry with the rest of the
 * preprocessing surface, so bindings can treat all stateless operators
 * uniformly (create / apply / destroy).
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCALING_SIMPLE_SCALE_H
#define CHEMOMETRICS4ALL_CORE_PP_SCALING_SIMPLE_SCALE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_simple_scale_state_t c4a_pp_simple_scale_state_t;

c4a_pp_simple_scale_state_t* c4a_pp_simple_scale_state_new(void);

void c4a_pp_simple_scale_state_free(c4a_pp_simple_scale_state_t* state);

c4a_status_t c4a_pp_simple_scale_apply(const c4a_pp_simple_scale_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCALING_SIMPLE_SCALE_H */
