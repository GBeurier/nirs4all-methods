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
#ifndef N4M_CORE_PP_SCALING_SIMPLE_SCALE_H
#define N4M_CORE_PP_SCALING_SIMPLE_SCALE_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_simple_scale_state_t n4m_pp_simple_scale_state_t;

n4m_pp_simple_scale_state_t* n4m_pp_simple_scale_state_new(void);

void n4m_pp_simple_scale_state_free(n4m_pp_simple_scale_state_t* state);

n4m_status_t n4m_pp_simple_scale_apply(const n4m_pp_simple_scale_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
                                       double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCALING_SIMPLE_SCALE_H */
