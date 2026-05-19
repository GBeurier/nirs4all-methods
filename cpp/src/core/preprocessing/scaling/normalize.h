/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Normalize — column-wise (axis=0) spectral normalization.
 *
 * Reference: nirs4all.operators.transforms.scalers.Normalize
 *
 * Two modes selected by the (min, max) feature range:
 *   - linalg-norm mode (default feature_range == (-1, 1)):
 *       norm_j = ||X[:, j]||_2     # L2 norm of each column
 *       X'[:, j] = X[:, j] / norm_j
 *   - user-defined range mode (any other feature_range):
 *       min_j  = min(X[:, j])
 *       max_j  = max(X[:, j])
 *       f_j    = (imax - imin) / (max_j - min_j)
 *       X'[:, j] = imin + f_j * (X[:, j] - min_j)
 *
 * The state stores the parameters; the apply step accepts the column
 * statistics inline (no fit/transform split — this is a stateless operator
 * because nirs4all's `fit` is itself a pure function of the input data,
 * and the same data is then used in `transform`).
 *
 * NOTE: although nirs4all formally splits fit and transform, the parity
 * fixture captures only `fit_transform(X)` so the C side computes the
 * stats and applies the transform in a single call.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCALING_NORMALIZE_H
#define CHEMOMETRICS4ALL_CORE_PP_SCALING_NORMALIZE_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_normalize_state_t c4a_pp_normalize_state_t;

/* Allocate a normalize state. `feature_min` / `feature_max` define the
 * desired range. The default (-1, 1) selects linalg-norm mode; any other
 * pair selects user-defined-range mode. */
c4a_pp_normalize_state_t* c4a_pp_normalize_state_new(double feature_min,
                                                     double feature_max);

void c4a_pp_normalize_state_free(c4a_pp_normalize_state_t* state);

c4a_status_t c4a_pp_normalize_apply_params(double feature_min,
                                           double feature_max,
                                           const double* X, int64_t rows,
                                           int64_t cols, double* out);

c4a_status_t c4a_pp_normalize_apply(const c4a_pp_normalize_state_t* state,
                                    const double* X, int64_t rows, int64_t cols,
                                    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCALING_NORMALIZE_H */
