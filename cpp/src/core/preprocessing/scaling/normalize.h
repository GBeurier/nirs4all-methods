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
 * Two entry points:
 *   - n4m_pp_normalize_apply_params (the public, stateless transform)
 *     computes the column statistics of the batch it transforms, as
 *     nirs4all's fit_transform(X) does;
 *   - the fitted state (n4m_pp_normalize_state_*) learns the column
 *     statistics at fit and applies them to new rows, so held-out rows are
 *     scaled with the training statistics (the transformer role). On the
 *     training rows both produce bit-identical results.
 */
#ifndef N4M_CORE_PP_SCALING_NORMALIZE_H
#define N4M_CORE_PP_SCALING_NORMALIZE_H

#include "n4m/n4m.h"
#include "core/estimator/state_io.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_normalize_state_t n4m_pp_normalize_state_t;

/* Allocate an unfitted normalize state. `feature_min` / `feature_max`
 * define the desired range. The default (-1, 1) selects linalg-norm mode;
 * any other pair selects user-defined-range mode. */
n4m_pp_normalize_state_t* n4m_pp_normalize_state_new(double feature_min,
                                                     double feature_max);

void n4m_pp_normalize_state_free(n4m_pp_normalize_state_t* state);

/* Learn the column statistics of X (rows >= 1): min and max in range mode,
 * the L2 norm in linalg-norm mode. */
n4m_status_t n4m_pp_normalize_state_fit(n4m_pp_normalize_state_t* state,
                                        const double* X, int64_t rows,
                                        int64_t cols);

/* Apply the learned statistics to X (cols must match the fit). */
n4m_status_t n4m_pp_normalize_apply(const n4m_pp_normalize_state_t* state,
                                    const double* X, int64_t rows, int64_t cols,
                                    double* out);

/* Stateless: statistics of the batch X itself. */
n4m_status_t n4m_pp_normalize_apply_params(double feature_min,
                                           double feature_max,
                                           const double* X, int64_t rows,
                                           int64_t cols, double* out);

/* Fitted-state serialization (core/estimator/state_io.h): the column min
 * and max (range mode) or the column norms (linalg-norm mode). */
n4m_status_t n4m_pp_normalize_state_save(const n4m_pp_normalize_state_t* state,
                                         n4m_state_writer_t* w);
n4m_status_t n4m_pp_normalize_state_load(n4m_pp_normalize_state_t* state,
                                         n4m_state_reader_t* r, int64_t n_features);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCALING_NORMALIZE_H */
