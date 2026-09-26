/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SimpleScale — per-column (axis=0) min-max scaling to the [0, 1] range.
 *
 * Reference: nirs4all.operators.transforms.scalers.SimpleScale
 *   min_ = np.min(X, axis=0)
 *   max_ = np.max(X, axis=0)
 *   X'   = (X - min_) / (max_ - min_)
 *
 * n4m_pp_simple_scale_apply (the public, stateless transform) computes the
 * column min / max of the batch it transforms; its state carries no tunable
 * parameters and is a placeholder for API symmetry. The fitted state
 * (n4m_pp_simple_scale_fit_state_*) learns the column min / max at fit and
 * applies them to new rows (the transformer role); on the training rows both
 * produce bit-identical results.
 */
#ifndef N4M_CORE_PP_SCALING_SIMPLE_SCALE_H
#define N4M_CORE_PP_SCALING_SIMPLE_SCALE_H

#include "n4m/n4m.h"
#include "core/estimator/state_io.h"

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

typedef struct n4m_pp_simple_scale_fit_state_t n4m_pp_simple_scale_fit_state_t;

n4m_pp_simple_scale_fit_state_t* n4m_pp_simple_scale_fit_state_new(void);

void n4m_pp_simple_scale_fit_state_free(n4m_pp_simple_scale_fit_state_t* state);

/* Learn the column min / max of X (rows >= 1). */
n4m_status_t n4m_pp_simple_scale_fit_state_fit(n4m_pp_simple_scale_fit_state_t* state,
                                               const double* X, int64_t rows,
                                               int64_t cols);

/* Apply the learned min / max to X (cols must match the fit). */
n4m_status_t n4m_pp_simple_scale_fit_state_apply(
    const n4m_pp_simple_scale_fit_state_t* state, const double* X, int64_t rows,
    int64_t cols, double* out);

/* Fitted-state serialization (core/estimator/state_io.h): the column min and
 * max. */
n4m_status_t n4m_pp_simple_scale_fit_state_save(
    const n4m_pp_simple_scale_fit_state_t* state, n4m_state_writer_t* w);
n4m_status_t n4m_pp_simple_scale_fit_state_load(
    n4m_pp_simple_scale_fit_state_t* state, n4m_state_reader_t* r, int64_t n_features);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCALING_SIMPLE_SCALE_H */
