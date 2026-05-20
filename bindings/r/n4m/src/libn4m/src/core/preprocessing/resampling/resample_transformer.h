/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for ResampleTransformer (linear resample to fixed target
 * column count). The public C ABI lives in n4m.h and is implemented in
 * c_api/c_api_resampling.cpp.
 *
 * Reference: nirs4all.operators.transforms.features.ResampleTransformer.
 *
 * Algorithm (stateless, per row):
 *   if len(x) == num_samples: copy unchanged.
 *   else:
 *     src = linspace(0, 1, len(x))
 *     dst = linspace(0, 1, num_samples)
 *     out_row = interp1d(src, x, kind='linear')(dst)
 *
 * Matches scipy.interp1d 'linear' bit-exact: per-target-sample, locate the
 * bracketing source pair (lo, hi) and compute
 *   slope = (y_hi - y_lo) / (x_hi - x_lo)
 *   y    = slope * (q - x_lo) + y_lo
 * where x_lo / x_hi are the linspace source coords.
 *
 * `num_samples == -1` is the identity sentinel (matches nirs4all's
 * `num_samples is None`): the engine passes the input straight through.
 */
#ifndef N4M_CORE_PP_RESAMPLING_RESAMPLE_TRANSFORMER_H
#define N4M_CORE_PP_RESAMPLING_RESAMPLE_TRANSFORMER_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_resample_state_t n4m_pp_resample_state_t;

/* Allocate a resample state with `num_samples` target columns. Pass -1 for
 * identity (passthrough). num_samples >= 2 is required for actual resampling
 * (linear interp needs at least two anchors). */
n4m_pp_resample_state_t* n4m_pp_resample_state_new(int64_t num_samples);

void n4m_pp_resample_state_free(n4m_pp_resample_state_t* state);

/* Compute output column count from input column count. For num_samples == -1
 * (identity), returns input_cols. Otherwise returns the configured value. */
int64_t n4m_pp_resample_output_cols_helper(const n4m_pp_resample_state_t* state,
                                            int64_t input_cols);

n4m_status_t n4m_pp_resample_apply(const n4m_pp_resample_state_t* state,
                                    const double* X,
                                    int64_t rows, int64_t cols,
                                    int64_t out_cols,
                                    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_RESAMPLING_RESAMPLE_TRANSFORMER_H */
