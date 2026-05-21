/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for CropTransformer (column-slice [start, end)). The
 * public C ABI lives in n4m.h and is implemented in
 * c_api/c_api_resampling.cpp.
 *
 * Reference: nirs4all.operators.transforms.features.CropTransformer.
 *
 * Algorithm (stateless):
 *   end_eff = (end < 0 || end > cols) ? cols : end
 *   out[i, k] = X[i, start + k]   for k in [0, end_eff - start)
 *
 * `end == -1` means "clamp to cols" (mirrors nirs4all's `end is None` flag).
 */
#ifndef N4M_CORE_PP_RESAMPLING_CROP_H
#define N4M_CORE_PP_RESAMPLING_CROP_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_crop_state_t n4m_pp_crop_state_t;

/* Allocate a crop state. `start >= 0`. `end == -1` selects "to end of row";
 * any `end > start` slices [start, end). `end > cols` is clamped at
 * transform time. */
n4m_pp_crop_state_t* n4m_pp_crop_state_new(int64_t start, int64_t end);

void n4m_pp_crop_state_free(n4m_pp_crop_state_t* state);

/* Compute the output column count for a given input column count. Returns 0
 * for degenerate cases (start >= cols, etc.). */
int64_t n4m_pp_crop_output_cols_helper(const n4m_pp_crop_state_t* state,
                                        int64_t input_cols);

n4m_status_t n4m_pp_crop_apply(const n4m_pp_crop_state_t* state,
                                const double* X,
                                int64_t rows, int64_t cols,
                                int64_t out_cols,
                                double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_RESAMPLING_CROP_H */
