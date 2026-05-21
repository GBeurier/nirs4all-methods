/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Robust Standard Normal Variate (RobustSNV / RSNV) — per-row centering
 * and scaling using median and Median Absolute Deviation (MAD).
 *
 * Reference: nirs4all.operators.transforms.scalers.RobustStandardNormalVariate
 *   med   = median(X, axis=1)
 *   mad   = median(|X - med|, axis=1)
 *   scale = k * mad                # k defaults to 1.4826 (Gaussian consistency)
 *   X'    = (X - med) / scale      # scale==0 → divide by 1
 *
 * NumPy's `np.median` uses an in-place partition-based selection (introselect)
 * and averages the two middle values for even-length samples; we implement
 * the equivalent linear-time median using nth_element-style partitioning on
 * a per-row scratch buffer.
 */
#ifndef N4M_CORE_PP_SCATTER_ROBUST_SNV_H
#define N4M_CORE_PP_SCATTER_ROBUST_SNV_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_rnv_state_t n4m_pp_rnv_state_t;

/* `k` is the MAD consistency constant (1.4826 for Gaussian data). */
n4m_pp_rnv_state_t* n4m_pp_rnv_state_new(int with_center, int with_scale, double k);

void n4m_pp_rnv_state_free(n4m_pp_rnv_state_t* state);

n4m_status_t n4m_pp_rnv_apply(const n4m_pp_rnv_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_SCATTER_ROBUST_SNV_H */
