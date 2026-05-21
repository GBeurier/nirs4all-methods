/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SavitzkyGolay operator. Public C ABI lives in
 * n4m.h (§10, Phase 4).
 *
 * Algorithm: precompute `window_length` convolution coefficients at create
 * time via a Vandermonde-QR pseudo-inverse, then apply 1-D cross-correlation
 * along axis=1 for every row of X. The five SciPy `mode` values are honoured
 * via the n4m_pad_resolve_index helper (mirror, constant, nearest, wrap),
 * plus a polynomial-fit-at-the-edges path for `interp` that matches
 * `scipy.signal._savitzky_golay._fit_edge` exactly.
 *
 * Lifecycle: stateless. `_create` allocates the coefficient buffer; `_apply`
 * is a pure function of (state, X, out); `_free` releases everything.
 */
#ifndef N4M_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H
#define N4M_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_savgol_state_t n4m_pp_savgol_state_t;

/* Allocate the SG state. Validates window_length >= 1 (odd), polyorder
 * in [0, window_length), deriv >= 0, delta != 0. Returns NULL on invalid
 * arguments or allocation failure. The mode argument is validated by the
 * C-API wrapper; the engine assumes a valid value. */
n4m_pp_savgol_state_t* n4m_pp_savgol_state_new(int32_t window_length,
                                                int32_t polyorder,
                                                int32_t deriv,
                                                double delta,
                                                n4m_pp_savgol_mode_t mode,
                                                double cval);

/* NULL-safe. */
void n4m_pp_savgol_state_free(n4m_pp_savgol_state_t* state);

/* Apply the SG filter row-by-row along axis=1. Input and output are
 * row-major (rows x cols). They may alias (in-place is supported by
 * staging each row in a local scratch buffer).
 *
 * Returns N4M_ERR_INVALID_ARGUMENT when cols < window_length AND
 * mode != interp; interp mode requires cols >= window_length too. */
n4m_status_t n4m_pp_savgol_state_apply(const n4m_pp_savgol_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols,
                                        double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H */
