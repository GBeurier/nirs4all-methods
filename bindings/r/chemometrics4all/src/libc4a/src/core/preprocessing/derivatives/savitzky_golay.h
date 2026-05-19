/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the SavitzkyGolay operator. Public C ABI lives in
 * c4a.h (§10, Phase 4).
 *
 * Algorithm: precompute `window_length` convolution coefficients at create
 * time via a Vandermonde-QR pseudo-inverse, then apply 1-D cross-correlation
 * along axis=1 for every row of X. The five SciPy `mode` values are honoured
 * via the c4a_pad_resolve_index helper (mirror, constant, nearest, wrap),
 * plus a polynomial-fit-at-the-edges path for `interp` that matches
 * `scipy.signal._savitzky_golay._fit_edge` exactly.
 *
 * Lifecycle: stateless. `_create` allocates the coefficient buffer; `_apply`
 * is a pure function of (state, X, out); `_free` releases everything.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H
#define CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_savgol_state_t c4a_pp_savgol_state_t;

/* Allocate the SG state. Validates window_length >= 1 (odd), polyorder
 * in [0, window_length), deriv >= 0, delta != 0. Returns NULL on invalid
 * arguments or allocation failure. The mode argument is validated by the
 * C-API wrapper; the engine assumes a valid value. */
c4a_pp_savgol_state_t* c4a_pp_savgol_state_new(int32_t window_length,
                                                int32_t polyorder,
                                                int32_t deriv,
                                                double delta,
                                                c4a_pp_savgol_mode_t mode,
                                                double cval);

/* NULL-safe. */
void c4a_pp_savgol_state_free(c4a_pp_savgol_state_t* state);

/* Apply the SG filter row-by-row along axis=1. Input and output are
 * row-major (rows x cols). They may alias (in-place is supported by
 * staging each row in a local scratch buffer).
 *
 * Returns C4A_ERR_INVALID_ARGUMENT when cols < window_length AND
 * mode != interp; interp mode requires cols >= window_length too. */
c4a_status_t c4a_pp_savgol_state_apply(const c4a_pp_savgol_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols,
                                        double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_DERIVATIVES_SAVITZKY_GOLAY_H */
