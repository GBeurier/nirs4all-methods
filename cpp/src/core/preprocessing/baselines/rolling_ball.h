/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * RollingBall — Kneen & Annegarn 1996 morphological baseline.
 *
 * For each row y of length n:
 *   1. min_filter of half-window R       -> e
 *   2. max_filter of e with half-window R -> z
 *   3. optional moving-average smoothing of z with half-window S
 *   4. out := y - z
 *
 * Edges use a centred window clipped to [0, n-1] (no reflection / padding).
 * Internal parity fixture: parity/python_generator/src/n4m_parity_pybaselines_ref/rolling_ball.py
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_ROLLING_BALL_H
#define N4M_CORE_PREPROCESSING_BASELINES_ROLLING_BALL_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_rolling_ball_state_t n4m_pp_rolling_ball_state_t;

n4m_pp_rolling_ball_state_t* n4m_pp_rolling_ball_state_new(int32_t half_window,
                                                             int32_t smooth_half_window);
void n4m_pp_rolling_ball_state_free(n4m_pp_rolling_ball_state_t* state);

n4m_status_t n4m_pp_rolling_ball_state_apply(
    const n4m_pp_rolling_ball_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_ROLLING_BALL_H */
