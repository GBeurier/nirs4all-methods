/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * KubelkaMunk — diffuse-reflectance transformation.
 *
 * Reference: nirs4all.operators.transforms.signal_conversion.KubelkaMunk
 *   if is_percent:
 *       X = X / 100.0
 *   R = np.clip(X, epsilon, 1.0 - epsilon)
 *   F_R = (1.0 - R) ** 2 / (2.0 * R)
 *
 * `is_percent` reflects the `source_type` (reflectance vs reflectance%).
 * `epsilon` is the symmetric clamp target (R is forced into
 * [epsilon, 1 - epsilon]); must satisfy 0 < epsilon < 0.5 to keep the
 * interval non-empty.
 */
#ifndef N4M_CORE_PP_SIGNAL_CONVERSION_KUBELKA_MUNK_H
#define N4M_CORE_PP_SIGNAL_CONVERSION_KUBELKA_MUNK_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_kubelka_munk_state_t n4m_pp_kubelka_munk_state_t;

n4m_pp_kubelka_munk_state_t* n4m_pp_kubelka_munk_state_new(
    int is_percent, double epsilon);

void n4m_pp_kubelka_munk_state_free(n4m_pp_kubelka_munk_state_t* state);

n4m_status_t n4m_pp_kubelka_munk_apply(
    const n4m_pp_kubelka_munk_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_SIGNAL_CONVERSION_KUBELKA_MUNK_H */
