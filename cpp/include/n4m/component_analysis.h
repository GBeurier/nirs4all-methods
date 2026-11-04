/* SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later */
#ifndef N4M_COMPONENT_ANALYSIS_H
#define N4M_COMPONENT_ANALYSIS_H
#include "n4m/n4m.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Compute dimensionless loss/baseline ratios. Values must be finite and >=0;
 * baselines finite and >0. Overflow is refused. All output is unchanged on
 * error, including when input and output alias. No allocation crosses the ABI. */
N4M_API n4m_status_t n4m_component_normalize_loss_v1(
    const double* values, const double* baselines, int64_t n, double* output);
#ifdef __cplusplus
}
#endif
#endif
